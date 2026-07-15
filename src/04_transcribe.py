"""Этап 4: Whisper — только по хитам и видео без субтитров.

Вход:  data/scored/videos.csv (is_hit=True или subs_missing=True)
Выход: data/transcripts/{video_id}.txt + {video_id}.json (сегменты)

Ускорение GPU:
- батчевый инференс (BatchedInferencePipeline) — 3-4x на видеокарте;
- аудио следующих видео скачивается в фоне, пока GPU занят текущим.
Аудио удаляется после транскрибации.

Проверка казахского (ТЗ, этап 4):
    python src/04_transcribe.py --language kk --video <video_id>
и честно оценить результат глазами. Если качество не годится —
записать в отчёт как ограничение, смотреть модели ISSAI (Soyle).
"""

import argparse
import json
import os
import site
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from utils import (DATA_DIR, load_settings, require_ffmpeg, setup_logger,
                   video_url, ytdlp_run)

SCORED_CSV = DATA_DIR / "scored" / "videos.csv"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

PREFETCH_AHEAD = 3  # сколько аудио качать наперёд

_cache = {}


def _add_nvidia_dll_dirs() -> None:
    """Windows: ctranslate2 ищет cuBLAS/cuDNN DLL в PATH, а pip-пакеты
    nvidia-cublas-cu12 / nvidia-cudnn-cu12 кладут их в site-packages —
    подключаем эти папки, иначе CUDA не поднимется."""
    if os.name != "nt":
        return
    for sp in site.getsitepackages():
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
            p = Path(sp) / sub
            if p.exists():
                os.add_dll_directory(str(p))
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")


def get_model(settings):
    """Ленивая загрузка faster-whisper (модель тяжёлая, грузим один раз)."""
    if "model" in _cache:
        return _cache["model"]
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit(
            "faster-whisper не установлен: pip install faster-whisper\n"
            "Без GPU поставь в settings.yaml transcribe.model: medium."
        )
    _add_nvidia_dll_dirs()
    device = settings["device"]
    compute = settings["compute_type"]
    if device == "auto":
        # faster-whisper работает на CTranslate2 (не на torch),
        # поэтому и CUDA проверяем через ctranslate2
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        # int8_float16: large-v3 занимает ~3 ГБ VRAM и влезает в 4-гиговые
        # карты (RTX 3050 и т.п.); float16 потребовал бы ~5 ГБ
        compute = "int8_float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(settings["model"], device=device, compute_type=compute)
    except Exception as e:
        # типичный случай на Windows: CUDA видна, но нет cuDNN/cuBLAS DLL
        print(f"Не удалось поднять Whisper на {device} ({e}), откатываюсь на CPU/int8")
        device = "cpu"
        model = WhisperModel(settings["model"], device="cpu", compute_type="int8")
    _cache["model"] = model
    _cache["device"] = device
    return model


def get_batched(settings, logger):
    """Батчевый пайплайн поверх модели — 3-4x на GPU. None = недоступен."""
    if "batched" in _cache:
        return _cache["batched"]
    model = get_model(settings)
    batched = None
    if _cache.get("device") == "cuda" and settings.get("batch_size", 0) > 1:
        try:
            from faster_whisper import BatchedInferencePipeline
            batched = BatchedInferencePipeline(model=model)
            logger.info("Батчевый инференс включён (batch_size=%s)",
                        settings["batch_size"])
        except Exception as e:
            logger.warning("Батчевый инференс недоступен (%s), обычный режим", e)
    _cache["batched"] = batched
    return batched


def download_audio(video_id: str, logger) -> "Path | None":
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    path = AUDIO_DIR / f"{video_id}.mp3"
    if path.exists():
        return path
    template = str(AUDIO_DIR / f"{video_id}.%(ext)s")
    ytdlp_run(
        ["-x", "--audio-format", "mp3", "--audio-quality", "5",
         "-o", template, video_url(video_id)],
        logger=logger,
    )
    return path if path.exists() else None


def _glossary_prompt(language, settings):
    # Глоссарий — только при явном русском: русский initial_prompt на
    # английском/казахском аудио сбивает модель.
    if language != "ru":
        return None
    glossary = []
    for terms in settings["glossary"].values():
        glossary.extend(terms)
    return "Термины: " + ", ".join(glossary)


def _run_whisper(audio, language, settings, logger):
    """Транскрибация: батчево если можно, иначе обычно. Возвращает (segments, info)."""
    initial_prompt = _glossary_prompt(language, settings)
    batched = get_batched(settings, logger)
    if batched is not None:
        kwargs = dict(language=language, batch_size=settings["batch_size"])
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        try:
            return batched.transcribe(str(audio), **kwargs)
        except TypeError:
            kwargs.pop("initial_prompt", None)  # старая версия без параметра
            return batched.transcribe(str(audio), **kwargs)
        except Exception as e:
            logger.warning("батч упал (%s), это видео — в обычном режиме", e)
            if "out of memory" in str(e).lower():
                _cache["batched"] = None  # больше не пытаемся батчевать
                logger.warning("похоже OOM: батч отключён; можно снизить "
                               "transcribe.batch_size в settings.yaml")
    model = get_model(settings)
    return model.transcribe(str(audio), language=language, vad_filter=True,
                            initial_prompt=initial_prompt)


def transcribe_audio(video_id: str, audio: Path, language, settings, logger) -> bool:
    txt_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
    t0 = time.time()
    segments, info = _run_whisper(audio, language, settings, logger)
    seg_list = [
        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
        for s in segments
    ]
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(" ".join(s["text"] for s in seg_list), encoding="utf-8")
    (TRANSCRIPTS_DIR / f"{video_id}.json").write_text(
        json.dumps({"language": info.language, "segments": seg_list},
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    audio.unlink(missing_ok=True)  # аудио удаляем — иначе диск кончится
    logger.info("%s: %d сегментов, %.1f мин обработки",
                video_id, len(seg_list), (time.time() - t0) / 60)
    return True


def transcribe_one(video_id: str, language, settings, logger) -> bool:
    """Один ролик целиком (используется в режиме --video)."""
    if (TRANSCRIPTS_DIR / f"{video_id}.txt").exists():
        return True
    audio = download_audio(video_id, logger)
    if not audio:
        logger.warning("не удалось скачать аудио %s", video_id)
        return False
    return transcribe_audio(video_id, audio, language, settings, logger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=None,
                        help="код языка (ru/kk/en), по умолчанию из settings.yaml")
    parser.add_argument("--video", default=None,
                        help="транскрибировать одно конкретное видео (тест казахского)")
    parser.add_argument("--limit", type=int, default=None,
                        help="максимум видео за запуск")
    args = parser.parse_args()

    logger = setup_logger("04_transcribe")
    require_ffmpeg()
    t0 = time.time()
    settings = load_settings()["transcribe"]
    language = args.language or settings["language"]
    if language in (None, "auto", ""):
        language = None  # Whisper определит язык каждого видео сам
    logger.info("Язык транскрибации: %s", language or "auto")

    if args.video:
        transcribe_one(args.video, language, settings, logger)
        return

    df = pd.read_csv(SCORED_CSV, dtype={"video_id": str})
    if "subs_missing" not in df.columns:
        # этап 03 не завершался/колонку стёрли — восстанавливаем по диску,
        # как это делает сам 03: субтитров нет, если нет data/subs/{id}.txt
        logger.warning("Колонки subs_missing нет — вычисляю по наличию data/subs/*.txt")
        subs_dir = DATA_DIR / "subs"
        df["subs_missing"] = ~df["video_id"].map(
            lambda v: (subs_dir / f"{v}.txt").exists())
    targets = df[df["is_hit"] | df["subs_missing"]]

    pending = [v for v in targets["video_id"]
               if not (TRANSCRIPTS_DIR / f"{v}.txt").exists()]
    if args.limit:
        # лимитируем именно НОВУЮ работу: --limit по targets при повторном
        # запуске брал бы те же первые N строк (уже готовые) и делал 0 работы
        pending = pending[:args.limit]
    logger.info("К транскрибации: %d видео (хиты + без субтитров), "
                "из них уже готово: %d", len(targets), len(targets) - len(pending))

    done = 0
    # скачиваем аудио наперёд в фоне, чтобы GPU не простаивал на сети
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}

        def ensure_prefetch(idx):
            for j in range(idx, min(idx + PREFETCH_AHEAD + 1, len(pending))):
                v = pending[j]
                if v not in futures:
                    futures[v] = pool.submit(download_audio, v, logger)

        for idx, vid in enumerate(pending):
            ensure_prefetch(idx)
            try:
                audio = futures[vid].result()
                if not audio:
                    logger.warning("не удалось скачать аудио %s", vid)
                    continue
                if transcribe_audio(vid, audio, language, settings, logger):
                    done += 1
            except Exception as e:  # одно упавшее видео не валит весь этап
                logger.error("ошибка на %s: %s", vid, e)
                if "out of memory" in str(e).lower():
                    logger.error("Похоже, не хватило VRAM: поставь в settings.yaml "
                                 "transcribe.model: medium и перезапусти этап")
    logger.info("Готово: %d/%d транскрибировано, %.1f мин",
                done, len(pending), (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
