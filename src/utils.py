"""Общие утилиты пайплайна: конфиги, логирование, yt-dlp, LLM (Groq) с кэшем."""

import hashlib
import json
import logging
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
LOGS_DIR = ROOT / "logs"
LLM_CACHE_DIR = DATA_DIR / ".cache_llm"

# .env ищем и в content-intel/, и на уровень выше (load_dotenv не
# перетирает уже загруженные значения, порядок задаёт приоритет)
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")


# ---------------------------------------------------------------- конфиги

def load_settings() -> dict:
    with open(CONFIG_DIR / "settings.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_channels() -> dict:
    with open(CONFIG_DIR / "channels.yaml", encoding="utf-8") as f:
        channels = yaml.safe_load(f) or {}
    channels = {v: urls for v, urls in channels.items() if urls}
    if not channels:
        sys.exit(
            "config/channels.yaml пуст. Заполни каналы-доноры перед запуском "
            "(см. комментарии в самом файле)."
        )
    return channels


# ------------------------------------------------------------ логирование

def setup_logger(stage: str) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    # Windows-консоль по умолчанию cp1251 — переключаем на UTF-8,
    # иначе русские заголовки видео роняют print/logging.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logger = logging.getLogger(stage)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(LOGS_DIR / f"{stage}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ----------------------------------------------------------------- yt-dlp

def ytdlp_json_lines(args: list[str], logger=None, timeout: int = 1800) -> list[dict]:
    """Запускает yt-dlp с --dump-json-подобным выводом, возвращает список dict.

    yt-dlp может частично упасть (недоступные видео) и вернуть ненулевой код,
    но валидные строки JSON всё равно печатает — их и забираем.
    """
    settings = load_settings()
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--sleep-requests", str(settings["harvest"]["sleep_requests"]),
        "--no-warnings",
        "--ignore-errors",
        # без этого YouTube отдаёт названия/описания в английском автопереводе
        "--extractor-args", "youtube:lang=ru",
    ] + args
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        if logger:
            logger.warning("yt-dlp завис (>%dс) на: %s", timeout, " ".join(args[-1:]))
        return []
    if proc.returncode != 0 and logger:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        logger.warning("yt-dlp код %s: %s", proc.returncode, " | ".join(tail))
    items = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def ytdlp_run(args: list[str], logger=None, timeout: int = 3600) -> bool:
    """Запускает yt-dlp без разбора вывода (субтитры, аудио). True = успех."""
    settings = load_settings()
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--sleep-requests", str(settings["harvest"]["sleep_requests"]),
        "--no-warnings",
        "--extractor-args", "youtube:lang=ru",
    ] + args
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        if logger:
            logger.warning("yt-dlp завис (>%dс) на: %s", timeout, " ".join(args[-1:]))
        return False
    if proc.returncode != 0 and logger:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        logger.warning("yt-dlp код %s: %s", proc.returncode, " | ".join(tail))
    return proc.returncode == 0


def require_ffmpeg() -> None:
    """ffmpeg нужен yt-dlp для --convert-subs и -x. Без него этапы 03/04
    "тихо" деградируют (остаются .vtt/.webm), поэтому падаем сразу и громко."""
    if not shutil.which("ffmpeg"):
        sys.exit(
            "ffmpeg не найден в PATH. Установи его перед запуском:\n"
            "  winget install ffmpeg   (или скачай с ffmpeg.org и добавь в PATH)\n"
            "После установки перезапусти терминал."
        )


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# --------------------------------------------------------------- LLM (Groq)

def ask_llm(prompt: str, system: str | None = None,
            max_tokens: int | None = None, use_cache: bool = True,
            logger=None, model_key: str = "llm") -> str:
    """Вызов LLM через Groq API с дисковым кэшем и retry.

    Кэш: повторный запуск этапа не тратит лимиты на те же вызовы (ТЗ, п.6).
    SDK сам ретраит 429/5xx (с учётом retry-after); сверху свой цикл на
    длинные недоступности. На бесплатном тарифе Groq есть лимиты в минуту
    И В ДЕНЬ: если дневной лимит исчерпан — этап падает на очередном видео,
    прогресс в кэше, перезапуск на следующий день продолжает с места падения.
    """
    import groq

    settings = load_settings()
    model = settings["models"].get(model_key) or settings["models"]["llm"]
    max_tokens = max_tokens or settings["models"]["max_tokens_default"]

    key = hashlib.sha256(
        json.dumps([model, system, prompt, max_tokens],
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    cache_file = LLM_CACHE_DIR / f"{key}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))["text"]

    client = groq.Groq(max_retries=5)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    extra = {}
    if "gpt-oss" in model:
        # reasoning-модель: на низком effort рассуждения не съедают
        # max_tokens (иначе на больших задачах финальный ответ приходит
        # пустым — весь бюджет уходит в reasoning до обрезки по length)
        extra["reasoning_effort"] = settings["models"].get(
            "reasoning_effort", "low")

    last_err = None
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                **extra,
            )
            choice = resp.choices[0]
            text = choice.message.content or ""
            _log_usage(model, resp.usage)
            time.sleep(settings["models"].get("sleep_between_calls", 0))
            if choice.finish_reason == "length":
                # обрезанный ответ НЕ кэшируем — иначе после повышения
                # max_tokens кэш продолжит отдавать обрубок
                if logger:
                    logger.warning("ответ обрезан по max_tokens=%s (не кэширую)",
                                   max_tokens)
                return text
            LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8"
            )
            return text
        except groq.APIStatusError as e:
            if e.status_code < 500 and e.status_code != 429:
                raise
            last_err = e
        except groq.APIConnectionError as e:
            last_err = e
        delay = min(10 * 2 ** attempt + random.uniform(0, 3), 120)
        if logger:
            logger.warning("Groq недоступен (%s), повтор через %.0fс", last_err, delay)
        time.sleep(delay)
    raise last_err


def _log_usage(model: str, usage) -> None:
    """Пишет расход токенов в logs/llm_usage.jsonl.

    Groq бесплатный, но токены считаем всё равно — по ним экстраполируется
    стоимость «точки ноль» на платном тарифе/другом провайдере.
    """
    LOGS_DIR.mkdir(exist_ok=True)
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }
    with open(LOGS_DIR / "llm_usage.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_json(text: str):
    """Достаёт JSON из ответа модели, даже если он обёрнут в markdown/прозу."""
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Не удалось разобрать JSON из ответа: {text[:300]}...")


def ask_llm_json(prompt: str, system: str | None = None,
                 max_tokens: int | None = None, logger=None,
                 model_key: str = "llm"):
    """ask_llm + разбор JSON; повтор при невалидном JSON.

    Повтор идёт с удвоенным max_tokens: частая причина битого JSON —
    обрезка ответа по лимиту токенов.
    """
    text = ask_llm(prompt, system=system, max_tokens=max_tokens,
                   logger=logger, model_key=model_key)
    try:
        return extract_json(text)
    except ValueError:
        if logger:
            logger.warning("Невалидный JSON от модели, повторяю запрос")
        base = max_tokens or load_settings()["models"]["max_tokens_default"]
        # промпт с суффиксом даёт другой кэш-ключ, коллизии с первым вызовом
        # нет — а кэш удачного ретрая делает результат стабильным между
        # перезапусками (иначе кластеры «плавали» бы при каждом прогоне)
        text = ask_llm(
            prompt + "\n\nВАЖНО: ответь СТРОГО валидным JSON, без пояснений "
                     "и без markdown-обёртки.",
            system=system, max_tokens=min(base * 2, 16000),
            logger=logger, model_key=model_key,
        )
        return extract_json(text)


# ------------------------------------------------------------------ разное

def read_text(path: Path, limit: int | None = None) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:limit] if limit else text
