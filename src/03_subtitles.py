"""Этап 3: дешёвый скрининг — авто-субтитры для ВСЕХ видео.

Вход:  data/scored/videos.csv
Выход: data/subs/{video_id}.txt (плоский текст),
       колонка subs_missing в data/scored/videos.csv

Качество авто-субтитров плохое (нет пунктуации, врут на терминах),
но для понимания "о чём ролик" достаточно. Whisper на всё не гоняем.
"""

import re
import time

import pandas as pd
from tqdm import tqdm

from utils import (DATA_DIR, load_settings, require_ffmpeg, setup_logger,
                   video_url, ytdlp_run)

SCORED_CSV = DATA_DIR / "scored" / "videos.csv"
SUBS_DIR = DATA_DIR / "subs"
SUBS_RAW_DIR = SUBS_DIR / "raw"

TAG_RE = re.compile(r"<[^>]+>")
TIMESTAMP_RE = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->")


def srt_to_text(srt_text: str) -> str:
    """SRT -> плоский текст: без номеров, таймкодов, тегов и повторов.

    Авто-субтитры YouTube «катятся» — одна и та же строка повторяется в
    соседних кью, поэтому дедупим последовательные повторы.
    """
    lines_out = []
    last = None
    for line in srt_text.splitlines():
        line = TAG_RE.sub("", line).strip()
        if not line or line.isdigit() or TIMESTAMP_RE.match(line):
            continue
        if line == last:
            continue
        lines_out.append(line)
        last = line
    return " ".join(lines_out)


def download_subs(video_id: str, languages: list[str], logger) -> bool:
    """Пытается скачать субтитры (ручные приоритетнее авто). True = есть .srt."""
    template = str(SUBS_RAW_DIR / f"{video_id}.%(ext)s")
    ytdlp_run(
        [
            "--skip-download",
            "--write-subs", "--write-auto-subs",
            "--sub-langs", ",".join(languages),
            "--convert-subs", "srt",
            # эндпоинт субтитров YouTube жёстко лимитирован: с паузой 4с
            # вчера словили 429 после ~сотни видео — держим 8с
            "--sleep-subtitles", "8",
            "-o", template,
            video_url(video_id),
        ],
        logger=logger,
    )
    return any(SUBS_RAW_DIR.glob(f"{video_id}.*.srt"))


def pick_srt(video_id: str, languages: list[str]):
    """Выбирает лучший .srt: приоритет по порядку языков в настройках."""
    files = list(SUBS_RAW_DIR.glob(f"{video_id}.*.srt"))
    if not files:
        return None
    for lang in languages:
        for f in files:
            if f".{lang}" in f.name:
                return f
    return files[0]


def main():
    logger = setup_logger("03_subtitles")
    require_ffmpeg()
    t0 = time.time()
    languages = load_settings()["subtitles"]["languages"]

    df = pd.read_csv(SCORED_CSV, dtype={"video_id": str})
    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    SUBS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    for vid in tqdm(df["video_id"], desc="субтитры"):
        txt_path = SUBS_DIR / f"{vid}.txt"
        if txt_path.exists():
            continue
        if not pick_srt(vid, languages):
            download_subs(vid, languages, logger)
        srt = pick_srt(vid, languages)
        if not srt:
            missing.append(vid)
            continue
        text = srt_to_text(srt.read_text(encoding="utf-8", errors="replace"))
        if len(text) < 100:  # мусор/пустышка — считаем что субтитров нет
            missing.append(vid)
            continue
        txt_path.write_text(text, encoding="utf-8")

    # subs_missing по фактическому наличию .txt (устойчиво к повторным запускам)
    df["subs_missing"] = ~df["video_id"].map(
        lambda v: (SUBS_DIR / f"{v}.txt").exists())
    df.to_csv(SCORED_CSV, index=False, encoding="utf-8-sig")

    n_missing = int(df["subs_missing"].sum())
    logger.info("Готово: субтитры есть у %d из %d, без субтитров %d "
                "(уйдут в Whisper). %.1f мин",
                len(df) - n_missing, len(df), n_missing, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
