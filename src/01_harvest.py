"""Этап 1: сбор метаданных видео с каналов-доноров.

Вход:  config/channels.yaml
Выход: data/raw/videos.csv

Видео НЕ скачиваем — только метаданные. Полные метаданные каждого видео
кэшируются в data/raw/meta/{id}.json, повторный запуск их не перекачивает.
"""

import json
import time
from datetime import datetime, timedelta

import pandas as pd

from utils import (DATA_DIR, load_channels, load_settings, setup_logger,
                   video_url, ytdlp_json_lines)

RAW_DIR = DATA_DIR / "raw"
META_DIR = RAW_DIR / "meta"

CSV_COLUMNS = [
    "video_id", "channel", "channel_subs", "vertical", "title", "description",
    "upload_date", "duration_sec", "view_count", "like_count", "comment_count",
    "url",
]


def list_channel_videos(channel_url: str, limit: int, logger) -> list[str]:
    """Плоский список последних video_id канала (без метаданных)."""
    entries = ytdlp_json_lines(
        ["--flat-playlist", "--dump-json", "--playlist-end", str(limit), channel_url],
        logger=logger,
    )
    return [e["id"] for e in entries if e.get("id")]


def fetch_video_meta(video_id: str, logger) -> dict | None:
    """Полные метаданные одного видео (с дисковым кэшем)."""
    cache = META_DIR / f"{video_id}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    items = ytdlp_json_lines(
        ["--dump-json", "--skip-download", video_url(video_id)], logger=logger
    )
    if not items:
        return None
    meta = items[0]
    slim = {
        "id": meta.get("id"),
        "title": meta.get("title"),
        "description": (meta.get("description") or "")[:5000],
        "channel": meta.get("channel") or meta.get("uploader"),
        "channel_follower_count": meta.get("channel_follower_count"),
        "upload_date": meta.get("upload_date"),
        "duration": meta.get("duration"),
        "view_count": meta.get("view_count"),
        "like_count": meta.get("like_count"),
        "comment_count": meta.get("comment_count"),
        "chapters": meta.get("chapters"),
    }
    META_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
    return slim


def main():
    logger = setup_logger("01_harvest")
    t0 = time.time()
    settings = load_settings()["harvest"]
    channels = load_channels()
    cutoff = (datetime.now() - timedelta(days=30 * settings["months_back"])).strftime("%Y%m%d")

    rows = []
    seen_ids = set()  # видео не должно попасть в две вертикали и не должно съедать лимит дважды
    for vertical, urls in channels.items():
        vertical_count = 0
        for channel_url in urls:
            if vertical_count >= settings["max_videos_per_vertical"]:
                logger.info("[%s] лимит вертикали достигнут, канал %s пропущен",
                            vertical, channel_url)
                break
            logger.info("[%s] сканирую %s", vertical, channel_url)
            video_ids = list_channel_videos(
                channel_url, settings["playlist_scan_limit"], logger)
            logger.info("[%s] найдено %d видео в плейлисте", vertical, len(video_ids))

            consecutive_old = 0
            channel_count = 0
            for vid in video_ids:
                if vertical_count >= settings["max_videos_per_vertical"]:
                    break
                if channel_count >= settings["max_videos_per_channel"]:
                    logger.info("канал: потолок %d видео достигнут",
                                settings["max_videos_per_channel"])
                    break
                if vid in seen_ids:
                    continue
                try:
                    meta = fetch_video_meta(vid, logger)
                except Exception as e:  # одно битое видео не валит весь сбор
                    logger.error("сбой метаданных %s: %s", vid, e)
                    continue
                if not meta or not meta.get("upload_date"):
                    logger.warning("нет метаданных для %s, пропуск", vid)
                    continue
                if meta["upload_date"] < cutoff:
                    consecutive_old += 1
                    # /videos отсортирован от новых к старым: после нескольких
                    # старых подряд дальше сканить канал нет смысла
                    if consecutive_old >= settings["stop_after_old"]:
                        logger.info("канал: %d старых видео подряд, дальше не сканирую",
                                    consecutive_old)
                        break
                    continue
                consecutive_old = 0
                seen_ids.add(vid)
                channel_count += 1
                rows.append({
                    "video_id": meta["id"],
                    "channel": meta["channel"],
                    "channel_subs": meta.get("channel_follower_count") or 0,
                    "vertical": vertical,
                    "title": meta["title"],
                    "description": meta["description"],
                    "upload_date": meta["upload_date"],
                    "duration_sec": meta.get("duration") or 0,
                    "view_count": meta.get("view_count") or 0,
                    "like_count": meta.get("like_count") or 0,
                    "comment_count": meta.get("comment_count") or 0,
                    "url": video_url(meta["id"]),
                })
                vertical_count += 1
        logger.info("[%s] собрано %d видео", vertical, vertical_count)

    if not rows:
        # именно sys.exit(1): иначе оркестратор посчитает этап успешным и
        # погонит пайплайн дальше на пустых/устаревших данных
        logger.error("Не собрано ни одного видео. Проверь channels.yaml и сеть.")
        raise SystemExit(1)

    df = pd.DataFrame(rows, columns=CSV_COLUMNS).drop_duplicates("video_id")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "videos.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("Готово: %d видео -> %s (%.1f мин)",
                len(df), out, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
