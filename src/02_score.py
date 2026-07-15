"""Этап 2: расчёт метрик "что залетело".

Вход:  data/raw/videos.csv
Выход: data/scored/videos.csv (отсортирован по outlier_score)

Главная метрика — outlier_score = vpd / медианный vpd канала:
во сколько раз ролик обогнал обычный ролик ЭТОГО канала.
Она отвечает на вопрос "тема выстрелила", а не "канал большой".
"""

from datetime import datetime

import pandas as pd

from utils import DATA_DIR, load_settings, setup_logger

RAW_CSV = DATA_DIR / "raw" / "videos.csv"
SCORED_DIR = DATA_DIR / "scored"


def main():
    logger = setup_logger("02_score")
    settings = load_settings()["scoring"]

    df = pd.read_csv(RAW_CSV, dtype={"video_id": str, "upload_date": str})
    logger.info("Загружено %d видео из %s", len(df), RAW_CSV)

    today = datetime.now()
    upload = pd.to_datetime(df["upload_date"], format="%Y%m%d", errors="coerce")
    df["days_live"] = (today - upload).dt.days.clip(lower=1)
    df["vpd"] = df["view_count"] / df["days_live"]

    channel_median = df.groupby("channel")["vpd"].transform("median")
    df["channel_median_vpd"] = channel_median
    df["outlier_score"] = (df["vpd"] / channel_median.clip(lower=0.01)).round(2)

    df["engagement"] = (
        (df["like_count"].fillna(0) + df["comment_count"].fillna(0))
        / df["view_count"].clip(lower=1)
    ).round(4)

    df["vpd"] = df["vpd"].round(1)
    df["channel_median_vpd"] = df["channel_median_vpd"].round(1)
    df["is_hit"] = df["outlier_score"] >= settings["hit_outlier_score"]

    df = df.sort_values("outlier_score", ascending=False)

    # subs_missing проставляет этап 03, но 02 может перезапускаться позже и
    # перезаписать CSV — восстанавливаем колонку по фактическому наличию
    # .txt на диске, иначе 04 потеряет список «без субтитров»
    subs_dir = DATA_DIR / "subs"
    df["subs_missing"] = ~df["video_id"].map(
        lambda v: (subs_dir / f"{v}.txt").exists())

    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    out = SCORED_DIR / "videos.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    hits = int(df["is_hit"].sum())
    logger.info("Готово: %s | хитов (outlier_score >= %s): %d из %d",
                out, settings["hit_outlier_score"], hits, len(df))
    for vertical, g in df.groupby("vertical"):
        logger.info("  [%s] топ-3 хита:", vertical)
        for _, r in g.head(3).iterrows():
            logger.info("    x%.1f  %s  (%s)", r["outlier_score"], r["title"], r["channel"])


if __name__ == "__main__":
    main()
