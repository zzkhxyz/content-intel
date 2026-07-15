"""Этап 5: извлечение тем из транскриптов/субтитров + кластеризация.

Вход:  data/transcripts/*.txt (приоритет) и data/subs/*.txt
Выход: data/topics_per_video/{id}.json — тема каждого видео
       data/clusters.json               — кластеры тем
       data/topics.csv                  — кластер -> метрики
"""

import json
import time

import pandas as pd
from tqdm import tqdm

from utils import (DATA_DIR, ask_llm_json, load_settings, read_text,
                   setup_logger)

SCORED_CSV = DATA_DIR / "scored" / "videos.csv"
SUBS_DIR = DATA_DIR / "subs"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
TOPICS_DIR = DATA_DIR / "topics_per_video"
CLUSTERS_JSON = DATA_DIR / "clusters.json"
TOPICS_CSV = DATA_DIR / "topics.csv"

TOPIC_SYSTEM = (
    "Ты аналитик YouTube-контента. Отвечаешь СТРОГО валидным JSON без "
    "markdown-обёртки и без пояснений."
)

TOPIC_PROMPT = """Проанализируй транскрипт YouTube-видео и верни JSON ровно такой структуры:
{{
  "topic": "краткая формулировка главной темы видео (одно предложение)",
  "subtopics": ["подтема 1", "подтема 2"],
  "audience_questions": ["вопросы, на которые видео отвечает зрителю"],
  "key_terms": ["ключевые термины"],
  "hook": "чем ролик цепляет в первые 15 секунд",
  "vertical": "{vertical}"
}}

Название видео: {title}
Канал: {channel}

Транскрипт (может быть без пунктуации — это авто-субтитры):
---
{text}
---"""

CLUSTER_PROMPT = """Ниже нумерованный список тем YouTube-видео одной вертикали.
Схлопни похожие темы в {cmin}-{cmax} уникальных кластеров. Каждый кластер —
одна конкретная контентная тема (не «шины вообще», а «стоит ли брать китайскую
зимнюю резину»). Каждое видео отнеси ровно к одному кластеру.

Верни JSON (в "videos" — номера из списка):
{{"clusters": [{{"name": "название кластера", "videos": [1, 2]}}]}}

Список тем:
{topics_list}"""


def get_source_text(video_id: str, max_chars: int) -> str | None:
    """Текст видео: Whisper-транскрипт приоритетнее авто-субтитров."""
    for base in (TRANSCRIPTS_DIR, SUBS_DIR):
        p = base / f"{video_id}.txt"
        if p.exists():
            return read_text(p, limit=max_chars)
    return None


def main():
    logger = setup_logger("05_topics")
    t0 = time.time()
    settings = load_settings()["topics"]
    df = pd.read_csv(SCORED_CSV, dtype={"video_id": str})
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. тема каждого видео -------------------------------------------
    per_video = {}
    skipped = 0
    for _, row in tqdm(list(df.iterrows()), desc="темы"):
        vid = row["video_id"]
        out = TOPICS_DIR / f"{vid}.json"
        if out.exists():
            topic = json.loads(out.read_text(encoding="utf-8"))
            # старые кэши могли сохранить vertical, придуманную моделью
            # («шины», «home repair», …) — чиним и файл, и данные в памяти,
            # иначе кластеризация дробится по мусорным вертикалям
            if topic.get("vertical") != row["vertical"]:
                topic["vertical"] = row["vertical"]
                out.write_text(json.dumps(topic, ensure_ascii=False, indent=1),
                               encoding="utf-8")
            per_video[vid] = topic
            continue
        text = get_source_text(vid, settings["max_chars"])
        if not text:
            skipped += 1
            continue
        try:
            topic = ask_llm_json(
                TOPIC_PROMPT.format(vertical=row["vertical"], title=row["title"],
                                    channel=row["channel"], text=text),
                system=TOPIC_SYSTEM, logger=logger,
                model_key="llm_light",  # массовый дешёвый вызов
                # тема — маленький JSON; большой запас на выход не нужен
                # и не влезает в 6000 TPM лёгкой модели
                max_tokens=1500,
            )
            if not isinstance(topic, dict):
                # extract_json может вернуть и список — не роняем этап
                raise ValueError(f"тема не dict: {type(topic).__name__}")
        except Exception as e:
            logger.error("тема для %s не извлечена: %s", vid, e)
            continue
        # вертикаль берём из собранных данных, а не из ответа модели —
        # лёгкие модели любят переписать её отсебятиной
        topic["vertical"] = row["vertical"]
        out.write_text(json.dumps(topic, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        per_video[vid] = topic
    logger.info("Темы извлечены: %d видео (пропущено без текста: %d)",
                len(per_video), skipped)

    # --- 2. кластеризация -------------------------------------------------
    # По вертикалям отдельными вызовами: кластеры всё равно не смешивают
    # вертикали, а полный список тем не влезает в 8000 TPM бесплатного Groq.
    verticals = sorted({t.get("vertical", "?") for t in per_video.values()})
    n_vert = max(len(verticals), 1)
    cmin_v = max(3, settings["clusters_min"] // n_vert)
    cmax_v = max(cmin_v + 2, settings["clusters_max"] // n_vert)
    clusters = []
    for vertical in verticals:
        # В промпт и ответ идут порядковые номера, а не video_id: на ~140
        # видео это экономит тысячи токенов и удерживает запрос в минутном
        # лимите gpt-oss-120b (8000 токенов вход+выход на весь запрос).
        vids = [vid for vid, t in per_video.items()
                if t.get("vertical") == vertical]
        if not vids:
            continue
        topics_list = "\n".join(
            f"{i}: {str(per_video[vid].get('topic', '?'))[:90]}"
            for i, vid in enumerate(vids, start=1)
        )
        result = ask_llm_json(
            CLUSTER_PROMPT.format(cmin=cmin_v, cmax=cmax_v,
                                  topics_list=topics_list),
            system=TOPIC_SYSTEM, max_tokens=2000, logger=logger,
        )
        if isinstance(result, list):  # модель могла отдать голый список
            result = {"clusters": result}
        elif not isinstance(result, dict):
            result = {}
        part = [c for c in result.get("clusters", []) if isinstance(c, dict)]
        for c in part:
            c["vertical"] = vertical  # не доверяем этому полю от модели
            nums = []
            for v in c.get("videos", c.get("video_ids", [])):
                try:
                    nums.append(int(v))
                except (TypeError, ValueError):
                    continue
            c["video_ids"] = [vids[n - 1] for n in nums if 1 <= n <= len(vids)]
        clusters.extend(part)
        logger.info("[%s] кластеров: %d", vertical, len(part))
    for c in clusters:
        # dict.fromkeys: дубли от модели удвоили бы метрики кластера в 06/07
        c["video_ids"] = list(dict.fromkeys(c["video_ids"]))
    clusters = [c for c in clusters if c["video_ids"] and c.get("name")]
    if not clusters:
        raise SystemExit("Модель вернула 0 валидных кластеров — смотри logs/05_topics.log")

    # имена кластеров должны быть уникальны: по ним джойнятся 06 и 07
    seen_names = {}
    for c in clusters:
        base = c["name"]
        if base in seen_names:
            seen_names[base] += 1
            c["name"] = f"{base} ({seen_names[base]})"
        else:
            seen_names[base] = 1
    CLUSTERS_JSON.write_text(
        json.dumps({"clusters": clusters}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    logger.info("Кластеров: %d", len(clusters))

    # --- 3. метрики кластеров ----------------------------------------------
    metrics = df.set_index("video_id")
    rows = []
    for c in clusters:
        sub = metrics.loc[[v for v in c["video_ids"] if v in metrics.index]]
        rows.append({
            "topic_cluster": c["name"],
            "vertical": c["vertical"],
            "videos_count": len(sub),
            "total_views": int(sub["view_count"].sum()),
            "median_vpd": round(float(sub["vpd"].median()), 1),
            "max_outlier_score": round(float(sub["outlier_score"].max()), 2),
            "video_ids": ";".join(sub.index),
        })
    topics_df = pd.DataFrame(rows).sort_values("max_outlier_score", ascending=False)
    topics_df.to_csv(TOPICS_CSV, index=False, encoding="utf-8-sig")
    logger.info("Готово: %s (%.1f мин)", TOPICS_CSV, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
