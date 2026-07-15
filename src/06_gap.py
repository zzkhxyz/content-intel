"""Этап 6: GAP-АНАЛИЗ — ядро пилота.

Для каждого топ-кластера проверяем, что есть по теме в казахстанском поле:
поиск через yt-dlp (ytsearch), фильтрация KZ-контента (маркеры + LLM),
метрика дыры gap_score = ru_total_views / (kz_total_views + 1000).

Вход:  data/topics.csv, data/clusters.json
Выход: output/gap_table.csv, output/gap_analysis.png (график для слайда)
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (DATA_DIR, OUTPUT_DIR, ask_llm_json, load_settings,
                   setup_logger, ytdlp_json_lines)

TOPICS_CSV = DATA_DIR / "topics.csv"
SEARCH_CACHE_DIR = DATA_DIR / "gap_search"
GAP_TABLE = OUTPUT_DIR / "gap_table.csv"
GAP_PNG = OUTPUT_DIR / "gap_analysis.png"

JSON_SYSTEM = ("Ты аналитик рынка YouTube-контента. Отвечаешь СТРОГО валидным "
               "JSON без markdown и пояснений.")

QUERIES_PROMPT = """Для каждой темы ниже составь поисковые запросы YouTube, которыми
казахстанский зритель искал бы такой контент. По 2 запроса на русском и,
если тема осмысленно ищется на казахском, 1 запрос на казахском языке.
Запросы короткие, как реально ищут люди.

Темы (index: тема [вертикаль]):
{topics}

Верни JSON: {{"queries": [{{"index": 0, "ru": ["запрос 1", "запрос 2"], "kk": ["сұраныс"]}}]}}
Поле "kk" — пустой список, если казахский запрос не осмыслен."""

KZ_CLASSIFY_PROMPT = """Ниже результаты поиска YouTube по теме «{topic}».
Определи, какие видео сделаны казахстанскими авторами или явно ориентированы
на аудиторию Казахстана (казахский язык, города/бренды/цены Казахстана,
казахстанские каналы и СТО). Признаки-подсказки: {markers}.

Результаты (index: НАЗВАНИЕ | канал):
{results}

Верни JSON: {{"kz_indices": [3, 7]}} — индексы казахстанских видео.
Если таких нет — пустой список."""


def search_youtube(query: str, n: int, logger) -> list[dict]:
    """ytsearch с дисковым кэшем (повторный запуск не ходит в сеть)."""
    SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import hashlib
    key = hashlib.sha256(f"{n}:{query}".encode("utf-8")).hexdigest()[:24]
    cache = SEARCH_CACHE_DIR / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    entries = ytdlp_json_lines(
        ["--flat-playlist", "--dump-json", f"ytsearch{n}:{query}"], logger=logger
    )
    slim = [
        {
            "id": e.get("id"),
            "title": e.get("title") or "",
            "channel": e.get("channel") or e.get("uploader") or "",
            "view_count": e.get("view_count") or 0,
        }
        for e in entries if e.get("id")
    ]
    # пустой результат НЕ кэшируем: чаще всего это сбой сети/бан yt-dlp,
    # а закэшированная пустота навсегда завышает gap_score при перезапусках
    if slim:
        cache.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
    return slim


def classify_kz(topic: str, results: list[dict], markers: list[str], logger) -> set[int]:
    """Индексы казахстанских видео: маркеры + LLM-классификация."""
    kz = set()
    for i, r in enumerate(results):
        haystack = f"{r['title']} {r['channel']}".lower()
        if any(str(m).lower() in haystack for m in markers):
            kz.add(i)
    listing = "\n".join(f"{i}: {r['title']} | {r['channel']}"
                        for i, r in enumerate(results))
    try:
        resp = ask_llm_json(
            KZ_CLASSIFY_PROMPT.format(topic=topic, markers=", ".join(map(str, markers)),
                                      results=listing),
            system=JSON_SYSTEM, logger=logger,
            model_key="llm_light",  # массовый дешёвый вызов
            max_tokens=1000,  # ответ — короткий список индексов
        )
        kz |= {int(i) for i in resp.get("kz_indices", [])
               if isinstance(i, (int, str)) and str(i).isdigit()
               and 0 <= int(i) < len(results)}
    except Exception as e:
        logger.warning("KZ-LLM-классификация упала (%s), остаёмся на маркерах", e)
    return kz


def plot_gap(df: pd.DataFrame, top_n: int, logger):
    top = df.head(top_n).iloc[::-1]  # разворот, чтобы лидер был сверху
    y = np.arange(len(top))
    h = 0.38

    fig, ax = plt.subplots(figsize=(14, 0.6 * len(top) + 2.5), dpi=200)
    ax.barh(y + h / 2, top["ru_views"].clip(lower=1), height=h,
            color="#2f6db3", label="RU/US-контент, просмотры")
    ax.barh(y - h / 2, top["kz_views"].clip(lower=1), height=h,
            color="#e07b39", label="KZ-контент, просмотры")

    ax.set_xscale("log")
    ax.set_yticks(y)
    labels = [t if len(t) <= 60 else t[:57] + "..." for t in top["topic_cluster"]]
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Просмотры (логарифмическая шкала)", fontsize=11)
    ax.set_title("Темы, которые залетают в RU/US и отсутствуют в казнете",
                 fontsize=15, fontweight="bold", pad=15)

    for yi, (_, r) in zip(y, top.iterrows()):
        ax.text(max(r["kz_views"], 1) * 1.15, yi - h / 2,
                "0" if r["kz_views"] == 0 else f"{int(r['kz_views']):,}".replace(",", " "),
                va="center", fontsize=8, color="#7a4a20")
        ax.text(max(r["ru_views"], 1) * 1.15, yi + h / 2,
                f"{int(r['ru_views']):,}".replace(",", " "),
                va="center", fontsize=8, color="#1d4470")

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(GAP_PNG, bbox_inches="tight")
    plt.close(fig)
    logger.info("График сохранён: %s", GAP_PNG)


def main():
    logger = setup_logger("06_gap")
    t0 = time.time()
    settings = load_settings()["gap"]
    OUTPUT_DIR.mkdir(exist_ok=True)

    topics = pd.read_csv(TOPICS_CSV)
    topics = topics.sort_values("total_views", ascending=False).head(
        settings["top_clusters"]).reset_index(drop=True)
    logger.info("Gap-анализ по %d кластерам", len(topics))

    # --- поисковые запросы на все кластеры одним вызовом -------------------
    topics_listing = "\n".join(
        f"{i}: {r['topic_cluster']} [{r['vertical']}]" for i, r in topics.iterrows())
    qresp = ask_llm_json(
        QUERIES_PROMPT.format(topics=topics_listing),
        system=JSON_SYSTEM, max_tokens=3000, logger=logger,
    )
    if isinstance(qresp, list):  # модель могла отдать голый список
        qresp = {"queries": qresp}
    elif not isinstance(qresp, dict):
        qresp = {}
    queries_by_index = {int(q["index"]): q for q in qresp.get("queries", [])
                        if isinstance(q, dict)
                        and str(q.get("index", "")).lstrip("-").isdigit()}

    def as_query_list(value) -> list[str]:
        """модель может вернуть строку вместо списка — нормализуем."""
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [s for s in value if isinstance(s, str) and s.strip()]
        return []

    rows = []
    for i, r in tqdm(list(topics.iterrows()), desc="gap"):
        q = queries_by_index.get(i, {})
        query_list = as_query_list(q.get("ru")) + as_query_list(q.get("kk"))
        if not query_list:
            query_list = [r["topic_cluster"]]

        # собираем уникальные результаты по всем запросам темы
        seen, results = set(), []
        for query in query_list:
            for item in search_youtube(query, settings["search_results"], logger):
                if item["id"] not in seen:
                    seen.add(item["id"])
                    results.append(item)

        kz_idx = classify_kz(r["topic_cluster"], results,
                             settings["kz_markers"], logger) if results else set()
        kz_items = [results[j] for j in sorted(kz_idx)]
        kz_views = sum(it["view_count"] for it in kz_items)

        gap_score = r["total_views"] / (kz_views + 1000)
        rows.append({
            "topic_cluster": r["topic_cluster"],
            "vertical": r["vertical"],
            "ru_views": int(r["total_views"]),
            "ru_videos": int(r["videos_count"]),
            "kz_views": int(kz_views),
            "kz_videos": len(kz_items),
            "gap_score": round(gap_score, 1),
        })
        logger.info("[%s] RU %s против KZ %s -> gap %.1f",
                    r["topic_cluster"], rows[-1]["ru_views"], kz_views, gap_score)

    if not rows:
        raise SystemExit("Gap-анализ не дал ни одной строки — смотри logs/06_gap.log")
    gap_df = pd.DataFrame(rows).sort_values("gap_score", ascending=False)
    gap_df.to_csv(GAP_TABLE, index=False, encoding="utf-8-sig")
    logger.info("Таблица: %s", GAP_TABLE)

    plot_gap(gap_df, settings["chart_top"], logger)
    logger.info("Готово за %.1f мин", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
