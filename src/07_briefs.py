"""Этап 7: контент-брифы по топ-10 тем gap-анализа.

Вход:  output/gap_table.csv, data/clusters.json, транскрипты, темы
Выход: output/briefs/brief_01.md ... brief_10.md

Правовая рамка: бриф — оригинальный аналитический документ. Из транскриптов
извлекаем факты/темы/вопросы, НЕ формулировки. Прямые цитаты запрещены промптом.
"""

import json
import time

import pandas as pd

from utils import (DATA_DIR, OUTPUT_DIR, ask_llm, load_settings, read_text,
                   setup_logger)

GAP_TABLE = OUTPUT_DIR / "gap_table.csv"
CLUSTERS_JSON = DATA_DIR / "clusters.json"
SCORED_CSV = DATA_DIR / "scored" / "videos.csv"
TOPICS_DIR = DATA_DIR / "topics_per_video"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
SUBS_DIR = DATA_DIR / "subs"
BRIEFS_DIR = OUTPUT_DIR / "briefs"

BRIEF_SYSTEM = """Ты контент-стратег, который готовит брифы для казахстанского медиа
по темам «шины» и «ремонт». Ты пишешь ОРИГИНАЛЬНЫЙ аналитический бриф.

Жёсткие правила:
- НЕ пересказывай транскрипты близко к тексту.
- НЕ давай прямых цитат из видео.
- НЕ упоминай конкретные чужие видео: никаких ссылок, ID видео, названий
  роликов и имён каналов. Тезисы агрегируй по ВСЕЙ теме, а не по одному
  конкретному ролику.
- Из материалов бери только факты, темы, вопросы аудитории, структуру спроса
  и терминологию. Формулировки — свои.
- Столица Казахстана — Астана (не пиши «Нур-Султан»).
- НЕ выдумывай конкретные бренды, компании, СТО, цены и «факты о Казахстане»,
  которых нет в материалах. Где нужна локальная конкретика, которой нет, —
  пиши задание на проверку (например: «уточнить цены у местных поставщиков»).
- Пиши по-русски, конкретно, без воды."""

BRIEF_TEMPLATE = """Подготовь контент-бриф в точности по этому markdown-шаблону
(заполни каждый раздел содержательно):

# Бриф: {topic}

**Вертикаль:** {vertical_ru}
**Gap-score:** {gap_score}  |  RU: {ru_views_k}k просмотров / {ru_videos} видео  |  KZ: {kz_views_k}k / {kz_videos} видео

## Почему эта тема залетает
2-3 предложения: что именно цепляет аудиторию — по данным из материалов ниже.

## Что раскрывают конкуренты
5-7 тезисов, агрегированных по всей теме (НЕ цитаты, НЕ пересказ отдельных
роликов, без названий видео и каналов).

## Вопросы аудитории (из комментариев и содержания)
Маркированный список реальных вопросов.

## Чего конкуренты НЕ раскрывают
Пробелы = наше конкурентное преимущество.

## Локальная адаптация под Казахстан
Климат, наши дороги, цены в тенге, доступные у нас бренды, казахоязычная
аудитория, местные СТО/поставщики — что из этого релевантно теме и как обыграть.

## Предлагаемый формат
Статья / видео / серия. Заголовок-черновик. Ключевые слова.

=== МАТЕРИАЛЫ ДЛЯ АНАЛИЗА (сырьё, не для копирования) ===

Вопросы аудитории, собранные по видео кластера:
{questions}

Транскрипты видео кластера (обрезаны):
{transcripts}

Ответь ТОЛЬКО готовым markdown-брифом, без преамбулы."""

VERTICAL_RU = {"tires": "шины", "repair": "ремонт"}


def pick_top_topics(gap_df: pd.DataFrame, count: int, min_per_vertical: int) -> pd.DataFrame:
    """Топ по gap_score с гарантией минимума на каждую вертикаль.

    Гарантия работает, пока (число вертикалей * min_per_vertical) <= count —
    для пилота (2 * 3 <= 10) это всегда так.
    """
    if gap_df.empty:
        raise SystemExit("output/gap_table.csv пуст — сначала успешный 06_gap.py")
    picked = []
    for vertical in gap_df["vertical"].unique():
        sub = gap_df[gap_df["vertical"] == vertical].head(min_per_vertical)
        picked.append(sub)
    picked_df = pd.concat(picked).drop_duplicates("topic_cluster")
    rest = gap_df[~gap_df["topic_cluster"].isin(picked_df["topic_cluster"])]
    need = count - len(picked_df)
    result = pd.concat([picked_df, rest.head(max(need, 0))])
    return result.sort_values("gap_score", ascending=False).head(count)


def cluster_materials(video_ids: list[str], scored: pd.DataFrame,
                      settings) -> tuple[str, str]:
    """Собирает вопросы аудитории и транскрипты по кластеру.

    Ссылки/названия чужих видео в промпт НЕ идут: правовая рамка запрещает
    любые идентификаторы чужого контента в брифах, а модель охотно тащит
    в ответ всё, что видит в материалах. Транскрипты нумеруем обезличенно.
    """
    questions, transcripts = [], []
    total = 0
    for vid in video_ids:
        tpath = TOPICS_DIR / f"{vid}.json"
        if tpath.exists():
            t = json.loads(tpath.read_text(encoding="utf-8"))
            aq = t.get("audience_questions")
            if isinstance(aq, list):  # LLM-выход без схемы — валидируем сами
                questions.extend(q for q in aq if isinstance(q, str))
            elif isinstance(aq, str):
                questions.append(aq)
        for base in (TRANSCRIPTS_DIR, SUBS_DIR):
            p = base / f"{vid}.txt"
            if p.exists():
                chunk = read_text(p, limit=settings["max_chars_per_transcript"])
                if total + len(chunk) <= settings["max_total_chars"]:
                    transcripts.append(
                        f"--- материал {len(transcripts) + 1} ---\n{chunk}")
                    total += len(chunk)
                break
    # капы, чтобы запрос влезал в 8000 TPM бесплатного Groq
    uniq_q = list(dict.fromkeys(questions))[:15]
    q_text = "\n".join(f"- {q}" for q in uniq_q) or "- (не собраны)"
    return q_text, "\n\n".join(transcripts)


def main():
    logger = setup_logger("07_briefs")
    t0 = time.time()
    settings = load_settings()["briefs"]
    max_tokens = load_settings()["models"]["max_tokens_brief"]

    gap_df = pd.read_csv(GAP_TABLE)
    clusters = json.loads(CLUSTERS_JSON.read_text(encoding="utf-8"))["clusters"]
    cluster_by_name = {c["name"]: c for c in clusters}
    scored = pd.read_csv(SCORED_CSV, dtype={"video_id": str}).set_index("video_id")

    top = pick_top_topics(gap_df, settings["count"], settings["min_per_vertical"])
    logger.info("Брифы по %d темам: %s", len(top),
                "; ".join(top["topic_cluster"].head(10)))

    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    # brief_NN привязан к ПОЗИЦИИ в текущем gap_table.csv, поэтому старые
    # файлы сносим целиком: если ранжирование изменилось после перезапуска
    # 06_gap (или тем стало меньше), старый brief_NN не должен «прилипнуть»
    # к чужой теме. Повторная генерация с теми же входами бесплатна —
    # вызовы LLM кэшируются.
    for old in BRIEFS_DIR.glob("brief_*.md"):
        old.unlink()
    for n, (_, r) in enumerate(top.iterrows(), start=1):
        out = BRIEFS_DIR / f"brief_{n:02d}.md"
        cluster = cluster_by_name.get(r["topic_cluster"])
        if not cluster:
            logger.warning("кластер «%s» не найден в clusters.json", r["topic_cluster"])
            continue
        questions, transcripts = cluster_materials(
            cluster["video_ids"], scored, settings)

        prompt = BRIEF_TEMPLATE.format(
            topic=r["topic_cluster"],
            vertical_ru=VERTICAL_RU.get(r["vertical"], r["vertical"]),
            gap_score=r["gap_score"],
            ru_views_k=round(r["ru_views"] / 1000),
            ru_videos=r["ru_videos"],
            kz_views_k=round(r["kz_views"] / 1000),
            kz_videos=r["kz_videos"],
            questions=questions,
            transcripts=transcripts or "(транскрипты недоступны)",
        )
        brief = ask_llm(prompt, system=BRIEF_SYSTEM,
                           max_tokens=max_tokens, logger=logger)
        if not brief.strip():
            # gpt-oss — reasoning-модель: она может потратить весь max_tokens
            # на размышления и вернуть пустой content. Повтор без кэша,
            # с запасом побольше (но в пределах 8000 TPM бесплатного Groq).
            logger.warning("пустой ответ модели, повтор с max_tokens=3500")
            brief = ask_llm(prompt, system=BRIEF_SYSTEM, max_tokens=3500,
                            use_cache=False, logger=logger)
        if not brief.strip():
            raise SystemExit(f"Бриф «{r['topic_cluster']}» пуст после повтора — "
                             "смотри logs/07_briefs.log")
        out.write_text(brief.strip() + "\n", encoding="utf-8")
        logger.info("Готов %s: %s", out.name, r["topic_cluster"])

    logger.info("Готово за %.1f мин -> %s", (time.time() - t0) / 60, BRIEFS_DIR)


if __name__ == "__main__":
    main()
