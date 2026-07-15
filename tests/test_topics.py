# -*- coding: utf-8 -*-
"""Тесты src/05_topics.py: выбор источника текста, кластеризация по
порядковым НОМЕРАМ (1-based) вместо video_id (экономия TPM Groq),
чистка кластеров (валидация номеров, дедуп, уникализация имён), метрики.

LLM полностью замокан, все пути уведены в tmp_path.
"""

import json
import re

import pandas as pd
import pytest

SETTINGS = {"topics": {"max_chars": 6000, "clusters_min": 15, "clusters_max": 25}}


# ------------------------------------------------------------ get_source_text

def test_get_source_text_prefers_transcript_over_subs(topics_mod, tmp_path,
                                                      monkeypatch):
    tr = tmp_path / "transcripts"
    sb = tmp_path / "subs"
    tr.mkdir(), sb.mkdir()
    (tr / "v1.txt").write_text("whisper-текст", encoding="utf-8")
    (sb / "v1.txt").write_text("авто-субтитры", encoding="utf-8")
    (sb / "v2.txt").write_text("только субтитры", encoding="utf-8")
    monkeypatch.setattr(topics_mod, "TRANSCRIPTS_DIR", tr)
    monkeypatch.setattr(topics_mod, "SUBS_DIR", sb)

    assert topics_mod.get_source_text("v1", 100) == "whisper-текст"
    assert topics_mod.get_source_text("v2", 100) == "только субтитры"
    assert topics_mod.get_source_text("нет_такого", 100) is None


def test_get_source_text_respects_max_chars(topics_mod, tmp_path, monkeypatch):
    tr = tmp_path / "transcripts"
    tr.mkdir()
    (tr / "v1.txt").write_text("a" * 500, encoding="utf-8")
    monkeypatch.setattr(topics_mod, "TRANSCRIPTS_DIR", tr)
    monkeypatch.setattr(topics_mod, "SUBS_DIR", tmp_path / "subs")
    assert len(topics_mod.get_source_text("v1", 100)) == 100


# ------------------------------------------------------- main() с моками LLM

@pytest.fixture()
def topics_env(topics_mod, tmp_path, monkeypatch, no_real_logs):
    scored_dir = tmp_path / "scored"
    scored_dir.mkdir()
    for name in ("subs", "transcripts", "topics_per_video"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(topics_mod, "SCORED_CSV", scored_dir / "videos.csv")
    monkeypatch.setattr(topics_mod, "SUBS_DIR", tmp_path / "subs")
    monkeypatch.setattr(topics_mod, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(topics_mod, "TOPICS_DIR", tmp_path / "topics_per_video")
    monkeypatch.setattr(topics_mod, "CLUSTERS_JSON", tmp_path / "clusters.json")
    monkeypatch.setattr(topics_mod, "TOPICS_CSV", tmp_path / "topics.csv")
    monkeypatch.setattr(topics_mod, "load_settings", lambda: SETTINGS)
    monkeypatch.setattr(topics_mod, "setup_logger", no_real_logs)
    return tmp_path


def _scored_df():
    return pd.DataFrame([
        {"video_id": "t1", "title": "Шины 1", "channel": "c1",
         "vertical": "tires", "view_count": 1000, "vpd": 10.0,
         "outlier_score": 1.5},
        {"video_id": "t2", "title": "Шины 2", "channel": "c1",
         "vertical": "tires", "view_count": 3000, "vpd": 30.0,
         "outlier_score": 4.5},
        {"video_id": "r1", "title": "Ремонт 1", "channel": "c2",
         "vertical": "repair", "view_count": 500, "vpd": 5.0,
         "outlier_score": 2.2},
    ])


def _seed(env, df):
    """scored/videos.csv + кэш темы для каждого видео (LLM тем не нужен)."""
    df.to_csv(env / "scored" / "videos.csv", index=False, encoding="utf-8")
    for _, row in df.iterrows():
        (env / "topics_per_video" / f"{row['video_id']}.json").write_text(
            json.dumps({"topic": f"тема про {row['title']}",
                        "vertical": row["vertical"]}, ensure_ascii=False),
            encoding="utf-8")


def _prompt_numbers(prompt):
    """Номера строк '1: тема' из списка тем в промпте кластеризации."""
    listing = prompt.split("Список тем:")[1]
    return [int(n) for n in re.findall(r"^(\d+): ", listing, re.M)]


def _fake_cluster_llm(extra_videos=(), dup_first=True):
    """Мок ask_llm_json: читает НОМЕРА тем (1-based) из промпта и возвращает
    кластеры с этими номерами (+ мусор для проверки чистки)."""
    calls = []

    def fake(prompt, system=None, max_tokens=None, logger=None, model_key="llm"):
        calls.append({"prompt": prompt, "max_tokens": max_tokens})
        nums = _prompt_numbers(prompt)
        videos = list(nums) + list(extra_videos)
        if dup_first and nums:
            videos.append(nums[0])  # дубль от модели
        return {"clusters": [
            {"name": "Общий кластер", "vertical": "мусор от модели",
             "videos": videos},
            {"name": "Пустой", "videos": []},         # должен отфильтроваться
            {"name": "", "videos": list(nums)},        # без имени — отфильтр.
        ]}

    return fake, calls


def test_main_prompt_uses_numbers_not_ids(topics_mod, topics_env, monkeypatch):
    """Новый контракт: нумерованный список тем без video_id и без [vertical]."""
    _seed(topics_env, _scored_df())
    fake, calls = _fake_cluster_llm()
    monkeypatch.setattr(topics_mod, "ask_llm_json", fake)

    topics_mod.main()

    assert len(calls) == 2  # по вызову на вертикаль (repair, tires)
    for call in calls:
        prompt = call["prompt"]
        # нумерация с единицы, подряд
        nums = _prompt_numbers(prompt)
        assert nums == list(range(1, len(nums) + 1))
        # video_id и [vertical] в списке тем больше не передаются
        listing = prompt.split("Список тем:")[1]
        assert "t1:" not in listing and "t2:" not in listing
        assert "r1:" not in listing
        assert "[tires]" not in listing and "[repair]" not in listing
        # max_tokens кластеризации снижен до 2000 (TPM-лимит Groq)
        assert call["max_tokens"] == 2000
    # repair — одна тема в списке, tires — две
    assert sorted(len(_prompt_numbers(c["prompt"])) for c in calls) == [1, 2]


def test_main_cleans_clusters_and_uniquifies_names(topics_mod, topics_env,
                                                   monkeypatch):
    _seed(topics_env, _scored_df())
    # мусор от модели: номер вне диапазона + нечисловые значения
    fake, calls = _fake_cluster_llm(extra_videos=[99, "abc", None])
    monkeypatch.setattr(topics_mod, "ask_llm_json", fake)

    topics_mod.main()

    data = json.loads((topics_env / "clusters.json").read_text(encoding="utf-8"))
    clusters = data["clusters"]
    # пустые и безымянные кластеры выброшены
    assert len(clusters) == 2
    # одинаковые имена от модели уникализированы
    assert [c["name"] for c in clusters] == ["Общий кластер",
                                             "Общий кластер (2)"]
    for c in clusters:
        # vertical модели перетёрт реальным
        assert c["vertical"] in ("tires", "repair")
        # дубли схлопнуты
        assert len(c["video_ids"]) == len(set(c["video_ids"]))
    by_vert = {c["vertical"]: c for c in clusters}
    # номера смаплены обратно в реальные video_id, мусор выброшен
    assert by_vert["tires"]["video_ids"] == ["t1", "t2"]
    assert by_vert["repair"]["video_ids"] == ["r1"]


def test_main_drops_out_of_range_and_non_numeric(topics_mod, topics_env,
                                                 monkeypatch):
    """0 и len+1 — вне диапазона 1..len; нечисловые молча отбрасываются;
    числовые строки от модели принимаются."""
    df = _scored_df()[_scored_df()["vertical"] == "tires"]  # только t1, t2
    _seed(topics_env, df)
    monkeypatch.setattr(
        topics_mod, "ask_llm_json",
        lambda *a, **kw: {"clusters": [
            {"name": "К", "videos": [0, 3, "abc", None, "2", 1]},
        ]})

    topics_mod.main()

    data = json.loads((topics_env / "clusters.json").read_text(encoding="utf-8"))
    assert len(data["clusters"]) == 1
    # 0 и 3 вне диапазона, "abc"/None — мусор; "2" -> t2, 1 -> t1
    assert data["clusters"][0]["video_ids"] == ["t2", "t1"]


def test_main_accepts_legacy_video_ids_key_with_numbers(topics_mod, topics_env,
                                                        monkeypatch):
    df = _scored_df()[_scored_df()["vertical"] == "tires"]
    _seed(topics_env, df)
    monkeypatch.setattr(
        topics_mod, "ask_llm_json",
        lambda *a, **kw: {"clusters": [
            {"name": "Легаси", "video_ids": [2, 1]},
        ]})

    topics_mod.main()

    data = json.loads((topics_env / "clusters.json").read_text(encoding="utf-8"))
    assert data["clusters"][0]["video_ids"] == ["t2", "t1"]


def test_main_videos_key_wins_over_legacy(topics_mod, topics_env, monkeypatch):
    df = _scored_df()[_scored_df()["vertical"] == "tires"]
    _seed(topics_env, df)
    monkeypatch.setattr(
        topics_mod, "ask_llm_json",
        lambda *a, **kw: {"clusters": [
            {"name": "Оба ключа", "videos": [1], "video_ids": [2]},
        ]})

    topics_mod.main()

    data = json.loads((topics_env / "clusters.json").read_text(encoding="utf-8"))
    assert data["clusters"][0]["video_ids"] == ["t1"]


def test_main_metrics_per_cluster(topics_mod, topics_env, monkeypatch):
    _seed(topics_env, _scored_df())
    fake, _ = _fake_cluster_llm()
    monkeypatch.setattr(topics_mod, "ask_llm_json", fake)

    topics_mod.main()

    df = pd.read_csv(topics_env / "topics.csv").set_index("vertical")
    tires = df.loc["tires"]
    assert tires["videos_count"] == 2
    assert tires["total_views"] == 4000            # 1000 + 3000
    assert tires["median_vpd"] == pytest.approx(20.0)   # медиана 10 и 30
    assert tires["max_outlier_score"] == pytest.approx(4.5)
    assert set(tires["video_ids"].split(";")) == {"t1", "t2"}
    # сортировка по max_outlier_score убыв.: tires (4.5) раньше repair (2.2)
    df2 = pd.read_csv(topics_env / "topics.csv")
    assert list(df2["vertical"]) == ["tires", "repair"]


def test_main_fixes_stale_vertical_in_cached_topic(topics_mod, topics_env,
                                                   monkeypatch):
    """Старый кэш темы с выдуманной вертикалью чинится по данным CSV."""
    _seed(topics_env, _scored_df())
    (topics_env / "topics_per_video" / "t1.json").write_text(
        json.dumps({"topic": "тема t1", "vertical": "шины (выдумка модели)"},
                   ensure_ascii=False), encoding="utf-8")
    fake, _ = _fake_cluster_llm()
    monkeypatch.setattr(topics_mod, "ask_llm_json", fake)

    topics_mod.main()

    fixed = json.loads(
        (topics_env / "topics_per_video" / "t1.json").read_text(encoding="utf-8"))
    assert fixed["vertical"] == "tires"


def test_main_raises_when_no_valid_clusters(topics_mod, topics_env, monkeypatch):
    _seed(topics_env, _scored_df())
    monkeypatch.setattr(topics_mod, "ask_llm_json",
                        lambda *a, **kw: {"clusters": []})
    with pytest.raises(SystemExit):
        topics_mod.main()


def test_main_skips_videos_without_text(topics_mod, topics_env, monkeypatch):
    """Видео без транскрипта/субтитров и без кэша темы просто пропускается."""
    df = _scored_df()
    df.to_csv(topics_env / "scored" / "videos.csv", index=False,
              encoding="utf-8")
    # темы есть только у t1 и t2; у r1 нет ни темы, ни текста
    for vid, title in (("t1", "Шины 1"), ("t2", "Шины 2")):
        (topics_env / "topics_per_video" / f"{vid}.json").write_text(
            json.dumps({"topic": f"тема про {title}", "vertical": "tires"},
                       ensure_ascii=False), encoding="utf-8")
    fake, calls = _fake_cluster_llm()
    monkeypatch.setattr(topics_mod, "ask_llm_json", fake)

    topics_mod.main()

    data = json.loads((topics_env / "clusters.json").read_text(encoding="utf-8"))
    all_ids = {v for c in data["clusters"] for v in c["video_ids"]}
    assert all_ids == {"t1", "t2"}
    assert len(calls) == 1  # одна вертикаль — один вызов кластеризации
