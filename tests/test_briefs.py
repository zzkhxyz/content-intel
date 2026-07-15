# -*- coding: utf-8 -*-
"""Тесты src/07_briefs.py: pick_top_topics (топ по gap_score с гарантией
минимума на вертикаль) и cluster_materials (сбор вопросов/транскриптов
с капами). Без LLM.

Правовая рамка: cluster_materials НЕ собирает ссылки/названия чужих видео,
а транскрипты в промпте нумеруются обезличенно (без video_id) — это
контракт, закреплённый тестами ниже."""

import json

import pandas as pd
import pytest


def _gap_df(rows):
    return pd.DataFrame(rows, columns=["topic_cluster", "vertical", "gap_score"])


# -------------------------------------------------------------- pick_top_topics

def test_pick_top_topics_guarantees_min_per_vertical(briefs_mod):
    df = _gap_df([
        ("t1", "tires", 100.0),
        ("t2", "tires", 90.0),
        ("t3", "tires", 80.0),
        ("t4", "tires", 70.0),
        ("r1", "repair", 5.0),   # без гарантии repair бы не попал
        ("r2", "repair", 4.0),
    ])
    top = briefs_mod.pick_top_topics(df, count=4, min_per_vertical=2)
    assert len(top) == 4
    assert set(top[top["vertical"] == "repair"]["topic_cluster"]) == {"r1", "r2"}
    assert set(top[top["vertical"] == "tires"]["topic_cluster"]) == {"t1", "t2"}
    # результат отсортирован по gap_score по убыванию
    assert list(top["gap_score"]) == sorted(top["gap_score"], reverse=True)


def test_pick_top_topics_fills_rest_by_score(briefs_mod):
    df = _gap_df([
        ("t1", "tires", 100.0),
        ("t2", "tires", 90.0),
        ("r1", "repair", 50.0),
        ("t3", "tires", 40.0),
        ("r2", "repair", 30.0),
    ])
    top = briefs_mod.pick_top_topics(df, count=4, min_per_vertical=1)
    # гарантия: t1, r1; добор по скору: t2 (90), t3 (40)
    assert list(top["topic_cluster"]) == ["t1", "t2", "r1", "t3"]


def test_pick_top_topics_caps_at_count(briefs_mod):
    df = _gap_df([(f"t{i}", "tires", 100.0 - i) for i in range(10)])
    top = briefs_mod.pick_top_topics(df, count=3, min_per_vertical=3)
    assert len(top) == 3
    assert list(top["topic_cluster"]) == ["t0", "t1", "t2"]


def test_pick_top_topics_no_duplicates(briefs_mod):
    df = _gap_df([("t1", "tires", 10.0), ("r1", "repair", 9.0)])
    top = briefs_mod.pick_top_topics(df, count=10, min_per_vertical=3)
    assert list(top["topic_cluster"]) == ["t1", "r1"]
    assert top["topic_cluster"].is_unique


def test_pick_top_topics_empty_exits(briefs_mod):
    with pytest.raises(SystemExit):
        briefs_mod.pick_top_topics(_gap_df([]), count=10, min_per_vertical=3)


# ------------------------------------------------------------ cluster_materials

@pytest.fixture()
def materials_env(briefs_mod, tmp_path, monkeypatch):
    for name in ("topics_per_video", "transcripts", "subs"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(briefs_mod, "TOPICS_DIR", tmp_path / "topics_per_video")
    monkeypatch.setattr(briefs_mod, "TRANSCRIPTS_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(briefs_mod, "SUBS_DIR", tmp_path / "subs")
    return tmp_path


def _scored_index():
    return pd.DataFrame([
        {"video_id": "v1", "title": "Видео 1", "url": "https://y.tb/v1",
         "view_count": 1234567},
        {"video_id": "v2", "title": "Видео 2", "url": "https://y.tb/v2",
         "view_count": 100},
    ]).set_index("video_id")


BRIEF_SETTINGS = {"max_chars_per_transcript": 1000, "max_total_chars": 10000}


def test_cluster_materials_collects_questions_and_transcripts(
        briefs_mod, materials_env):
    (materials_env / "topics_per_video" / "v1.json").write_text(
        json.dumps({"audience_questions": ["Какие шины брать?",
                                           "Какие шины брать?",  # дубль
                                           42,                   # мусор
                                           "Сколько стоит?"]},
                   ensure_ascii=False), encoding="utf-8")
    (materials_env / "topics_per_video" / "v2.json").write_text(
        json.dumps({"audience_questions": "Один вопрос строкой"},
                   ensure_ascii=False), encoding="utf-8")
    (materials_env / "transcripts" / "v1.txt").write_text("whisper v1",
                                                          encoding="utf-8")
    (materials_env / "subs" / "v1.txt").write_text("subs v1 (не должен попасть)",
                                                   encoding="utf-8")
    (materials_env / "subs" / "v2.txt").write_text("subs v2", encoding="utf-8")

    q, tr = briefs_mod.cluster_materials(
        ["v1", "v2", "нет_такого"], _scored_index(), BRIEF_SETTINGS)

    # вопросы: дедуп, мусор отброшен, строка тоже принята
    assert q == ("- Какие шины брать?\n- Сколько стоит?\n- Один вопрос строкой")
    # транскрипт приоритетнее субтитров
    assert "whisper v1" in tr and "не должен попасть" not in tr
    assert "subs v2" in tr


def test_cluster_materials_no_video_identifiers_in_prompt_materials(
        briefs_mod, materials_env):
    """Правовая рамка: в материалах промпта нет video_id, URL и названий
    чужих видео — только обезличенные «материал N»."""
    (materials_env / "transcripts" / "dQw4w9WgXcQ.txt").write_text(
        "текст первый", encoding="utf-8")
    (materials_env / "subs" / "abc123XYZ_0.txt").write_text(
        "текст второй", encoding="utf-8")
    q, tr = briefs_mod.cluster_materials(
        ["dQw4w9WgXcQ", "abc123XYZ_0"], _scored_index(), BRIEF_SETTINGS)
    assert "dQw4w9WgXcQ" not in tr and "abc123XYZ_0" not in tr
    assert "--- материал 1 ---" in tr and "--- материал 2 ---" in tr
    assert "http" not in tr and "http" not in q


def test_cluster_materials_respects_total_chars_cap(briefs_mod, materials_env):
    (materials_env / "transcripts" / "v1.txt").write_text("a" * 900,
                                                          encoding="utf-8")
    (materials_env / "transcripts" / "v2.txt").write_text("b" * 900,
                                                          encoding="utf-8")
    settings = {"max_chars_per_transcript": 1000, "max_total_chars": 1000}
    q, tr = briefs_mod.cluster_materials(["v1", "v2"], _scored_index(),
                                         settings)
    # второй транскрипт не влез в общий кап
    assert "a" * 900 in tr
    assert "b" not in tr


def test_cluster_materials_per_transcript_cap(briefs_mod, materials_env):
    (materials_env / "transcripts" / "v1.txt").write_text("x" * 5000,
                                                          encoding="utf-8")
    settings = {"max_chars_per_transcript": 100, "max_total_chars": 10000}
    _, tr = briefs_mod.cluster_materials(["v1"], _scored_index(), settings)
    body = tr.split("---\n", 1)[1]
    assert len(body) == 100


def test_cluster_materials_caps_questions_at_15(briefs_mod, materials_env):
    ids = []
    for i in range(20):
        vid = f"q{i}"
        ids.append(vid)
        (materials_env / "topics_per_video" / f"{vid}.json").write_text(
            json.dumps({"audience_questions": [f"Вопрос {i}?"]},
                       ensure_ascii=False), encoding="utf-8")
    q, _ = briefs_mod.cluster_materials(ids, _scored_index(), BRIEF_SETTINGS)
    assert len(q.splitlines()) == 15


def test_cluster_materials_empty_defaults(briefs_mod, materials_env):
    q, tr = briefs_mod.cluster_materials(["призрак"], _scored_index(),
                                         BRIEF_SETTINGS)
    assert q == "- (не собраны)"
    assert tr == ""
