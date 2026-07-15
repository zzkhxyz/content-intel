# -*- coding: utf-8 -*-
"""Тесты src/06_gap.py: gap_score = ru_views / (kz_views + 1000),
KZ-классификация (маркеры + LLM), дисковый кэш поиска.

Сеть и LLM полностью замоканы, все пути уведены в tmp_path.
"""

import json

import pandas as pd
import pytest

SETTINGS = {"gap": {
    "search_results": 5,
    "top_clusters": 25,
    "chart_top": 15,
    "kz_markers": ["Алматы", "тенге", ".kz"],
}}


# ----------------------------------------------------------------- classify_kz

def test_classify_kz_markers_case_insensitive(gap_mod, monkeypatch, dummy_logger):
    monkeypatch.setattr(gap_mod, "ask_llm_json",
                        lambda *a, **kw: {"kz_indices": []})
    results = [
        {"title": "Обзор шин", "channel": "Канал РФ", "view_count": 10},
        {"title": "Шины в АЛМАТЫ дёшево", "channel": "x", "view_count": 20},
        {"title": "цена 5000 Тенге", "channel": "y", "view_count": 30},
        {"title": "site", "channel": "kolesa.KZ", "view_count": 40},
    ]
    kz = gap_mod.classify_kz("тема", results, ["Алматы", "тенге", ".kz"],
                             dummy_logger)
    assert kz == {1, 2, 3}


def test_classify_kz_merges_llm_indices_and_validates(gap_mod, monkeypatch,
                                                      dummy_logger):
    # LLM добавляет валидные индексы (в т.ч. строкой), мусор отбрасывается
    monkeypatch.setattr(
        gap_mod, "ask_llm_json",
        lambda *a, **kw: {"kz_indices": [0, "1", -1, 99, "мусор", 2.5]})
    results = [{"title": f"v{i}", "channel": "c", "view_count": 1}
               for i in range(3)]
    kz = gap_mod.classify_kz("тема", results, ["тенге"], dummy_logger)
    assert kz == {0, 1}


def test_classify_kz_survives_llm_failure(gap_mod, monkeypatch, dummy_logger):
    def boom(*a, **kw):
        raise RuntimeError("Groq упал")

    monkeypatch.setattr(gap_mod, "ask_llm_json", boom)
    results = [
        {"title": "шины Алматы", "channel": "c", "view_count": 1},
        {"title": "шины Москва", "channel": "c", "view_count": 1},
    ]
    # LLM упал — остаёмся на маркерах, исключения наружу нет
    assert gap_mod.classify_kz("т", results, ["Алматы"], dummy_logger) == {0}


def test_classify_kz_survives_llm_returning_list(gap_mod, monkeypatch,
                                                 dummy_logger):
    # extract_json может вернуть список — .get упадёт, но except прикрывает
    monkeypatch.setattr(gap_mod, "ask_llm_json", lambda *a, **kw: [1, 2])
    results = [{"title": "тенге", "channel": "c", "view_count": 1},
               {"title": "рубли", "channel": "c", "view_count": 1}]
    assert gap_mod.classify_kz("т", results, ["тенге"], dummy_logger) == {0}


# -------------------------------------------------------------- search_youtube

@pytest.fixture()
def search_env(gap_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(gap_mod, "SEARCH_CACHE_DIR", tmp_path / "gap_search")
    return tmp_path


def test_search_youtube_caches_nonempty_results(gap_mod, search_env,
                                                monkeypatch, dummy_logger):
    calls = []

    def fake_ytdlp(args, logger=None, timeout=1800):
        calls.append(args)
        return [
            {"id": "x1", "title": "Видео", "channel": "Канал", "view_count": 7},
            {"id": None, "title": "без id — отбрасывается"},
            {"id": "x2", "title": None, "uploader": "Автор"},  # channel из uploader
        ]

    monkeypatch.setattr(gap_mod, "ytdlp_json_lines", fake_ytdlp)

    first = gap_mod.search_youtube("шины алматы", 5, dummy_logger)
    assert first == [
        {"id": "x1", "title": "Видео", "channel": "Канал", "view_count": 7},
        {"id": "x2", "title": "", "channel": "Автор", "view_count": 0},
    ]
    assert len(calls) == 1
    assert any("ytsearch5:шины алматы" in str(a) for a in calls[0])

    # повторный вызов идёт из кэша, второго похода в сеть нет
    second = gap_mod.search_youtube("шины алматы", 5, dummy_logger)
    assert second == first
    assert len(calls) == 1
    assert list((search_env / "gap_search").glob("*.json"))


def test_search_youtube_does_not_cache_empty(gap_mod, search_env, monkeypatch,
                                             dummy_logger):
    calls = []

    def fake_ytdlp(args, logger=None, timeout=1800):
        calls.append(args)
        return []

    monkeypatch.setattr(gap_mod, "ytdlp_json_lines", fake_ytdlp)
    assert gap_mod.search_youtube("запрос", 3, dummy_logger) == []
    # пустота не закэширована: второй вызов снова идёт в yt-dlp
    assert gap_mod.search_youtube("запрос", 3, dummy_logger) == []
    assert len(calls) == 2
    assert not list((search_env / "gap_search").glob("*.json"))


def test_search_youtube_cache_key_depends_on_n(gap_mod, search_env, monkeypatch,
                                               dummy_logger):
    calls = []

    def fake_ytdlp(args, logger=None, timeout=1800):
        calls.append(args)
        return [{"id": f"id{len(calls)}", "title": "t", "channel": "c",
                 "view_count": 1}]

    monkeypatch.setattr(gap_mod, "ytdlp_json_lines", fake_ytdlp)
    gap_mod.search_youtube("q", 5, dummy_logger)
    gap_mod.search_youtube("q", 10, dummy_logger)  # другой n -> другой ключ
    assert len(calls) == 2


# ------------------------------------------------- main(): формула gap_score

@pytest.fixture()
def gap_env(gap_mod, tmp_path, monkeypatch, no_real_logs):
    monkeypatch.setattr(gap_mod, "TOPICS_CSV", tmp_path / "topics.csv")
    monkeypatch.setattr(gap_mod, "SEARCH_CACHE_DIR", tmp_path / "gap_search")
    monkeypatch.setattr(gap_mod, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(gap_mod, "GAP_TABLE", tmp_path / "output" / "gap_table.csv")
    monkeypatch.setattr(gap_mod, "GAP_PNG", tmp_path / "output" / "gap.png")
    monkeypatch.setattr(gap_mod, "load_settings", lambda: SETTINGS)
    monkeypatch.setattr(gap_mod, "setup_logger", no_real_logs)
    return tmp_path


def test_main_gap_score_formula(gap_mod, gap_env, monkeypatch):
    pd.DataFrame([
        {"topic_cluster": "Китайская зимняя резина", "vertical": "tires",
         "videos_count": 5, "total_views": 1_000_000, "median_vpd": 10.0,
         "max_outlier_score": 3.0, "video_ids": "a;b"},
        {"topic_cluster": "Санузел под ключ", "vertical": "repair",
         "videos_count": 2, "total_views": 200_000, "median_vpd": 5.0,
         "max_outlier_score": 2.0, "video_ids": "c"},
    ]).to_csv(gap_env / "topics.csv", index=False, encoding="utf-8")

    # запросы: для темы 0 — от LLM, темы 1 нет в ответе -> fallback на имя темы
    monkeypatch.setattr(
        gap_mod, "ask_llm_json",
        lambda *a, **kw: {"queries": [
            {"index": 0, "ru": ["зимняя резина обзор"], "kk": "қысқы шина"}]})

    searched = []

    def fake_search(query, n, logger):
        searched.append(query)
        if "резина" in query or "шина" in query:
            return [{"id": "kz1", "title": "шины Алматы", "channel": "c",
                     "view_count": 4000},
                    {"id": "ru1", "title": "шины Москва", "channel": "c",
                     "view_count": 999_999}]
        return []  # по теме 1 KZ-поле пустое

    monkeypatch.setattr(gap_mod, "search_youtube", fake_search)
    monkeypatch.setattr(gap_mod, "classify_kz",
                        lambda topic, results, markers, logger: {
                            i for i, r in enumerate(results)
                            if "Алматы" in r["title"]})

    gap_mod.main()

    out = pd.read_csv(gap_env / "output" / "gap_table.csv").set_index(
        "topic_cluster")
    r0 = out.loc["Китайская зимняя резина"]
    # gap_score = ru_views / (kz_views + 1000)
    assert r0["kz_views"] == 4000
    assert r0["kz_videos"] == 1
    assert r0["gap_score"] == pytest.approx(1_000_000 / (4000 + 1000))  # 200.0
    r1 = out.loc["Санузел под ключ"]
    assert r1["kz_views"] == 0
    assert r1["gap_score"] == pytest.approx(200_000 / 1000)  # 200.0 при нуле KZ
    # сортировка по gap_score убыв. и наличие графика
    assert (gap_env / "output" / "gap.png").exists()
    # строковый kk-запрос нормализован в список, fallback-запрос — имя темы
    assert "қысқы шина" in searched
    assert "Санузел под ключ" in searched


def test_main_dedupes_search_results_across_queries(gap_mod, gap_env,
                                                    monkeypatch):
    pd.DataFrame([
        {"topic_cluster": "Тема", "vertical": "tires", "videos_count": 1,
         "total_views": 100_000, "median_vpd": 1.0, "max_outlier_score": 1.0,
         "video_ids": "a"},
    ]).to_csv(gap_env / "topics.csv", index=False, encoding="utf-8")
    monkeypatch.setattr(
        gap_mod, "ask_llm_json",
        lambda *a, **kw: {"queries": [{"index": 0, "ru": ["q1", "q2"],
                                       "kk": []}]})
    # оба запроса возвращают одно и то же KZ-видео
    dup = {"id": "same", "title": "Алматы", "channel": "c", "view_count": 500}
    monkeypatch.setattr(gap_mod, "search_youtube",
                        lambda q, n, logger: [dict(dup)])
    monkeypatch.setattr(gap_mod, "classify_kz",
                        lambda topic, results, markers, logger: set(
                            range(len(results))))

    gap_mod.main()

    out = pd.read_csv(gap_env / "output" / "gap_table.csv")
    # дубль по id схлопнут: просмотры не удвоились
    assert out.loc[0, "kz_views"] == 500
    assert out.loc[0, "kz_videos"] == 1
    # в таблице значение округлено до 1 знака
    assert out.loc[0, "gap_score"] == pytest.approx(round(100_000 / 1500, 1))
