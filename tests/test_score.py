# -*- coding: utf-8 -*-
"""Тесты src/02_score.py: outlier_score = vpd / медиана канала, порог хита 2.0,
engagement, восстановление subs_missing по наличию .txt на диске.

main() запускается целиком, но все пути (CSV, subs, логи) уведены в tmp_path.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

SETTINGS = {"scoring": {"hit_outlier_score": 2.0}}


def _upload_date(days_ago: int) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")


@pytest.fixture()
def score_env(score_mod, tmp_path, monkeypatch, no_real_logs):
    """Уводит все пути этапа 02 в tmp_path и подсовывает настройки."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (tmp_path / "subs").mkdir()
    monkeypatch.setattr(score_mod, "DATA_DIR", tmp_path)
    monkeypatch.setattr(score_mod, "RAW_CSV", raw_dir / "videos.csv")
    monkeypatch.setattr(score_mod, "SCORED_DIR", tmp_path / "scored")
    monkeypatch.setattr(score_mod, "load_settings", lambda: SETTINGS)
    monkeypatch.setattr(score_mod, "setup_logger", no_real_logs)
    return tmp_path


def _write_raw(tmp_path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "raw" / "videos.csv", index=False, encoding="utf-8")


def _base_row(**over):
    row = {
        "video_id": "v1", "title": "t", "channel": "ch", "vertical": "tires",
        "upload_date": _upload_date(10), "view_count": 100,
        "like_count": 0, "comment_count": 0,
        "url": "https://youtube.com/watch?v=v1",
    }
    row.update(over)
    return row


def test_outlier_score_vs_channel_median_and_hit_threshold(score_mod, score_env):
    # один канал, одинаковый возраст роликов: outlier = views / median(views)
    _write_raw(score_env, [
        _base_row(video_id="a", view_count=100),
        _base_row(video_id="b", view_count=200),
        _base_row(video_id="c", view_count=800),
    ])
    score_mod.main()

    out = pd.read_csv(score_env / "scored" / "videos.csv",
                      dtype={"video_id": str}).set_index("video_id")
    assert out.loc["a", "outlier_score"] == pytest.approx(0.5)
    assert out.loc["b", "outlier_score"] == pytest.approx(1.0)
    assert out.loc["c", "outlier_score"] == pytest.approx(4.0)
    # порог хита 2.0: хит только у x4
    assert bool(out.loc["c", "is_hit"]) is True
    assert bool(out.loc["a", "is_hit"]) is False
    assert bool(out.loc["b", "is_hit"]) is False
    # сортировка по убыванию outlier_score
    assert list(out.index) == ["c", "b", "a"]


def test_outlier_score_exactly_two_is_hit(score_mod, score_env):
    # порог включительный: outlier_score == 2.0 -> хит
    _write_raw(score_env, [
        _base_row(video_id="m1", view_count=100),
        _base_row(video_id="m2", view_count=100),
        _base_row(video_id="hit", view_count=200),
    ])
    score_mod.main()
    out = pd.read_csv(score_env / "scored" / "videos.csv",
                      dtype={"video_id": str}).set_index("video_id")
    assert out.loc["hit", "outlier_score"] == pytest.approx(2.0)
    assert bool(out.loc["hit", "is_hit"]) is True


def test_median_is_per_channel_not_global(score_mod, score_env):
    # большой канал не должен «прибивать» outlier маленького
    _write_raw(score_env, [
        _base_row(video_id="b1", channel="big", view_count=1_000_000),
        _base_row(video_id="b2", channel="big", view_count=1_000_000),
        _base_row(video_id="s1", channel="small", view_count=100),
        _base_row(video_id="s2", channel="small", view_count=400),
    ])
    score_mod.main()
    out = pd.read_csv(score_env / "scored" / "videos.csv",
                      dtype={"video_id": str}).set_index("video_id")
    # медиана small = 250 -> 400/250 = 1.6, независимо от канала big
    assert out.loc["s2", "outlier_score"] == pytest.approx(1.6)
    assert out.loc["b1", "outlier_score"] == pytest.approx(1.0)


def test_vpd_uses_days_live(score_mod, score_env):
    # 1000 просмотров за 10 дней = 100 vpd; свежий ролик clip(lower=1)
    _write_raw(score_env, [
        _base_row(video_id="old", view_count=1000, upload_date=_upload_date(10)),
        _base_row(video_id="new", view_count=50, upload_date=_upload_date(0)),
    ])
    score_mod.main()
    out = pd.read_csv(score_env / "scored" / "videos.csv",
                      dtype={"video_id": str}).set_index("video_id")
    assert out.loc["old", "vpd"] == pytest.approx(100.0)
    assert out.loc["old", "days_live"] == 10
    # days_live не бывает меньше 1 — деления на ноль нет
    assert out.loc["new", "days_live"] == 1
    assert out.loc["new", "vpd"] == pytest.approx(50.0)


def test_engagement_formula_and_nan_counts(score_mod, score_env):
    _write_raw(score_env, [
        _base_row(video_id="e1", view_count=1000, like_count=30, comment_count=10),
        _base_row(video_id="e2", view_count=1000, like_count=None,
                  comment_count=None),
    ])
    score_mod.main()
    out = pd.read_csv(score_env / "scored" / "videos.csv",
                      dtype={"video_id": str}).set_index("video_id")
    assert out.loc["e1", "engagement"] == pytest.approx(0.04)  # (30+10)/1000
    assert out.loc["e2", "engagement"] == pytest.approx(0.0)   # NaN -> 0


def test_subs_missing_restored_from_disk(score_mod, score_env):
    _write_raw(score_env, [
        _base_row(video_id="has_subs", view_count=100),
        _base_row(video_id="no_subs", view_count=100),
    ])
    (score_env / "subs" / "has_subs.txt").write_text("текст", encoding="utf-8")
    score_mod.main()
    out = pd.read_csv(score_env / "scored" / "videos.csv",
                      dtype={"video_id": str}).set_index("video_id")
    assert bool(out.loc["has_subs", "subs_missing"]) is False
    assert bool(out.loc["no_subs", "subs_missing"]) is True
