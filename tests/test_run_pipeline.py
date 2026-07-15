# -*- coding: utf-8 -*-
"""Тесты run_pipeline.py: выбор этапов (--from/--to/--only), стоп на ошибке,
preflight-проверки (channels.yaml, GROQ_API_KEY). Подпроцессы замоканы."""

import sys
from types import SimpleNamespace

import pytest


# ------------------------------------------------------------- выбор этапов

@pytest.fixture()
def runner(pipeline_mod, monkeypatch):
    """Мокает subprocess.run и preflight, возвращает список запущенных скриптов."""
    launched = []

    def fake_run(cmd, cwd=None, **kw):
        launched.append(cmd[-1].split("\\")[-1].split("/")[-1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline_mod, "preflight", lambda first, last: None)
    return launched


def _run(pipeline_mod, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"] + argv)
    pipeline_mod.main()


ALL_STAGES = ["01_harvest.py", "02_score.py", "03_subtitles.py",
              "04_transcribe.py", "05_topics.py", "06_gap.py", "07_briefs.py"]


def test_default_runs_all_stages_in_order(pipeline_mod, monkeypatch, runner):
    _run(pipeline_mod, monkeypatch, [])
    assert runner == ALL_STAGES


def test_from_runs_tail(pipeline_mod, monkeypatch, runner):
    _run(pipeline_mod, monkeypatch, ["--from", "5"])
    assert runner == ["05_topics.py", "06_gap.py", "07_briefs.py"]


def test_to_runs_head(pipeline_mod, monkeypatch, runner):
    _run(pipeline_mod, monkeypatch, ["--to", "3"])
    assert runner == ["01_harvest.py", "02_score.py", "03_subtitles.py"]


def test_from_to_runs_window(pipeline_mod, monkeypatch, runner):
    _run(pipeline_mod, monkeypatch, ["--from", "2", "--to", "4"])
    assert runner == ["02_score.py", "03_subtitles.py", "04_transcribe.py"]


def test_only_runs_single_stage(pipeline_mod, monkeypatch, runner):
    _run(pipeline_mod, monkeypatch, ["--only", "6"])
    assert runner == ["06_gap.py"]


def test_only_overrides_from_and_to(pipeline_mod, monkeypatch, runner):
    _run(pipeline_mod, monkeypatch, ["--from", "1", "--to", "7", "--only", "2"])
    assert runner == ["02_score.py"]


def test_stops_on_stage_failure_with_exit_code(pipeline_mod, monkeypatch):
    launched = []

    def fake_run(cmd, cwd=None, **kw):
        script = cmd[-1].split("\\")[-1].split("/")[-1]
        launched.append(script)
        code = 3 if script == "02_score.py" else 0
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(pipeline_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(pipeline_mod, "preflight", lambda first, last: None)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"])
    with pytest.raises(SystemExit) as exc:
        pipeline_mod.main()
    assert exc.value.code == 3
    # после упавшего этапа ничего не запускалось
    assert launched == ["01_harvest.py", "02_score.py"]


# ----------------------------------------------------------------- preflight

@pytest.fixture()
def fake_root(pipeline_mod, tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    monkeypatch.setattr(pipeline_mod, "ROOT", tmp_path)
    return tmp_path


def test_preflight_fails_without_channels_yaml(pipeline_mod, fake_root,
                                               monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    with pytest.raises(SystemExit) as exc:
        pipeline_mod.preflight(1, 4)
    assert "channels.yaml" in str(exc.value)


def test_preflight_channels_not_needed_when_starting_later(pipeline_mod,
                                                           fake_root):
    # старт с этапа 2 не требует channels.yaml, LLM-ключ до этапа 4 не нужен
    pipeline_mod.preflight(2, 4)


def test_preflight_requires_groq_key_for_llm_stages(pipeline_mod, fake_root,
                                                    monkeypatch):
    (fake_root / "config" / "channels.yaml").write_text("tires:\n - u",
                                                        encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        pipeline_mod.preflight(1, 7)
    assert "GROQ_API_KEY" in str(exc.value)


def test_preflight_accepts_key_from_env_var(pipeline_mod, fake_root,
                                            monkeypatch):
    (fake_root / "config" / "channels.yaml").write_text("tires:\n - u",
                                                        encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    pipeline_mod.preflight(1, 7)  # не должно падать


def test_preflight_accepts_key_from_dotenv_file(pipeline_mod, fake_root,
                                                monkeypatch):
    (fake_root / "config" / "channels.yaml").write_text("tires:\n - u",
                                                        encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    (fake_root / ".env").write_text("GROQ_API_KEY=gsk_y\n", encoding="utf-8")
    pipeline_mod.preflight(1, 7)  # ключ найден в .env — не падаем


def test_preflight_no_llm_key_needed_up_to_stage_4(pipeline_mod, fake_root,
                                                   monkeypatch):
    (fake_root / "config" / "channels.yaml").write_text("tires:\n - u",
                                                        encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    pipeline_mod.preflight(1, 4)  # --to 4 работает без ключа
