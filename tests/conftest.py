# -*- coding: utf-8 -*-
"""Общие фикстуры: загрузка модулей пайплайна через importlib.

Скрипты называются 02_score.py и т.п. (начинаются с цифры), поэтому обычный
import невозможен — грузим через importlib.util.spec_from_file_location.
Модули src/ делают `from utils import ...`, поэтому src добавлен в sys.path.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

CONTENT_INTEL = Path(__file__).resolve().parent.parent
SRC = CONTENT_INTEL / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def utils_mod():
    import utils
    return utils


@pytest.fixture(scope="session")
def score_mod():
    return load_module("score_02", SRC / "02_score.py")


@pytest.fixture(scope="session")
def topics_mod():
    return load_module("topics_05", SRC / "05_topics.py")


@pytest.fixture(scope="session")
def gap_mod():
    return load_module("gap_06", SRC / "06_gap.py")


@pytest.fixture(scope="session")
def briefs_mod():
    return load_module("briefs_07", SRC / "07_briefs.py")


@pytest.fixture(scope="session")
def pipeline_mod():
    return load_module("run_pipeline_mod", CONTENT_INTEL / "run_pipeline.py")


@pytest.fixture()
def dummy_logger():
    """Логгер без файловых хендлеров — тесты не трогают реальный logs/."""
    logger = logging.getLogger("tests.dummy")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


@pytest.fixture()
def no_real_logs(monkeypatch, dummy_logger):
    """Фабрика setup_logger, не пишущая в реальный logs/."""
    return lambda stage: dummy_logger
