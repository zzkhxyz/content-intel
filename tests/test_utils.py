# -*- coding: utf-8 -*-
"""Тесты src/utils.py: extract_json, ask_llm (кэш), ask_llm_json (ретрай),
load_channels, _log_usage, read_text, video_url. Без сети и без Groq."""

import hashlib
import json

import pytest


# ---------------------------------------------------------------- extract_json

class TestExtractJson:
    def test_plain_object(self, utils_mod):
        assert utils_mod.extract_json('{"a": 1}') == {"a": 1}

    def test_markdown_fence_json(self, utils_mod):
        text = 'Вот ответ:\n```json\n{"topic": "шины"}\n```\nГотово.'
        assert utils_mod.extract_json(text) == {"topic": "шины"}

    def test_markdown_fence_without_lang(self, utils_mod):
        text = '```\n[1, 2, 3]\n```'
        assert utils_mod.extract_json(text) == [1, 2, 3]

    def test_object_wrapped_in_prose(self, utils_mod):
        text = 'Конечно! Вот JSON: {"kz_indices": [3, 7]} — надеюсь, помог.'
        assert utils_mod.extract_json(text) == {"kz_indices": [3, 7]}

    def test_array_wrapped_in_prose(self, utils_mod):
        text = 'Ответ: ["a", "b"] конец'
        assert utils_mod.extract_json(text) == ["a", "b"]

    def test_nested_object(self, utils_mod):
        payload = {"clusters": [{"name": "n", "video_ids": ["x"]}]}
        text = "прелюдия " + json.dumps(payload, ensure_ascii=False) + " постлюдия"
        assert utils_mod.extract_json(text) == payload

    def test_invalid_raises_value_error(self, utils_mod):
        with pytest.raises(ValueError):
            utils_mod.extract_json("тут вообще нет JSON")

    def test_broken_json_raises(self, utils_mod):
        with pytest.raises(ValueError):
            utils_mod.extract_json('{"a": 1,,,}')

    def test_fence_has_priority_over_outer_braces(self, utils_mod):
        # мусорные скобки вокруг валидного fenced-блока не должны мешать
        text = '{ битый json\n```json\n{"ok": true}\n```\n}'
        assert utils_mod.extract_json(text) == {"ok": True}


# --------------------------------------------------------------------- разное

def test_video_url(utils_mod):
    assert utils_mod.video_url("abc123") == "https://www.youtube.com/watch?v=abc123"


def test_read_text_full_and_limited(utils_mod, tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("привет мир", encoding="utf-8")
    assert utils_mod.read_text(p) == "привет мир"
    assert utils_mod.read_text(p, limit=6) == "привет"


def test_read_text_replaces_bad_bytes(utils_mod, tmp_path):
    p = tmp_path / "bad.txt"
    p.write_bytes(b"ok \xff\xfe bad")
    text = utils_mod.read_text(p)
    assert text.startswith("ok ")  # не падает на битой кодировке


# --------------------------------------------------------------- load_channels

def test_load_channels_filters_empty_and_returns_dict(utils_mod, tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "channels.yaml").write_text(
        "tires:\n  - https://youtube.com/@a\nrepair:\n", encoding="utf-8")
    monkeypatch.setattr(utils_mod, "CONFIG_DIR", cfg)
    channels = utils_mod.load_channels()
    assert channels == {"tires": ["https://youtube.com/@a"]}


def test_load_channels_exits_when_empty(utils_mod, tmp_path, monkeypatch):
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "channels.yaml").write_text("tires:\n", encoding="utf-8")
    monkeypatch.setattr(utils_mod, "CONFIG_DIR", cfg)
    with pytest.raises(SystemExit):
        utils_mod.load_channels()


# ------------------------------------------------------------------ _log_usage

def test_log_usage_writes_jsonl(utils_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(utils_mod, "LOGS_DIR", tmp_path / "logs")

    class Usage:
        prompt_tokens = 11
        completion_tokens = 22

    utils_mod._log_usage("m1", Usage())
    utils_mod._log_usage("m2", None)  # usage может отсутствовать

    lines = (tmp_path / "logs" / "llm_usage.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    rec1, rec2 = (json.loads(x) for x in lines)
    assert rec1["model"] == "m1"
    assert rec1["input_tokens"] == 11 and rec1["output_tokens"] == 22
    assert rec2["input_tokens"] == 0 and rec2["output_tokens"] == 0


# ------------------------------------------------------- ask_llm: дисковый кэш

FAKE_SETTINGS = {"models": {"llm": "test-model", "llm_light": "test-light",
                            "max_tokens_default": 100,
                            "sleep_between_calls": 0}}


def _cache_key(model, system, prompt, max_tokens):
    return hashlib.sha256(
        json.dumps([model, system, prompt, max_tokens],
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()


@pytest.fixture()
def llm_env(utils_mod, tmp_path, monkeypatch):
    """Изолирует кэш/логи/настройки ask_llm в tmp_path."""
    pytest.importorskip("groq")
    monkeypatch.setattr(utils_mod, "LLM_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(utils_mod, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(utils_mod, "load_settings", lambda: FAKE_SETTINGS)
    return tmp_path


def test_ask_llm_cache_hit_skips_groq(utils_mod, llm_env, monkeypatch):
    import groq
    key = _cache_key("test-model", None, "вопрос", 100)
    cache_dir = llm_env / "cache"
    cache_dir.mkdir()
    (cache_dir / f"{key}.json").write_text(
        json.dumps({"text": "кэшированный ответ"}, ensure_ascii=False),
        encoding="utf-8")

    def boom(*a, **kw):
        raise AssertionError("Groq не должен вызываться при кэш-хите")

    monkeypatch.setattr(groq, "Groq", boom)
    assert utils_mod.ask_llm("вопрос") == "кэшированный ответ"


def _make_fake_groq(content="ответ модели", finish_reason="stop"):
    state = {"calls": 0}

    class Choice:
        pass

    class Msg:
        pass

    class Resp:
        pass

    class Client:
        def __init__(self, **kw):
            class Completions:
                @staticmethod
                def create(**kwargs):
                    state["calls"] += 1
                    msg = Msg()
                    msg.content = content
                    ch = Choice()
                    ch.message = msg
                    ch.finish_reason = finish_reason
                    resp = Resp()
                    resp.choices = [ch]
                    resp.usage = None
                    return resp

            class Chat:
                completions = Completions()

            self.chat = Chat()

    return Client, state


def test_ask_llm_writes_cache_on_success(utils_mod, llm_env, monkeypatch):
    import groq
    client_cls, state = _make_fake_groq("свежий ответ")
    monkeypatch.setattr(groq, "Groq", client_cls)

    assert utils_mod.ask_llm("новый вопрос") == "свежий ответ"
    assert state["calls"] == 1

    key = _cache_key("test-model", None, "новый вопрос", 100)
    cache_file = llm_env / "cache" / f"{key}.json"
    assert cache_file.exists()
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {"text": "свежий ответ"}

    # повторный вызов берётся из кэша, второго обращения к API нет
    assert utils_mod.ask_llm("новый вопрос") == "свежий ответ"
    assert state["calls"] == 1


def test_ask_llm_truncated_answer_not_cached(utils_mod, llm_env, monkeypatch,
                                             dummy_logger):
    import groq
    client_cls, state = _make_fake_groq("обруб", finish_reason="length")
    monkeypatch.setattr(groq, "Groq", client_cls)

    assert utils_mod.ask_llm("вопрос", logger=dummy_logger) == "обруб"
    key = _cache_key("test-model", None, "вопрос", 100)
    assert not (llm_env / "cache" / f"{key}.json").exists()


def test_ask_llm_model_key_fallback(utils_mod, llm_env, monkeypatch):
    """Неизвестный model_key откатывается на models.llm."""
    import groq
    seen = {}

    client_cls, state = _make_fake_groq("x")

    class Client(client_cls):
        def __init__(self, **kw):
            super().__init__(**kw)
            create = self.chat.completions.create

            def wrapper(**kwargs):
                seen.update(kwargs)
                return create(**kwargs)
            self.chat.completions.create = wrapper

    monkeypatch.setattr(groq, "Groq", Client)
    utils_mod.ask_llm("q1", model_key="no_such_key")
    assert seen["model"] == "test-model"


# ------------------------------------------ ask_llm_json: ремонт JSON + ретрай

def test_ask_llm_json_ok_first_try(utils_mod, monkeypatch):
    monkeypatch.setattr(utils_mod, "ask_llm",
                        lambda prompt, **kw: '{"a": 1}')
    assert utils_mod.ask_llm_json("p") == {"a": 1}


def test_ask_llm_json_retry_on_invalid(utils_mod, monkeypatch, dummy_logger):
    calls = []

    def fake_ask_llm(prompt, system=None, max_tokens=None, logger=None,
                     model_key="llm", **kw):
        calls.append({"prompt": prompt, "max_tokens": max_tokens})
        return "не json" if len(calls) == 1 else '{"ok": true}'

    monkeypatch.setattr(utils_mod, "ask_llm", fake_ask_llm)
    result = utils_mod.ask_llm_json("исходный промпт", max_tokens=1000,
                                    logger=dummy_logger)
    assert result == {"ok": True}
    assert len(calls) == 2
    # ретрай: тот же промпт + строгий суффикс, удвоенный max_tokens
    assert calls[1]["prompt"].startswith("исходный промпт")
    assert "СТРОГО валидным JSON" in calls[1]["prompt"]
    assert calls[1]["max_tokens"] == 2000


def test_ask_llm_json_retry_caps_at_16000(utils_mod, monkeypatch, dummy_logger):
    calls = []

    def fake_ask_llm(prompt, system=None, max_tokens=None, logger=None,
                     model_key="llm", **kw):
        calls.append(max_tokens)
        return "мусор" if len(calls) == 1 else "[]"

    monkeypatch.setattr(utils_mod, "ask_llm", fake_ask_llm)
    assert utils_mod.ask_llm_json("p", max_tokens=12000, logger=dummy_logger) == []
    assert calls[1] == 16000  # min(12000*2, 16000)


def test_ask_llm_json_raises_if_retry_also_invalid(utils_mod, monkeypatch,
                                                   dummy_logger):
    monkeypatch.setattr(utils_mod, "ask_llm", lambda prompt, **kw: "всё ещё не json")
    with pytest.raises(ValueError):
        utils_mod.ask_llm_json("p", max_tokens=500, logger=dummy_logger)
