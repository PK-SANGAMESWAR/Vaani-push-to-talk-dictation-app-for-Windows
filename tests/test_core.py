"""Unit tests that run anywhere (no GPU, mic, or Ollama needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vaani import cleanup
from vaani.cleanup import Cleaner, is_plausible
from vaani.config import load_config
from vaani.textproc import TextProcessor

ROOT = Path(__file__).resolve().parent.parent


# ---------- config ----------


def test_shipped_config_loads():
    cfg = load_config(ROOT / "config.toml")
    assert cfg.hotkey == "right ctrl"
    assert cfg.mode == "hold"
    assert cfg.replacements["pie torch"] == "PyTorch"
    assert cfg.asr_model == "nemo-parakeet-tdt-0.6b-v3"


def test_missing_config_uses_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.toml")
    assert cfg.sample_rate == 16_000 and cfg.llm_enabled


def test_hold_mode_rejects_combo(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[hotkey]\nkey = "ctrl+windows"\nmode = "hold"\n')
    with pytest.raises(ValueError):
        load_config(p)


# ---------- text processing ----------


def test_replacements_longest_first_and_case_insensitive():
    tp = TextProcessor({"pie torch": "PyTorch", "pie torch geometric": "PyTorch Geometric"})
    assert tp.replace("I use Pie Torch Geometric and pie torch.") == "I use PyTorch Geometric and PyTorch."


def test_replacements_respect_word_boundaries():
    tp = TextProcessor({"g n n": "GNN"})
    assert tp.replace("a g n n model") == "a GNN model"
    assert tp.replace("ag n nx") == "ag n nx"


def test_snippet_matches_whole_utterance_only():
    tp = TextProcessor(snippets={"insert my signature": "Best,\nSangu"})
    assert tp.snippet("Insert my signature.") == "Best,\nSangu"
    assert tp.snippet("please insert my signature here") is None


# ---------- cleanup guard ----------


def test_plausible_accepts_normal_cleanup():
    raw = "um so i think we should uh ship the model on friday you know"
    assert is_plausible(raw, "I think we should ship the model on Friday.")


def test_plausible_rejects_llm_answering():
    raw = "what is the capital of france"
    answer = "The capital of France is Paris. It is known for the Eiffel Tower and its museums and food."
    assert not is_plausible(raw, answer)


def test_plausible_rejects_empty():
    assert not is_plausible("hello there friend", "")


def test_cleaner_falls_back_when_ollama_down():
    c = Cleaner(url="http://127.0.0.1:9", timeout_s=0.5)
    assert c.clean("this text should come back unchanged ok") == "this text should come back unchanged ok"


def test_cleaner_uses_llm_output(monkeypatch):
    def fake_post(self, path, payload, timeout):
        assert path == "/api/chat"
        assert payload["options"]["temperature"] == 0
        assert "<transcript>" in payload["messages"][1]["content"]
        assert "think" not in payload
        return {"message": {"content": "<transcript>So we ship on Friday.</transcript>"}}

    monkeypatch.setattr(Cleaner, "_post", fake_post)
    assert Cleaner().clean("so um we ship on friday") == "So we ship on Friday."


def test_cleaner_vocab_in_prompt():
    c = Cleaner(vocabulary=["PyTorch", "LangGraph"])
    assert "PyTorch, LangGraph" in c.system
    assert json.dumps(c.system)  # serialisable
    assert cleanup.SYSTEM_PROMPT in c.system


def _fake_chat(content):
    def fake_post(self, path, payload, timeout):
        return {"message": {"content": content}}

    return fake_post


def test_cleaner_think_flag_and_token_cap(monkeypatch):
    seen = {}

    def fake_post(self, path, payload, timeout):
        seen.update(payload)
        return {"message": {"content": "ok"}}

    monkeypatch.setattr(Cleaner, "_post", fake_post)
    Cleaner(think_flag=True).clean("one two three four five six seven eight")
    assert seen["think"] is False and seen["options"]["num_predict"] == 64 and seen["stream"] is False


def test_cleaner_strips_wrapping_quotes(monkeypatch):
    monkeypatch.setattr(Cleaner, "_post", _fake_chat('"We ship on Friday, okay."'))
    assert Cleaner().clean("we ship on friday okay") == "We ship on Friday, okay."


def test_cleaner_returns_raw_when_output_implausible(monkeypatch):
    raw = "what is the capital of france and why is it famous"
    monkeypatch.setattr(Cleaner, "_post", _fake_chat("Paris."))
    assert Cleaner().clean(raw) == raw


@pytest.mark.parametrize("bad", [{}, {"message": {}}])
def test_cleaner_returns_raw_on_malformed_response(monkeypatch, bad):
    monkeypatch.setattr(Cleaner, "_post", lambda self, p, pl, timeout: bad)
    assert Cleaner().clean("keep this exact text please ok") == "keep this exact text please ok"


def test_cleaner_returns_raw_on_timeout(monkeypatch):
    def slow(self, path, payload, timeout):
        raise TimeoutError

    monkeypatch.setattr(Cleaner, "_post", slow)
    assert Cleaner().clean("keep this exact text please ok") == "keep this exact text please ok"


def test_prewarm_reports_success_and_failure(monkeypatch):
    monkeypatch.setattr(Cleaner, "_post", lambda self, p, pl, timeout: {})
    assert Cleaner().prewarm() is True

    def down(self, p, pl, timeout):
        raise OSError("down")

    monkeypatch.setattr(Cleaner, "_post", down)
    assert Cleaner().prewarm() is False


def test_cleaner_only_talks_to_configured_local_url(monkeypatch):
    urls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"message": {"content": "hi there"}}'

    monkeypatch.setattr(
        cleanup.urllib.request, "urlopen", lambda req, timeout: urls.append(req.full_url) or Resp()
    )
    Cleaner().clean("hi there friend")
    assert urls == ["http://127.0.0.1:11434/api/chat"]


def test_plausible_short_utterances_allow_small_growth():
    assert is_plausible("hi", "Hi there!")
    assert not is_plausible("hi", "Hello, how can I help you with your day today?")


def test_rejection_log_does_not_leak_transcript(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        is_plausible(
            "secret words go right here today",
            "leaky output text that is far too long to be a cleanup of the input",
        )
    assert "rejected" in caplog.text
    assert "leaky" not in caplog.text and "secret" not in caplog.text


# ---------- config (more) ----------


def test_every_config_field_is_in_sections_and_documented():
    from dataclasses import fields

    from vaani.config import _SECTIONS, Config

    in_sections = {k for keys in _SECTIONS.values() for k in keys}
    free = {"replacements", "snippets", "path"}
    names = {f.name for f in fields(Config)} - free
    assert names == in_sections
    toml = (ROOT / "config.toml").read_text(encoding="utf-8")
    for name in names - {"hotkey"}:  # "hotkey" is documented under its friendlier alias `key`
        assert name in toml, f"{name} is not documented in config.toml"
    assert "key  =" in toml or "key =" in toml


def test_unknown_and_misplaced_keys_are_ignored(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[hotkey]\nbogus = 1\nmode = "toggle"\n[audio]\nmode = "hold"\nsample_rate = 8000\n')
    cfg = load_config(p)
    assert cfg.mode == "toggle" and cfg.sample_rate == 8000


def test_toggle_mode_allows_combo(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[hotkey]\nkey = "ctrl+windows"\nmode = "toggle"\n')
    assert load_config(p).hotkey == "ctrl+windows"


@pytest.mark.parametrize(
    "body",
    ['[hotkey]\nmode = "double"', '[asr]\ndevice = "tpu"', '[output]\npaste_method = "telepathy"'],
)
def test_invalid_enum_values_raise(tmp_path, body):
    p = tmp_path / "c.toml"
    p.write_text(body)
    with pytest.raises(ValueError):
        load_config(p)


def test_replacements_and_snippets_are_loaded_as_strings(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[replacements]\n"a b" = "AB"\n[snippets]\n"my mail" = "x@y.z"\n')
    cfg = load_config(p)
    assert cfg.replacements == {"a b": "AB"} and cfg.snippets == {"my mail": "x@y.z"}


def test_shipped_config_has_safe_defaults():
    cfg = load_config(ROOT / "config.toml")
    assert cfg.save_history is False  # history is opt-in
    assert cfg.llm_url.startswith("http://127.0.0.1")  # only localhost is allowed
    assert cfg.llm_min_words >= 1  # the LLM never runs on tiny utterances by default


# ---------- text processing (more) ----------


def test_vocabulary_is_sorted_and_deduplicated():
    tp = TextProcessor({"pie torch": "PyTorch", "py torch": "PyTorch", "lang graph": "LangGraph"})
    assert tp.vocabulary == ["LangGraph", "PyTorch"]


def test_empty_processor_is_identity():
    tp = TextProcessor()
    assert tp.replace("anything goes") == "anything goes" and tp.snippet("x") is None


def test_replacement_with_regex_chars_is_literal():
    tp = TextProcessor({"c plus plus": "C++", "dot net": ".NET"})
    assert tp.replace("I write c plus plus and dot net") == "I write C++ and .NET"


def test_replacement_output_with_backslash_is_literal():
    tp = TextProcessor({"backslash n": r"\n", "temp dir": r"C:\temp\1", "digits": r"\d+"})
    assert tp.replace("type backslash n here") == r"type \n here"
    assert tp.replace("go to temp dir") == r"go to C:\temp\1"
    assert tp.replace("regex digits") == r"regex \d+"  # must not raise re.error


def test_word_count():
    from vaani.textproc import word_count

    assert word_count("") == 0 and word_count("  a  b\nc ") == 3
