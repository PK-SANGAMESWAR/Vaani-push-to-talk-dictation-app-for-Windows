"""Pipeline tests: routing, cleanup status, timings (models faked)."""

from __future__ import annotations

import numpy as np
import pytest

from vaani.config import Config
from vaani.pipeline import Pipeline

SR = 16_000
AUDIO = np.ones(SR, np.float32)
LONG = "so um we should ship the pie torch model on friday okay"  # 12 words


class FakeStt:
    def __init__(self, text: str):
        self.text, self.calls = text, []

    def transcribe(self, audio, sr, vad=True):
        self.calls.append(vad)
        return self.text


class StatusCleaner:
    def __init__(self, out: str, status: str = "ok"):
        self.out, self.status, self.calls = out, status, []

    def clean_with_status(self, text):
        self.calls.append(text)
        return self.out, self.status


def make(text: str, cleaner=None, use_vad: bool = True, **cfg) -> Pipeline:
    cfg.setdefault("llm_enabled", False)
    return Pipeline(Config(**cfg), FakeStt(text), cleaner=cleaner or StatusCleaner(""), use_vad=use_vad)


def test_silence_is_empty_route_with_asr_timing():
    r = make("").process(np.zeros(SR, np.float32), SR)
    assert (r.raw, r.text, r.route, r.cleanup_status) == ("", "", "empty", None)
    assert r.audio_s == 1.0 and r.asr_ms >= 0 and r.total_ms == r.asr_ms and r.cleanup_ms == 0


def test_replacements_route():
    r = make("I love pie torch", replacements={"pie torch": "PyTorch"}).process(AUDIO, SR)
    assert (r.raw, r.text, r.route) == ("I love pie torch", "I love PyTorch", "replace")
    assert r.total_ms >= r.asr_ms


def test_snippet_route_skips_cleanup():
    p = make("Insert my signature.", llm_enabled=True, snippets={"insert my signature": "Best,\nS"})
    r = p.process(AUDIO, SR)
    assert (r.text, r.route) == ("Best,\nS", "snippet") and p.cleaner.calls == []


def test_cleanup_route_applies_replacements_before_and_after():
    cleaner = StatusCleaner("We should ship the pie torch model on Friday.")
    p = make(LONG, cleaner, llm_enabled=True, replacements={"pie torch": "PyTorch"})
    r = p.process(AUDIO, SR)
    assert cleaner.calls == ["so um we should ship the PyTorch model on friday okay"]
    assert r.text == "We should ship the PyTorch model on Friday."
    assert (r.route, r.cleanup_status) == ("cleanup", "ok") and r.cleanup_ms >= 0


@pytest.mark.parametrize("status", ["error", "rejected"])
def test_cleanup_fallback_status_is_reported(status):
    r = make(LONG, StatusCleaner(LONG, status), llm_enabled=True).process(AUDIO, SR)
    assert r.text == LONG and r.cleanup_status == status


def test_short_utterance_skips_cleanup():
    p = make("send it now", llm_enabled=True)
    r = p.process(AUDIO, SR)
    assert r.route == "replace" and r.cleanup_status is None and p.cleaner.calls == []


def test_llm_toggle_is_read_live():
    p = make(LONG, StatusCleaner("Cleaned text here and more words."), llm_enabled=True)
    p.cfg.llm_enabled = False  # tray toggle
    assert p.process(AUDIO, SR).route == "replace"
    p.cfg.llm_enabled = True
    assert p.process(AUDIO, SR).route == "cleanup"


def test_cleaner_without_status_is_supported():
    class Plain:
        def clean(self, text):
            return "Plain cleaned output with enough words."

    r = make(LONG, Plain(), llm_enabled=True).process(AUDIO, SR)
    assert r.route == "cleanup" and r.cleanup_status is None


def test_vad_flag_is_passed_through():
    p = make("hi", use_vad=False)
    p.process(AUDIO, SR)
    assert p.stt.calls == [False]
    p.use_vad = True
    p.process(AUDIO, SR)
    assert p.stt.calls == [False, True]


def test_cleaner_clean_with_status(monkeypatch):
    from vaani.cleanup import Cleaner

    c = Cleaner()
    monkeypatch.setattr(c, "_post", lambda *a, **k: {"message": {"content": "Hello there, how are you?"}})
    assert c.clean_with_status("hello there how are you") == ("Hello there, how are you?", "ok")
    monkeypatch.setattr(c, "_post", lambda *a, **k: {"message": {"content": "word " * 50}})
    assert c.clean_with_status("one two three four five") == ("one two three four five", "rejected")

    def down(*a, **k):
        raise OSError("refused")

    monkeypatch.setattr(c, "_post", down)
    assert c.clean_with_status("hello there") == ("hello there", "error")
    assert c.clean("hello there") == "hello there"
