"""Streaming dictation: pause detection, the session loop, recorder peek/consume, partial pipeline."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from vaani import streaming
from vaani.config import Config, load_config
from vaani.pipeline import Pipeline
from vaani.streaming import StreamSession, trailing_pause_ms

SR = 16_000


def tone(seconds: float, level: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_trailing_pause_measures_voiced_and_silence():
    voiced, quiet = trailing_pause_ms(np.concatenate([tone(1.0), silence(0.9)]), SR)
    assert voiced == pytest.approx(1000, abs=60) and quiet == pytest.approx(900, abs=60)
    assert trailing_pause_ms(tone(1.0), SR)[1] == 0  # still speaking
    assert trailing_pause_ms(silence(2.0), SR) == (0.0, pytest.approx(2000, abs=30))
    assert trailing_pause_ms(np.zeros(0, dtype=np.float32), SR) == (0.0, 0.0)


def test_quiet_speech_still_counts_as_voiced():
    voiced, quiet = trailing_pause_ms(np.concatenate([tone(1.0, 0.004), silence(0.8)]), SR)
    assert voiced > 900 and quiet == pytest.approx(800, abs=60)


class FakeRecorder:
    sample_rate = SR

    def __init__(self, audio):
        self.audio = audio

    def peek(self):
        return self.audio

    def consume(self, n):
        self.audio = self.audio[n:]


class FakePipeline:
    def __init__(self, text="hello there"):
        self.text, self.calls = text, []

    def process(self, audio, sr, partial=False):
        self.calls.append((audio.size, partial))
        return SimpleNamespace(text=self.text, audio_s=audio.size / sr, asr_ms=1.0)


def session(audio, text="hello there", pause_ms=700):
    out = []
    rec, pipe = FakeRecorder(audio), FakePipeline(text)
    return StreamSession(rec, pipe, out.append, pause_ms), rec, pipe, out


def test_step_emits_a_phrase_after_a_long_enough_pause_and_consumes_it():
    s, rec, pipe, out = session(np.concatenate([tone(1.0), silence(0.8)]))
    assert s.step() is True
    assert out == ["hello there"] and pipe.calls[0][1] is True  # partial phrase
    assert rec.audio.size == 0 and s.phrases == 1


def test_step_waits_while_speaking_or_pause_is_short_or_no_speech_yet():
    for audio in (tone(2.0), np.concatenate([tone(1.0), silence(0.3)]), silence(3.0), tone(0.1)):
        s, rec, pipe, out = session(audio)
        assert s.step() is False and not out and not pipe.calls and rec.audio.size == audio.size


def test_phrase_without_text_is_consumed_but_not_emitted():
    s, rec, _pipe, out = session(np.concatenate([tone(0.5), silence(0.9)]), text="")
    assert s.step() is False and not out and rec.audio.size == 0 and s.phrases == 0


def test_thread_runs_and_finish_reports_whether_anything_streamed(monkeypatch):
    monkeypatch.setattr(streaming, "POLL_S", 0.01)
    s, _rec, _pipe, out = session(np.concatenate([tone(1.0), silence(0.8)]))
    s.start()
    deadline = threading.Event()
    for _ in range(200):
        if out:
            break
        deadline.wait(0.01)
    assert s.finish() is True and out == ["hello there"]
    quiet_session, *_ = session(tone(1.0))
    quiet_session.start()
    assert quiet_session.finish() is False


def test_step_errors_do_not_kill_the_thread(monkeypatch):
    monkeypatch.setattr(streaming, "POLL_S", 0.01)
    s, _rec, pipe, _out = session(np.concatenate([tone(1.0), silence(0.8)]))
    pipe.process = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    s.start()
    threading.Event().wait(0.1)
    assert s._thread.is_alive()
    s.finish()


def test_recorder_peek_and_consume():
    from vaani.audio import Recorder

    r = object.__new__(Recorder)  # no microphone in tests
    r._lock = threading.Lock()
    r._frames = [np.arange(5, dtype=np.float32), np.arange(5, 10, dtype=np.float32)]
    assert r.peek().tolist() == list(range(10))
    r.consume(7)
    assert r.peek().tolist() == [7.0, 8.0, 9.0]
    r.consume(99)
    assert r.peek().size == 0 and r._frames == []


def test_partial_pipeline_skips_snippets_and_cleanup():
    class Stt:
        def transcribe(self, audio, sr, vad=True):
            return "insert my email"

    class Cleaner:
        def clean_with_status(self, text):
            raise AssertionError("cleanup must not run on a streamed phrase")

    cfg = Config(snippets={"insert my email": "me@example.com"}, replacements={"email": "e-mail"})
    pipe = Pipeline(cfg, Stt(), cleaner=Cleaner())
    part = pipe.process(np.ones(16_000, dtype=np.float32), SR, partial=True)
    assert part.route == "partial" and part.text == "insert my e-mail" and part.cleanup_status is None
    assert pipe.process(np.ones(16_000, dtype=np.float32), SR).route == "snippet"


def test_streaming_config_defaults_and_toml(tmp_path):
    assert Config().streaming is True and Config().stream_pause_ms == 700
    p = tmp_path / "c.toml"
    p.write_text("[streaming]\nstreaming = false\nstream_pause_ms = 500\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.streaming is False and cfg.stream_pause_ms == 500
