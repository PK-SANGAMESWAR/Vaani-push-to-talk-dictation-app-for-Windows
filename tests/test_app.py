"""App orchestration tests: state machine, queues, processing pipeline (all hardware faked)."""

from __future__ import annotations

import queue
import sys
import threading
import types

import numpy as np
import pytest

from vaani.app import App
from vaani.config import Config

SR = 16_000


class FakeRecorder:
    sample_rate = SR

    def __init__(self, seconds: float = 1.0, fail_start: bool = False):
        self.seconds, self.fail_start = seconds, fail_start
        self.started = self.stopped = self.closed = 0

    def start(self):
        if self.fail_start:
            raise OSError("no mic")
        self.started += 1

    def stop(self):
        self.stopped += 1
        return np.ones(int(self.seconds * SR), dtype=np.float32)

    def close(self):
        self.closed += 1


class FakeStt:
    device = "cuda"

    def __init__(self, text: str = "hello world"):
        self.text, self.calls = text, 0

    def transcribe(self, audio, sr):
        self.calls += 1
        return self.text


class FakeCleaner:
    def __init__(self, result: str | None = None):
        self.result, self.calls = result, []

    def clean(self, text):
        self.calls.append(text)
        return self.result if self.result is not None else text


@pytest.fixture
def inserted(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr("vaani.output.insert_text", lambda *a: calls.append(a))
    return calls


def make_app(tmp_path, recorder=None, stt=None, cleaner=None, **cfg) -> App:
    cfg.setdefault("sounds", False)
    cfg.setdefault("llm_enabled", False)
    cfg.setdefault("streaming", False)  # streaming has its own tests below
    cfg.setdefault("release_tail_ms", 0)
    app = App(Config(**cfg), tmp_path)
    app.recorder = recorder or FakeRecorder()
    app.stt = stt or FakeStt()
    app.cleaner = cleaner or FakeCleaner()
    return app


# ---------- recording state machine ----------


def test_start_is_idempotent_against_key_autorepeat(tmp_path):
    app = make_app(tmp_path)
    for _ in range(5):
        app._start()
    assert app.recorder.started == 1 and app.state == "recording"
    app._timer.cancel()


def test_start_before_models_loaded_is_ignored(tmp_path):
    app = make_app(tmp_path)
    app.recorder = None
    app._start()
    assert not app._recording and app.state == "loading"


def test_mic_failure_resets_so_next_press_can_retry(tmp_path):
    app = make_app(tmp_path, recorder=FakeRecorder(fail_start=True))
    app._start()
    assert not app._recording and app.state == "loading"
    app.recorder.fail_start = False
    app._start()
    assert app._recording and app.state == "recording"
    app._timer.cancel()


def test_stop_without_start_is_a_noop(tmp_path):
    app = make_app(tmp_path)
    app._stop()
    assert app.recorder.stopped == 0 and app.jobs.empty()


def test_stop_queues_audio_for_worker(tmp_path):
    app = make_app(tmp_path)
    app._start()
    app._stop()
    assert app.state == "busy" and not app._recording
    assert app.jobs.get_nowait()[0].size == SR


def test_short_tap_is_ignored_and_does_not_wedge_state(tmp_path):
    app = make_app(tmp_path, recorder=FakeRecorder(seconds=0.1))
    for _ in range(3):  # rapid taps
        app._start()
        app._stop()
        assert app.state == "idle" and app.jobs.empty()
    app.recorder.seconds = 1.0
    app._start()
    app._stop()
    assert app.jobs.qsize() == 1  # a real dictation still works afterwards


def test_toggle_mode_alternates(tmp_path):
    app = make_app(tmp_path)
    app._toggle()
    assert app._recording
    app._toggle()
    assert not app._recording and app.jobs.qsize() == 1


def test_max_duration_timer_requests_stop(tmp_path):
    app = make_app(tmp_path, max_duration_s=0.05)
    app._start()
    assert app.ctrl.get(timeout=2) == "stop"
    app._stop()


# ---------- control / worker threads ----------


def test_control_loop_runs_actions_in_order_and_survives_errors(tmp_path):
    app = make_app(tmp_path)
    seen: list[str] = []

    def boom():
        raise RuntimeError("x")

    app._start = lambda: seen.append("start")
    app._stop = lambda: seen.append("stop")
    app._toggle = boom
    for a in ("start", "toggle", "stop", None):
        app.ctrl.put(a)
    t = threading.Thread(target=app._control_loop)
    t.start()
    t.join(2)
    assert not t.is_alive() and seen == ["start", "stop"]


def test_worker_processes_jobs_returns_to_idle_and_survives_errors(tmp_path):
    app = make_app(tmp_path)
    done = threading.Event()
    results: list[int] = []

    def process(audio, streamed=False):
        if audio.size == 1:
            raise RuntimeError("boom")
        results.append(audio.size)
        done.set()

    app._process = process
    app.state = "busy"
    t = threading.Thread(target=app._worker)
    t.start()
    app.jobs.put((np.zeros(1, np.float32), False))  # raises inside _process
    app.jobs.put((np.zeros(5, np.float32), False))
    assert done.wait(2)
    for _ in range(200):  # state flips to idle in the worker's `finally`
        if app.state == "idle":
            break
        threading.Event().wait(0.01)
    app.jobs.put(None)
    t.join(2)
    assert not t.is_alive() and results == [5] and app.state == "idle"


def test_worker_does_not_go_idle_while_user_is_recording_again(tmp_path):
    app = make_app(tmp_path)
    app._process = lambda audio, streamed=False: None
    app.state = "recording"
    app._recording = True
    app.jobs.put((np.zeros(1, np.float32), False))
    app.jobs.put(None)
    app._worker()
    assert app.state == "recording"


# ---------- processing pipeline ----------


def test_silence_pastes_nothing(tmp_path, inserted):
    app = make_app(tmp_path, stt=FakeStt(""))
    app._process(np.zeros(SR, np.float32))
    assert inserted == []


def test_plain_dictation_is_replaced_and_gets_trailing_space(tmp_path, inserted):
    app = make_app(tmp_path, stt=FakeStt("I love pie torch"), replacements={"pie torch": "PyTorch"})
    app._process(np.ones(SR, np.float32))
    assert inserted == [("I love PyTorch ", "clipboard", True, 150)]
    assert app.cleaner.calls == []  # LLM disabled


def test_trailing_space_can_be_disabled_and_not_doubled(tmp_path, inserted):
    app = make_app(tmp_path, stt=FakeStt("hi there"), trailing_space=False)
    app._process(np.ones(SR, np.float32))
    assert inserted[0][0] == "hi there"


def test_snippet_bypasses_replacements_and_llm(tmp_path, inserted):
    app = make_app(
        tmp_path,
        stt=FakeStt("Insert my signature."),
        llm_enabled=True,
        snippets={"insert my signature": "Best,\nSangu"},
        replacements={"best": "WORST"},
    )
    app._process(np.ones(SR, np.float32))
    assert inserted[0][0] == "Best,\nSangu "
    assert app.cleaner.calls == []


LONG = "so um we should ship the pie torch model on friday okay"  # 12 words


def test_llm_runs_for_long_utterances_and_replacements_apply_after(tmp_path, inserted):
    cleaner = FakeCleaner("We should ship the pie torch model on Friday.")
    app = make_app(
        tmp_path, stt=FakeStt(LONG), cleaner=cleaner, llm_enabled=True, replacements={"pie torch": "PyTorch"}
    )
    app._process(np.ones(SR, np.float32))
    assert cleaner.calls == ["so um we should ship the PyTorch model on friday okay"]
    assert inserted[0][0] == "We should ship the PyTorch model on Friday. "


def test_llm_skipped_for_short_utterances(tmp_path, inserted):
    app = make_app(tmp_path, stt=FakeStt("send it now"), llm_enabled=True)
    app._process(np.ones(SR, np.float32))
    assert app.cleaner.calls == [] and inserted[0][0] == "send it now "


def test_llm_disabled_from_tray_toggle_is_respected(tmp_path, inserted):
    app = make_app(tmp_path, stt=FakeStt(LONG), llm_enabled=True)
    app.cfg.llm_enabled = False  # what Tray._toggle_cleanup does
    app._process(np.ones(SR, np.float32))
    assert app.cleaner.calls == [] and inserted[0][0].startswith("so um")


def test_history_is_opt_in(tmp_path, inserted):
    app = make_app(tmp_path, stt=FakeStt("hello world"))
    app._process(np.ones(SR, np.float32))
    assert not (tmp_path / "history.txt").exists()
    app.cfg.save_history = True
    app._process(np.ones(SR, np.float32))
    line = (tmp_path / "history.txt").read_text(encoding="utf-8")
    assert line.rstrip("\n").endswith("\thello world")


def test_transcripts_are_not_logged_at_info(tmp_path, inserted, caplog):
    import logging

    app = make_app(tmp_path, stt=FakeStt("my secret password is hunter2"))
    with caplog.at_level(logging.INFO):
        app._process(np.ones(SR, np.float32))
    assert "hunter2" not in caplog.text
    assert "asr" in caplog.text and "total" in caplog.text  # timing line is present


# ---------- startup / shutdown wiring ----------


class FakeKeyboard(types.ModuleType):
    def __init__(self):
        super().__init__("keyboard")
        self.press, self.release, self.hotkeys, self.unhooked = [], [], [], False

    def on_press_key(self, key, cb):
        self.press.append((key, cb))

    def on_release_key(self, key, cb):
        self.release.append((key, cb))

    def add_hotkey(self, combo, cb):
        self.hotkeys.append((combo, cb))

    def unhook_all(self):
        self.unhooked = True


@pytest.fixture
def wired(monkeypatch):
    kb = FakeKeyboard()
    audio = types.ModuleType("vaani.audio")
    audio.Recorder = lambda *a, **k: FakeRecorder()
    stt = types.ModuleType("vaani.stt")
    stt.Transcriber = lambda *a, **k: types.SimpleNamespace(device="cuda", warmup=lambda sr: None)
    monkeypatch.setitem(sys.modules, "keyboard", kb)
    monkeypatch.setitem(sys.modules, "vaani.audio", audio)
    monkeypatch.setitem(sys.modules, "vaani.stt", stt)
    return kb


def test_init_hold_mode_hooks_only_enqueue(tmp_path, wired):
    app = make_app(tmp_path, mode="hold", hotkey="right ctrl")
    app._control_loop = lambda: None  # keep ctrl queue observable
    app._init()
    assert app.state == "idle" and app.device == "cuda"
    (pk, press), (rk, release) = wired.press[0], wired.release[0]
    assert pk == rk == "right ctrl" and not wired.hotkeys
    press("evt")
    release("evt")
    assert [app.ctrl.get_nowait(), app.ctrl.get_nowait()] == ["start", "stop"]
    assert app.recorder.started == 0  # hook callbacks did no real work


def test_init_toggle_mode_registers_combo(tmp_path, wired):
    app = make_app(tmp_path, mode="toggle", hotkey="ctrl+windows")
    app._control_loop = lambda: None
    app._init()
    combo, cb = wired.hotkeys[0]
    assert combo == "ctrl+windows" and not wired.press
    cb()
    assert app.ctrl.get_nowait() == "toggle"


def test_init_failure_sets_error_state(tmp_path, wired, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("no model")

    sys.modules["vaani.stt"].Transcriber = broken
    app = make_app(tmp_path)
    app._init()
    assert app.state == "error"


def test_shutdown_unhooks_and_stops_threads(tmp_path, wired):
    app = make_app(tmp_path)
    app.shutdown()
    assert wired.unhooked and app.recorder.closed == 1
    assert app.ctrl.get_nowait() is None and app.jobs.get_nowait() is None


def test_queues_are_unbounded_fifo(tmp_path):
    app = make_app(tmp_path)
    assert isinstance(app.jobs, queue.Queue) and app.jobs.maxsize == 0


# ---------- streaming ----------


class StreamingRecorder(FakeRecorder):
    """Recorder with a phrase already spoken: 1 s of speech then a long pause, then a short tail."""

    def __init__(self):
        super().__init__(seconds=0.5)
        t = np.arange(SR) / SR
        self.pending = np.concatenate(
            [(0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), np.zeros(SR, dtype=np.float32)]
        )

    def peek(self):
        return self.pending

    def consume(self, n):
        self.pending = self.pending[n:]


def test_streamed_phrase_is_pasted_during_the_hold_and_tail_is_a_partial_job(tmp_path, inserted):
    from vaani import streaming

    streaming.POLL_S = 0.01
    try:
        app = make_app(tmp_path, recorder=StreamingRecorder(), streaming=True, stream_pause_ms=500)
        app.pipeline.stt = FakeStt("first phrase")
        app._start()
        for _ in range(300):
            if inserted:
                break
            threading.Event().wait(0.01)
        assert [c[0] for c in inserted] == ["first phrase "] and app.state == "recording"
        app._stop()
    finally:
        streaming.POLL_S = 0.15
    audio, streamed = app.jobs.get_nowait()
    assert streamed is True and audio.size == SR // 2  # the recorder's remaining tail


def test_streamed_tail_is_processed_raw_and_silent_tail_logs_nothing(tmp_path, inserted):
    app = make_app(tmp_path, llm_enabled=True, llm_min_words=1, cleaner=FakeCleaner("CLEANED"))
    app._process(np.ones(SR, np.float32), True)
    assert inserted[0][0] == "hello world " and app.cleaner.calls == []  # no cleanup after streaming
    app.stt = FakeStt("")
    app._process(np.ones(SR, np.float32), True)
    assert len(inserted) == 1
