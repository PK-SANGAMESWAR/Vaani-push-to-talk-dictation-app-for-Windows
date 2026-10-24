"""Recorder tests with a fake sounddevice (no microphone needed)."""

from __future__ import annotations

import numpy as np
import pytest

from vaani import audio


class FakeStream:
    instances: list[FakeStream] = []

    def __init__(self, **kw):
        self.kw = kw
        self.callback = kw["callback"]
        self.started = self.stopped = self.closed = False
        FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def feed(self, value: float, n: int = 480):
        self.callback(np.full((n, 1), value, dtype=np.float32), n, None, None)


class FakeSd:
    InputStream = FakeStream

    def __init__(self, supported: bool = True):
        self.supported = supported

    def check_input_settings(self, **kw):
        if not self.supported:
            raise ValueError("unsupported rate")

    def query_devices(self, device, kind):
        return {"default_samplerate": 44_100.0}


@pytest.fixture(autouse=True)
def fake_sd(monkeypatch):
    FakeStream.instances = []
    monkeypatch.setattr(audio, "sd", FakeSd())


def test_uses_requested_rate_when_supported():
    assert audio.Recorder(16_000).sample_rate == 16_000


def test_falls_back_to_device_rate(monkeypatch):
    monkeypatch.setattr(audio, "sd", FakeSd(supported=False))
    assert audio.Recorder(16_000).sample_rate == 44_100


def test_start_stop_opens_and_closes_stream_and_returns_audio():
    r = audio.Recorder(16_000)
    assert not FakeStream.instances  # lazy: mic is not opened until needed
    r.start()
    stream = FakeStream.instances[0]
    assert stream.started and stream.kw["samplerate"] == 16_000 and stream.kw["blocksize"] == 480
    stream.feed(0.25)
    stream.feed(0.5)
    out = r.stop()
    assert stream.stopped and stream.closed
    assert out.dtype == np.float32 and out.size == 960
    assert out[0] == 0.25 and out[-1] == 0.5


def test_stop_without_audio_returns_empty_float32():
    r = audio.Recorder(16_000)
    r.start()
    out = r.stop()
    assert out.size == 0 and out.dtype == np.float32


def test_frames_outside_recording_are_dropped_when_not_keep_open():
    r = audio.Recorder(16_000)
    r.start()
    stream = FakeStream.instances[0]
    r.stop()
    stream.feed(1.0)  # late callback after stop
    r.start()
    assert r.stop().size == 0


def test_keep_open_preroll_is_prepended_and_bounded():
    r = audio.Recorder(16_000, keep_open=True, preroll_ms=90)  # 3 blocks of 30 ms
    assert len(FakeStream.instances) == 1 and FakeStream.instances[0].started
    stream = FakeStream.instances[0]
    for v in (0.1, 0.2, 0.3, 0.4):  # only the last 3 survive
        stream.feed(v)
    r.start()
    stream.feed(0.9)
    out = r.stop()
    assert not stream.stopped  # stays open between dictations
    assert [round(float(out[i * 480]), 1) for i in range(4)] == [0.2, 0.3, 0.4, 0.9]
    # pre-roll was consumed: a second dictation must not replay old audio
    r.start()
    assert r.stop().size == 0


def test_close_releases_stream():
    r = audio.Recorder(16_000, keep_open=True)
    stream = FakeStream.instances[0]
    r.close()
    assert stream.stopped and stream.closed
    r.close()  # idempotent


def test_partial_mic_name_resolves_to_one_device_on_the_default_host_api(monkeypatch):
    from vaani import audio

    devices = [
        {"name": "Microphone Array (Realtek(R) Au", "max_input_channels": 2, "hostapi": 0},
        {"name": "Speakers (Realtek(R) Audio)", "max_input_channels": 0, "hostapi": 0},
        {"name": "Microphone Array (Realtek(R) Audio)", "max_input_channels": 2, "hostapi": 1},
        {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 2, "hostapi": 2},
    ]

    class FakeSd:
        default = type("D", (), {"hostapi": 1})

        @staticmethod
        def query_devices(i=None):
            return devices if i is None else devices[i]

    monkeypatch.setattr(audio, "sd", FakeSd)
    assert audio.resolve_input_device("microphone array") == 2  # default host API wins
    assert audio.resolve_input_device("CABLE Output") == 3  # only match, other host API
    assert audio.resolve_input_device(5) == 5 and audio.resolve_input_device(None) is None
    with pytest.raises(ValueError, match="no input device"):
        audio.resolve_input_device("Speakers")  # output-only devices never match
