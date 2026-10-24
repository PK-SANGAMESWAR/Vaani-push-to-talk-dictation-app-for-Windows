"""Transcriber tests with fake onnx_asr / onnxruntime modules (no GPU or model files needed)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from vaani import stt


class FakeModel:
    def __init__(self, providers, quant):
        self.providers, self.quant = providers, quant
        self.vad_kwargs: dict = {}
        self.segments: list[str] = []
        self.recognized: list[tuple] = []

    def with_vad(self, vad, **kw):
        self.vad_kwargs = kw
        return self

    def recognize(self, audio, sample_rate):
        self.recognized.append((audio.shape, sample_rate))
        return (SimpleNamespace(text=t, start=0, end=1) for t in self.segments)


@pytest.fixture
def fakes(monkeypatch):
    state = SimpleNamespace(loads=[], cuda=True, cuda_load_fails=False, preloaded=False)

    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: [stt.CUDA, stt.CPU] if state.cuda else [stt.CPU]

    def preload():
        state.preloaded = True

    ort.preload_dlls = preload

    asr = types.ModuleType("onnx_asr")

    def load_model(name, quantization=None, providers=None):
        state.loads.append((name, quantization, providers))
        if state.cuda_load_fails and any(p == stt.CUDA or p[0] == stt.CUDA for p in providers):
            raise RuntimeError("cuda boom")
        return FakeModel(providers, quantization)

    asr.load_model = load_model
    asr.load_vad = lambda name, providers=None: object()
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "onnx_asr", asr)
    return state


def test_auto_uses_cuda_fp32_when_available(fakes):
    t = stt.Transcriber()
    assert t.device == "cuda" and fakes.preloaded
    assert fakes.loads == [("nemo-parakeet-tdt-0.6b-v3", None, [(stt.CUDA, stt.CUDA_OPTIONS), stt.CPU])]


def test_auto_uses_cpu_int8_without_cuda(fakes):
    fakes.cuda = False
    t = stt.Transcriber()
    assert t.device == "cpu"
    assert fakes.loads == [("nemo-parakeet-tdt-0.6b-v3", "int8", [stt.CPU])]


def test_cuda_load_failure_falls_back_to_cpu_int8(fakes):
    fakes.cuda_load_fails = True
    t = stt.Transcriber()
    assert t.device == "cpu"
    assert fakes.loads[-1][1:] == ("int8", [stt.CPU])


def test_explicit_cuda_without_provider_raises(fakes):
    fakes.cuda = False
    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        stt.Transcriber(device="cuda")


def test_explicit_cpu_never_touches_cuda(fakes):
    t = stt.Transcriber(device="cpu", quantization="none")
    assert t.device == "cpu" and not fakes.preloaded
    assert fakes.loads[0][1:] == (None, [stt.CPU])


def test_vad_is_configured(fakes):
    t = stt.Transcriber(vad_threshold=0.6, vad_min_silence_ms=500, vad_speech_pad_ms=100)
    assert t.vad_model.vad_kwargs == {
        "threshold": 0.6,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 100,
        "max_speech_duration_s": 20,
    }


def test_transcribe_joins_segments_and_drops_blanks(fakes):
    t = stt.Transcriber()
    t.vad_model.segments = [" hello there ", "  ", "general kenobi"]
    assert t.transcribe(np.ones(16_000, dtype=np.float32)) == "hello there general kenobi"


def test_silence_yields_no_text(fakes):
    t = stt.Transcriber()
    t.vad_model.segments = []  # VAD found no speech
    assert t.transcribe(np.zeros(16_000, dtype=np.float32)) == ""


def test_empty_audio_skips_the_model(fakes):
    t = stt.Transcriber()
    assert t.transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert not t.vad_model.recognized


def test_warmup_runs_one_inference(fakes):
    t = stt.Transcriber()
    t.warmup(16_000)
    assert t.model.recognized == [((16_000,), 16_000)]
