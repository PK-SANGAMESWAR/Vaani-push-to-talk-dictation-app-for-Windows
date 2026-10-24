"""Eval-only Whisper ASR backend (faster-whisper / CTranslate2), for baseline comparisons. Not shipped.

It plugs into `vaani.pipeline.Pipeline` in place of the Parakeet `Transcriber` (same `transcribe`,
`warmup`, `device`, `quantization`), so snippets, replacements and cleanup run exactly as in the app.
Speech detection uses faster-whisper's own Silero VAD with Vaani's VAD settings: Whisper hallucinates on
silence without one, and the comparison should be Whisper as a user would deploy it.

Install separately (it would otherwise pull the CPU `onnxruntime`, which breaks `onnxruntime-gpu`):
    .\\.venv\\Scripts\\python -m pip install --no-deps -r requirements-whisper.txt
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODELS = {"whisper-large-v3-turbo": "large-v3-turbo"}  # Vaani name -> faster-whisper model id
BEAM_SIZE = 1  # greedy: dictation is latency-bound; recorded in provenance


def _add_cuda12_dlls() -> None:
    """CTranslate2 4.x links CUDA 12 (cublas64_12.dll); onnxruntime-gpu brings CUDA 13. Make the pip
    `nvidia-cublas-cu12` DLLs findable (Windows only; elsewhere the loader path is used as is)."""
    if sys.platform != "win32":
        return
    import site

    for base in site.getsitepackages():
        for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
            d = Path(base) / sub
            if d.is_dir():
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "whisper-large-v3-turbo",
        device: str = "auto",
        quantization: str = "auto",
        vad_threshold: float = 0.5,
        vad_min_silence_ms: int = 600,
        vad_speech_pad_ms: int = 200,
    ):
        _add_cuda12_dlls()
        import ctranslate2
        from faster_whisper import WhisperModel

        use_cuda = device == "cuda" or (device == "auto" and ctranslate2.get_cuda_device_count() > 0)
        self.device = "cuda" if use_cuda else "cpu"
        if quantization in ("auto", "none"):
            # fp16 is CTranslate2's native GPU type; "none" means unquantized weights on either device
            self.quantization = ("float16" if use_cuda else "int8") if quantization == "auto" else "float32"
        else:
            self.quantization = "int8_float16" if use_cuda else "int8"
        t0 = time.perf_counter()
        self.model = WhisperModel(MODELS[model_name], device=self.device, compute_type=self.quantization)
        self.vad_parameters = {
            "threshold": vad_threshold,
            "min_silence_duration_ms": vad_min_silence_ms,
            "speech_pad_ms": vad_speech_pad_ms,
        }
        self.decode = {"language": "en", "beam_size": BEAM_SIZE, "condition_on_previous_text": False}
        log.info("ASR %s loaded on %s in %.1fs", model_name, self.device, time.perf_counter() - t0)

    def warmup(self, sample_rate: int = 16_000) -> None:
        t = np.arange(sample_rate) / sample_rate
        tone = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        list(self.model.transcribe(tone, vad_filter=False, **self.decode)[0])

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000, vad: bool = True) -> str:
        if audio.size == 0:
            return ""
        if sample_rate != 16_000:
            raise ValueError("Whisper backend expects 16 kHz audio")
        segments, _info = self.model.transcribe(
            audio, vad_filter=vad, vad_parameters=self.vad_parameters if vad else None, **self.decode
        )
        return " ".join(t for s in segments if (t := s.text.strip()))
