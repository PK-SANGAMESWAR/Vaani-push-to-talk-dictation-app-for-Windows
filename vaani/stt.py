"""Speech-to-text: NVIDIA Parakeet TDT 0.6B v3 (ONNX) + Silero VAD, via onnx-asr.

Models are loaded once and stay resident. GPU (CUDA) is used when available,
falling back to CPU with int8 weights.
"""

from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)

CUDA = "CUDAExecutionProvider"
CPU = "CPUExecutionProvider"


def _cuda_available() -> bool:
    import onnxruntime as ort

    if CUDA not in ort.get_available_providers():
        return False
    # onnxruntime-gpu >= 1.21: load CUDA/cuDNN DLLs from the nvidia-* pip packages.
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as e:  # pragma: no cover - environment specific
            log.warning("preload_dlls failed (%s); relying on system CUDA install", e)
    return True


class Transcriber:
    def __init__(
        self,
        model_name: str = "nemo-parakeet-tdt-0.6b-v3",
        device: str = "auto",
        quantization: str = "auto",
        vad_threshold: float = 0.5,
        vad_min_silence_ms: int = 600,
        vad_speech_pad_ms: int = 200,
    ):
        import onnx_asr

        use_cuda = device == "cuda" or (device == "auto" and _cuda_available())
        if device == "cuda" and not _cuda_available():
            raise RuntimeError("device = 'cuda' but onnxruntime-gpu has no CUDAExecutionProvider")

        t0 = time.perf_counter()
        self.model = None
        if use_cuda:
            try:
                # int8 ops mostly fall back to CPU under the CUDA EP, so fp32 is faster on GPU.
                quant = None if quantization in ("auto", "none") else quantization
                self.model = onnx_asr.load_model(model_name, quantization=quant, providers=[CUDA, CPU])
                self.device = "cuda"
            except Exception:
                log.exception("CUDA load failed, falling back to CPU")
        if self.model is None:
            quant = "int8" if quantization == "auto" else (None if quantization == "none" else quantization)
            self.model = onnx_asr.load_model(model_name, quantization=quant, providers=[CPU])
            self.device = "cpu"

        # Silero is ~2 MB; CPU is faster than a GPU round-trip for it.
        vad = onnx_asr.load_vad("silero", providers=[CPU])
        self.vad_model = self.model.with_vad(
            vad,
            threshold=vad_threshold,
            min_silence_duration_ms=vad_min_silence_ms,
            speech_pad_ms=vad_speech_pad_ms,
            max_speech_duration_s=20,  # Parakeet's sweet spot; longer speech is split
        )
        log.info("ASR %s loaded on %s in %.1fs", model_name, self.device, time.perf_counter() - t0)

    def warmup(self, sample_rate: int = 16_000) -> None:
        """Run one inference so CUDA kernels/graphs are initialised before the first real dictation."""
        t0 = time.perf_counter()
        noise = (np.random.default_rng(0).standard_normal(sample_rate) * 1e-3).astype(np.float32)
        self.model.recognize(noise, sample_rate=sample_rate)
        log.info("ASR warm-up took %.0f ms", (time.perf_counter() - t0) * 1000)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16_000) -> str:
        """VAD-segment the clip (drops silence, splits long dictation) and join segment texts.

        Returns "" when no speech is detected, so silence never produces text.
        """
        if audio.size == 0:
            return ""
        segments = self.vad_model.recognize(audio, sample_rate=sample_rate)
        return " ".join(t for s in segments if (t := s.text.strip()))
