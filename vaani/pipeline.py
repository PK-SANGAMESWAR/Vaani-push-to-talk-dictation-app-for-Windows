"""Audio in, text out: VAD + ASR -> snippet / replacements -> optional LLM cleanup.

The app and the eval suite both run this, so evals measure exactly the code that ships.
It never pastes, logs transcripts, or touches the clipboard; that is App's job.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import Config
from .textproc import TextProcessor, word_count

if TYPE_CHECKING:
    from .cleanup import Cleaner
    from .stt import Transcriber


@dataclass
class Result:
    raw: str  # ASR text; "" = no speech detected
    text: str  # final text, before the app adds a trailing space
    route: str  # "empty" | "snippet" | "replace" | "cleanup"
    cleanup_status: str | None  # None = not run | "ok" | "error" | "rejected"
    audio_s: float
    asr_ms: float
    cleanup_ms: float
    total_ms: float


def make_transcriber(cfg: Config) -> Transcriber:
    from .stt import Transcriber

    return Transcriber(
        cfg.asr_model,
        cfg.device,
        cfg.quantization,
        cfg.vad_threshold,
        cfg.vad_min_silence_ms,
        cfg.vad_speech_pad_ms,
    )


def make_cleaner(cfg: Config, vocabulary: list[str]) -> Cleaner:
    from .cleanup import Cleaner

    return Cleaner(
        cfg.llm_url,
        cfg.llm_model,
        cfg.llm_timeout_s,
        cfg.llm_keep_alive,
        cfg.llm_think_flag,
        vocabulary,
    )


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        stt: Any = None,
        text: TextProcessor | None = None,
        cleaner: Any = None,
        use_vad: bool = True,
    ):
        self.cfg = cfg  # read live: the tray toggles cfg.llm_enabled at runtime
        self.stt = stt
        self.text = text or TextProcessor(cfg.replacements, cfg.snippets)
        self.cleaner = cleaner
        self.use_vad = use_vad  # False is an eval-only diagnostic; the app always uses VAD

    @classmethod
    def load(cls, cfg: Config, use_vad: bool = True) -> Pipeline:
        """Load models (slow, once). Caller should run `stt.warmup()` before timing anything."""
        text = TextProcessor(cfg.replacements, cfg.snippets)
        return cls(cfg, make_transcriber(cfg), text, make_cleaner(cfg, text.vocabulary), use_vad)

    def process(self, audio: np.ndarray, sample_rate: int) -> Result:
        c = self.cfg
        audio_s = audio.size / sample_rate
        t0 = time.perf_counter()
        if self.use_vad:
            raw = self.stt.transcribe(audio, sample_rate)
        else:
            raw = self.stt.transcribe(audio, sample_rate, vad=False)
        t_asr = time.perf_counter() - t0
        if not raw:
            return Result("", "", "empty", None, audio_s, t_asr * 1000, 0.0, t_asr * 1000)

        cleanup_ms = 0.0
        status = None
        snippet = self.text.snippet(raw)
        if snippet is not None:
            final, route = snippet, "snippet"
        else:
            final, route = self.text.replace(raw), "replace"
            if c.llm_enabled and word_count(final) >= c.llm_min_words:
                t1 = time.perf_counter()
                cleaned, status = self._clean(final)
                cleanup_ms = (time.perf_counter() - t1) * 1000
                final, route = self.text.replace(cleaned), "cleanup"
        t_total = time.perf_counter() - t0
        return Result(raw, final, route, status, audio_s, t_asr * 1000, cleanup_ms, t_total * 1000)

    def _clean(self, text: str) -> tuple[str, str | None]:
        with_status = getattr(self.cleaner, "clean_with_status", None)
        if with_status is not None:
            return with_status(text)
        return self.cleaner.clean(text), None  # cleaners without status (test fakes)
