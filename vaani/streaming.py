"""Live dictation: while the key is held, transcribe and paste each phrase as soon as you pause.

A `StreamSession` runs on its own thread (never the keyboard hook or the PortAudio callback). Every
POLL_S it looks at the audio recorded so far; when the speaker has been quiet for `pause_ms` after
speaking, that audio goes through `Pipeline.process(..., partial=True)` and the text is handed to `emit`.
Phrases are raw ASR text (replacements only): cleanup is too slow to run between phrases, and snippets
need the whole utterance. Whatever is left when the key is released is handled by the app as the last
phrase, so a short dictation with no pause still gets the full pipeline.

Pause detection is a plain level check, not a model, so polling is cheap; the VAD inside the pipeline
still decides what counts as speech. If background noise hides every pause, nothing streams and the
whole dictation is processed at release, as before.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

POLL_S = 0.15
HOP_MS = 30
MIN_VOICED_MS = 240  # speech needed before a pause can end a phrase
LEVEL_FLOOR = 5e-4  # RMS below this is silence however quiet the room is (about -66 dBFS)
SPEECH_FRACTION = 0.15  # a frame is speech if its RMS exceeds this fraction of the loud frames' RMS


def trailing_pause_ms(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    """(voiced_ms, trailing_silence_ms) of the audio, from 30 ms RMS frames."""
    hop = int(sample_rate * HOP_MS / 1000)
    n = audio.size // hop
    if n == 0:
        return 0.0, 0.0
    frames = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    voiced = rms > max(SPEECH_FRACTION * float(np.percentile(rms, 95)), LEVEL_FLOOR)
    trailing = 0
    for v in voiced[::-1]:
        if v:
            break
        trailing += 1
    return float(voiced.sum() * HOP_MS), float(trailing * HOP_MS)


class StreamSession:
    """Owns the polling thread for one dictation. `finish()` stops it and says whether it pasted anything."""

    def __init__(
        self,
        recorder: Any,
        pipeline: Any,
        emit: Callable[[str], None],
        pause_ms: int = 700,
        sample_rate: int | None = None,
    ):
        self.recorder = recorder
        self.pipeline = pipeline
        self.emit = emit
        self.pause_ms = pause_ms
        self.sample_rate = sample_rate or recorder.sample_rate
        self.phrases = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="stream", daemon=True)

    def start(self) -> StreamSession:
        self._thread.start()
        return self

    def finish(self) -> bool:
        """Stop polling (waits for a phrase that is mid-transcription). True if any phrase was emitted."""
        self._stop.set()
        self._thread.join(timeout=10)
        return self.phrases > 0

    def _run(self) -> None:
        while not self._stop.wait(POLL_S):
            try:
                self.step()
            except Exception:
                log.exception("streaming step failed")  # the audio stays pending: release handles it

    def step(self) -> bool:
        """One poll. Returns True if a phrase was transcribed and emitted."""
        audio = self.recorder.peek()
        voiced_ms, silence_ms = trailing_pause_ms(audio, self.sample_rate)
        if voiced_ms < MIN_VOICED_MS or silence_ms < self.pause_ms:
            return False
        result = self.pipeline.process(audio, self.sample_rate, partial=True)
        self.recorder.consume(audio.size)  # whether or not text came out: silence must not pile up
        if not result.text:
            return False
        self.phrases += 1
        self.emit(result.text)
        log.info("phrase %d: audio %.1fs | asr %.0f ms", self.phrases, result.audio_s, result.asr_ms)
        return True
