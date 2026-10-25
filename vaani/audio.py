"""Microphone capture. Records mono float32 straight at 16 kHz so no resampling is needed."""

from __future__ import annotations

import logging
import math
import threading
from collections import deque

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

BLOCK_MS = 30


def resolve_input_device(device: str | int | None) -> str | int | None:
    """Turn a partial mic name into a device index.

    Windows lists every mic once per host API (MME, DirectSound, WASAPI, WDM-KS), so passing a name to
    sounddevice raises "Multiple input devices found". Pick the match on the default host API (as the
    Windows default mic would be), else the first match. Indexes and None pass through unchanged.
    """
    if not isinstance(device, str):
        return device
    wanted = device.lower()
    matches = [
        i
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0 and wanted in d["name"].lower()
    ]
    if not matches:
        raise ValueError(f"no input device matches {device!r}")
    default_api = sd.default.hostapi
    return next((i for i in matches if sd.query_devices(i)["hostapi"] == default_api), matches[0])


class Recorder:
    def __init__(
        self,
        sample_rate: int = 16_000,
        device: str | int | None = None,
        keep_open: bool = False,
        preroll_ms: int = 300,
    ):
        self.device = resolve_input_device(device)
        self.keep_open = keep_open
        self.sample_rate = self._supported_rate(sample_rate)
        self._lock = threading.Lock()
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, math.ceil(preroll_ms / BLOCK_MS)))
        self._stream: sd.InputStream | None = None
        if keep_open:
            self._open()

    def _supported_rate(self, wanted: int) -> int:
        """Use 16 kHz if the device accepts it, otherwise the device default (onnx-asr resamples)."""
        try:
            sd.check_input_settings(device=self.device, samplerate=wanted, channels=1, dtype="float32")
            return wanted
        except Exception:
            rate = int(sd.query_devices(self.device, "input")["default_samplerate"])
            log.warning("Mic does not support %d Hz, recording at %d Hz instead", wanted, rate)
            return rate

    def _open(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=int(self.sample_rate * BLOCK_MS / 1000),
            callback=self._callback,
        )
        self._stream.start()

    def _close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata, frames, time_info, status) -> None:  # runs on PortAudio thread
        if status:
            log.debug("audio status: %s", status)
        chunk = indata[:, 0].copy()
        with self._lock:
            if self._recording:
                self._frames.append(chunk)
            elif self.keep_open:
                self._preroll.append(chunk)

    def start(self) -> None:
        with self._lock:
            self._frames = list(self._preroll) if self.keep_open else []
            self._preroll.clear()
            self._recording = True
        if not self.keep_open:
            self._open()

    def stop(self) -> np.ndarray:
        if not self.keep_open:
            self._close()
        with self._lock:
            self._recording = False
            frames, self._frames = self._frames, []
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).astype(np.float32, copy=False)

    def peek(self) -> np.ndarray:
        """Audio recorded since start (or the last `consume`), without removing it."""
        with self._lock:
            frames = list(self._frames)
        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).astype(np.float32, copy=False)

    def consume(self, n_samples: int) -> None:
        """Drop the first n_samples of the pending audio (a phrase that was already transcribed)."""
        with self._lock:
            pending = np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.float32)
            self._frames = [pending[n_samples:]] if pending.size > n_samples else []

    @property
    def recording(self) -> bool:
        return self._recording

    def close(self) -> None:
        self._close()
