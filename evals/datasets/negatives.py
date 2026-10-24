"""Synthetic silence / noise clips: Vaani must paste nothing for any of them.

Generated deterministically (numpy, fixed seeds, 16-bit PCM) into evals/data/negatives/, so there is
nothing to download and the committed manifest pins the exact set. Every clip has `ref == ""`; the
report counts clips that still produce text (the false-positive rate of VAD + ASR).

Not covered: real room noise, music, or background speech. Those need recordings (see `dictation`).
"""

from __future__ import annotations

import hashlib

import numpy as np

from .common import DATA_DIR, SAMPLE_RATE, Clip

NAME = "negatives"
DURATIONS_S = (1.5, 3.0, 6.0, 10.0)
VARIANTS = 2  # different seeds per (kind, duration)
SYNTH_VERSION = 1  # bump when a generator changes, so the manifest hash changes with the audio


def _dbfs(x: np.ndarray, dbfs: float) -> np.ndarray:
    """Scale to the given RMS level (dBFS); all-zero input stays zero."""
    rms = float(np.sqrt(np.mean(x**2)))
    return x if rms == 0 else x * (10 ** (dbfs / 20) / rms)


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    spec /= np.sqrt(np.maximum(np.arange(len(spec)), 1))
    return np.fft.irfft(spec, n)


def _brown(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.cumsum(rng.standard_normal(n))
    return x - x.mean()


def _hum(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    x = sum(np.sin(2 * np.pi * 60 * k * t + rng.uniform(0, 6.28)) / k for k in (1, 2, 3, 5))
    return x + 0.05 * rng.standard_normal(n)


def _clicks(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse keyboard-like transients (2-8 ms decaying noise bursts) over a quiet floor."""
    x = 0.002 * rng.standard_normal(n)
    for pos in rng.integers(0, max(n - 200, 1), size=max(int(n / SAMPLE_RATE * 3), 1)):
        m = int(rng.uniform(0.002, 0.008) * SAMPLE_RATE)
        x[pos : pos + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / (m / 4))
    return x


# kind -> (generator, RMS dBFS or None for "as generated")
KINDS: dict[str, tuple] = {
    "silence-digital": (lambda n, rng: np.zeros(n), None),
    "silence-dither": (lambda n, rng: rng.integers(-1, 2, n) / 32768, None),
    "noise-room": (_pink, -60.0),
    "noise-fan": (_brown, -45.0),
    "noise-white-loud": (lambda n, rng: rng.standard_normal(n), -30.0),
    "hum-60hz": (_hum, -40.0),
    "keyboard-clicks": (_clicks, -35.0),
}


def synth(kind: str, duration_s: float, seed: int) -> np.ndarray:
    gen, level = KINDS[kind]
    # hashlib, not hash(): str hashes are salted per process and would break determinism
    kind_seed = int.from_bytes(hashlib.sha256(kind.encode()).digest()[:4], "little")
    rng = np.random.default_rng([kind_seed, seed, SYNTH_VERSION])
    x = gen(int(duration_s * SAMPLE_RATE), rng)
    if level is not None:
        x = _dbfs(x, level)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def prepare(name: str, n: int, seed: int, full: bool) -> tuple[dict, list[Clip]]:
    import soundfile as sf

    out = DATA_DIR / NAME
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for kind in KINDS:
        for dur in DURATIONS_S:
            for v in range(VARIANTS):
                cid = f"{kind}-{dur:g}s-{v}"
                rel = f"{NAME}/{cid}.wav"
                sf.write(str(DATA_DIR / rel), synth(kind, dur, v), SAMPLE_RATE, subtype="PCM_16")
                clips.append(Clip(cid, NAME, rel, dur, "", kind, tags={"kind": kind}))
    meta = {
        "dataset": NAME,
        "source": "synthetic (evals/datasets/negatives.py)",
        "synth_version": SYNTH_VERSION,
        "license": "CC0-1.0",
        "kinds": list(KINDS),
        "durations_s": list(DURATIONS_S),
        "variants": VARIANTS,
    }
    return meta, clips
