"""WER/CER (Open ASR Leaderboard normalization), bootstrap CIs, and latency percentiles.

WER and CER are corpus-level: total edits / total reference units, as on the Open ASR Leaderboard.
CER counts characters of the normalized text, spaces included (jiwer's default).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 0
PERCENTILE_METHOD = "linear"


@lru_cache(maxsize=1)
def _normalizer():
    from .normalizer import EnglishTextNormalizer

    return EnglishTextNormalizer()


def normalize(text: str) -> str:
    return " ".join(_normalizer()(text).split())


@dataclass(frozen=True)
class Edits:
    errors: int  # substitutions + deletions + insertions
    total: int  # reference units (words or characters)


def word_edits(ref_norm: str, hyp_norm: str) -> Edits:
    return _edits(ref_norm.split(), hyp_norm.split())


def char_edits(ref_norm: str, hyp_norm: str) -> Edits:
    # One token per character; spaces become a placeholder so they count as characters.
    def tokens(s: str) -> list[str]:
        return ["<sp>" if c == " " else c for c in s]

    return _edits(tokens(ref_norm), tokens(hyp_norm))


def _edits(ref: list[str], hyp: list[str]) -> Edits:
    """Levenshtein edits between token lists (tokens contain no spaces), via jiwer.

    Empty sides are handled here because jiwer rejects an empty reference.
    """
    if not ref:
        return Edits(len(hyp), 0)
    if not hyp:
        return Edits(len(ref), len(ref))
    import jiwer

    out = jiwer.process_words(" ".join(ref), " ".join(hyp))
    return Edits(out.substitutions + out.deletions + out.insertions, len(ref))


def corpus_rate(edits: Iterable[Edits]) -> float | None:
    edits = list(edits)
    total = sum(e.total for e in edits)
    return None if total == 0 else sum(e.errors for e in edits) / total


def bootstrap_ci(
    edits: Sequence[Edits], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float] | None:
    """95% percentile CI of the corpus rate, resampling utterances with replacement."""
    if not edits or sum(e.total for e in edits) == 0:
        return None
    err = np.array([e.errors for e in edits], dtype=np.float64)
    tot = np.array([e.total for e in edits], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(edits), size=(resamples, len(edits)))
    t = tot[idx].sum(axis=1)
    rates = np.divide(err[idx].sum(axis=1), t, out=np.zeros(resamples), where=t > 0)
    lo, hi = np.percentile(rates, [2.5, 97.5], method=PERCENTILE_METHOD)
    return float(lo), float(hi)


def rate_with_ci(edits: Sequence[Edits]) -> dict:
    rate = corpus_rate(edits)
    ci = bootstrap_ci(edits)
    return {
        "value": rate,
        "ci95": list(ci) if ci else None,
        "errors": sum(e.errors for e in edits),
        "total": sum(e.total for e in edits),
    }


def latency_stats(values_ms: Sequence[float]) -> dict:
    if not values_ms:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "max": None}
    v = np.asarray(values_ms, dtype=np.float64)
    p50, p95 = np.percentile(v, [50, 95], method=PERCENTILE_METHOD)
    return {
        "n": int(v.size),
        "p50": float(p50),
        "p95": float(p95),
        "mean": float(v.mean()),
        "max": float(v.max()),
    }
