"""Manifest format, seeded subsets, downloads, and audio loading shared by all eval datasets.

Manifest = JSONL. Line 1 is {"_meta": {...}} (source, revision, checksums, seed, n). Every other line is
one clip (see `Clip`). Audio paths are relative to evals/data/, which is gitignored; manifests are
committed so the exact subset and references are pinned in git.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tarfile
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

EVALS_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EVALS_DIR / "data"
MANIFEST_DIR = EVALS_DIR / "datasets" / "manifests"
SAMPLE_RATE = 16_000
DEFAULT_N = 200
DEFAULT_SEED = 0


@dataclass
class Clip:
    id: str
    dataset: str
    audio: str  # path relative to DATA_DIR (posix separators)
    duration_s: float
    ref: str  # verbatim reference ("" for silence/noise-only clips)
    category: str
    speaker: str | None = None
    mic: str | None = None
    ref_clean: str | None = None  # intended clean text (dictation sets); None for public ASR sets
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return DATA_DIR / self.audio


# Datasets that are generated or hand-made and have no subset/split: one manifest, `<name>.fixed.jsonl`.
FIXED_DATASETS = frozenset({"negatives", "dictation", "safety"})


def manifest_path(dataset: str, n: int = DEFAULT_N, seed: int = DEFAULT_SEED, full: bool = False) -> Path:
    if dataset in FIXED_DATASETS:
        return MANIFEST_DIR / f"{dataset}.fixed.jsonl"
    return MANIFEST_DIR / (f"{dataset}.full.jsonl" if full else f"{dataset}.n{n}.s{seed}.jsonl")


def write_manifest(path: Path, meta: dict[str, Any], clips: list[Clip]) -> None:
    """Deterministic output (sorted clips, sorted keys, LF endings) so re-running prepare is a no-op diff."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"_meta": meta}, sort_keys=True, ensure_ascii=False)]
    lines += [
        json.dumps(asdict(c), sort_keys=True, ensure_ascii=False) for c in sorted(clips, key=lambda c: c.id)
    ]
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def read_manifest(path: Path) -> tuple[dict[str, Any], list[Clip]]:
    meta: dict[str, Any] = {}
    clips: list[Clip] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
            else:
                clips.append(Clip(**row))
    return meta, clips


def select_subset(ids: list[str], n: int, seed: int) -> list[str]:
    """Seeded sample of n ids (all ids if n >= len). Input order doesn't matter; output is sorted."""
    pool = sorted(set(ids))
    if n >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    """Stream url to dest (skipped if dest exists). Writes to .part first so a crash never leaves a bad file."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    tmp.replace(dest)
    return dest


def extract_tar(archive: Path, dest: Path) -> None:
    """Extract with the 'data' filter (no absolute paths, links outside dest, or device files)."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(dest, filter="data")


def load_audio(path: Path) -> np.ndarray:
    """Mono float32 at 16 kHz. Raises if the file isn't 16 kHz (all bundled sources are)."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path} is {sr} Hz; eval audio must be {SAMPLE_RATE} Hz")
    return audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]


def audio_info(path: Path) -> tuple[int, float]:
    """(sample_rate, duration_s) without decoding the whole file."""
    import soundfile as sf

    info = sf.info(str(path))
    return info.samplerate, info.frames / info.samplerate
