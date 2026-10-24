"""Dictation set: prompts in `dictation/script.jsonl`, read aloud and recorded by you.

Each prompt has `ref` (what is said, verbatim: fillers, self-corrections, spoken numbers) and
`ref_clean` (the text a user wants pasted: punctuation, casing, digits, no fillers). Record them with

    python -m evals.record_dictation --speaker me --mic "laptop array"

then `python -m evals.datasets.prepare dictation`. Audio stays in evals/data/dictation/ (gitignored,
never committed); the manifest lists only the clips you recorded. Because the audio is yours alone,
these numbers describe one speaker and one mic: report them as such, never as a benchmark.
"""

from __future__ import annotations

import json

from .common import DATA_DIR, EVALS_DIR, Clip, audio_info, sha256_file

NAME = "dictation"
SCRIPT = EVALS_DIR / "datasets" / "dictation" / "script.jsonl"
AUDIO_DIR = DATA_DIR / NAME


def load_script() -> list[dict]:
    with SCRIPT.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _recording_info() -> tuple[str | None, str | None]:
    info = AUDIO_DIR / "recording.json"
    if not info.exists():
        return None, None
    d = json.loads(info.read_text(encoding="utf-8"))
    return d.get("speaker"), d.get("mic")


def prepare(name: str, n: int, seed: int, full: bool) -> tuple[dict, list[Clip]]:
    speaker, mic = _recording_info()
    script = load_script()
    clips, missing = [], []
    for row in script:
        rel = f"{NAME}/{row['id']}.wav"
        if not (DATA_DIR / rel).exists():
            missing.append(row["id"])
            continue
        sr, dur = audio_info(DATA_DIR / rel)
        if sr != 16_000:
            raise ValueError(f"{rel} is {sr} Hz; record with evals.record_dictation")
        clips.append(
            Clip(
                row["id"],
                NAME,
                rel,
                round(dur, 3),
                row["ref"],
                row["category"],
                speaker=speaker,
                mic=mic,
                ref_clean=row["ref_clean"],
            )
        )
    if not clips:
        raise RuntimeError("no recordings in evals/data/dictation/; run: python -m evals.record_dictation")
    meta = {
        "dataset": NAME,
        "source": "evals/datasets/dictation/script.jsonl, recorded locally",
        "script_sha256": sha256_file(SCRIPT),
        "population": len(script),
        "missing": missing,
        "speaker": speaker,
        "mic": mic,
    }
    return meta, clips
