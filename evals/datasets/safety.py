"""Safety set: dictated questions/commands (prompt injection) and custom-vocabulary sentences.

Prompts live in `safety/script.jsonl`: `ref` (spoken, verbatim), `ref_clean` (intended text) and
`answer_markers` (words that only show up if the cleanup LLM answers or obeys instead of cleaning).
Audio is synthesized offline with the Windows SAPI voices (System.Speech, no network), alternating
voices, padded with 300 ms of silence each side like a push-to-talk take. TTS speech is cleaner than a
real voice, so these clips test the pipeline's *behaviour* (answering, vocabulary), not ASR accuracy.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .common import DATA_DIR, EVALS_DIR, SAMPLE_RATE, Clip, sha256_file

NAME = "safety"
SCRIPT = EVALS_DIR / "datasets" / "safety" / "script.jsonl"
AUDIO_DIR = DATA_DIR / NAME
VOICES = ("Microsoft Zira Desktop", "Microsoft David Desktop")
PAD_S = 0.3
SYNTH_VERSION = 1  # bump when synthesis changes, so the manifest hash changes with the audio

_PS = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($j in (Get-Content -Raw -Encoding UTF8 $args[0] | ConvertFrom-Json)) {
    $s.SelectVoice($j.voice); $s.Rate = 0; $s.Volume = 100
    $s.SetOutputToWaveFile($j.path, $fmt); $s.Speak($j.text); $s.SetOutputToNull()
}
$s.Dispose()
"""


def load_script() -> list[dict]:
    with SCRIPT.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def synthesize(jobs: list[dict]) -> None:
    """jobs: [{voice, path, text}]; one PowerShell process for all clips (SAPI start-up is slow)."""
    if sys.platform != "win32":
        raise RuntimeError("the safety set is synthesized with Windows SAPI voices; run prepare on Windows")
    with tempfile.TemporaryDirectory() as tmp:
        jf, ps = Path(tmp) / "jobs.json", Path(tmp) / "tts.ps1"
        jf.write_text(json.dumps(jobs), encoding="utf-8")
        ps.write_text(_PS, encoding="utf-8")
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps), str(jf)],
            check=True,
            env=os.environ | {"PYTHONIOENCODING": "utf-8"},
        )


def _pad(path: Path) -> float:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="int16")
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: SAPI wrote {sr} Hz")
    pad = np.zeros(int(PAD_S * SAMPLE_RATE), dtype=np.int16)
    audio = np.concatenate([pad, audio, pad])
    sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")
    return audio.size / SAMPLE_RATE


def prepare(name: str, n: int, seed: int, full: bool) -> tuple[dict, list[Clip]]:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    script = load_script()
    jobs = [
        {"voice": VOICES[i % len(VOICES)], "path": str(AUDIO_DIR / f"{row['id']}.wav"), "text": row["ref"]}
        for i, row in enumerate(script)
    ]
    synthesize(jobs)
    clips, audio_sha = [], {}
    for row, job in zip(script, jobs, strict=True):
        rel = f"{NAME}/{row['id']}.wav"
        dur = _pad(DATA_DIR / rel)
        audio_sha[row["id"]] = sha256_file(DATA_DIR / rel)
        tags = {"voice": job["voice"]}
        if row.get("answer_markers"):
            tags["answer_markers"] = "|".join(row["answer_markers"])
        clips.append(
            Clip(
                row["id"],
                NAME,
                rel,
                round(dur, 3),
                row["ref"],
                row["category"],
                speaker="tts",
                mic="synthetic",
                ref_clean=row["ref_clean"],
                tags=tags,
            )
        )
    meta = {
        "dataset": NAME,
        "source": "evals/datasets/safety/script.jsonl, synthesized offline with Windows SAPI voices",
        "script_sha256": sha256_file(SCRIPT),
        "synth_version": SYNTH_VERSION,
        "voices": list(VOICES),
        "pad_s": PAD_S,
        "audio_sha256": audio_sha,  # SAPI output can differ across Windows builds; this pins what we ran
        "license": "prompts: same as Vaani; audio: generated locally",
    }
    return meta, clips
