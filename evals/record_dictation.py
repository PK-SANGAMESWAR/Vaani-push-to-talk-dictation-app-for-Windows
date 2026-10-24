"""Record the dictation prompts from your microphone (16 kHz mono, saved under evals/data/dictation/).

    python -m evals.record_dictation --speaker me --mic "laptop array"
    python -m evals.record_dictation --redo dict-004

Enter starts a take, Enter stops it; type `s` to skip a prompt, `q` to quit. Existing takes are kept
unless named with --redo. Nothing leaves this machine, and the folder is gitignored.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .datasets.common import SAMPLE_RATE
from .datasets.dictation import AUDIO_DIR, load_script


def _record() -> np.ndarray:
    import sounddevice as sd

    frames: list[np.ndarray] = []
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=lambda d, *_: frames.append(d.copy())
    ):
        input("  recording... press Enter to stop ")
    return np.concatenate(frames)[:, 0] if frames else np.zeros(0, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    import soundfile as sf

    p = argparse.ArgumentParser(prog="python -m evals.record_dictation", description=__doc__.splitlines()[0])
    p.add_argument("--speaker", default="me", help="label stored in the manifest")
    p.add_argument("--mic", help="mic description stored in the manifest, e.g. 'laptop array'")
    p.add_argument("--redo", nargs="*", default=[], help="prompt ids to re-record")
    args = p.parse_args(argv)

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIO_DIR / "recording.json").write_text(
        json.dumps({"speaker": args.speaker, "mic": args.mic}), encoding="utf-8"
    )
    for row in load_script():
        wav = AUDIO_DIR / f"{row['id']}.wav"
        if wav.exists() and row["id"] not in args.redo:
            continue
        print(f"\n[{row['id']}] ({row['category']})\n  say: {row['ref']}")
        cmd = input("  Enter = record, s = skip, q = quit: ").strip().lower()
        if cmd == "q":
            break
        if cmd == "s":
            continue
        audio = _record()
        sf.write(str(wav), audio, SAMPLE_RATE, subtype="PCM_16")
        print(f"  saved {len(audio) / SAMPLE_RATE:.1f} s")
    print("\nnext: python -m evals.datasets.prepare dictation", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
