"""Render the README benchmark table from committed result files, and check the README still matches.

    python -m evals.bench_table            # print the block
    python -m evals.bench_table --write    # replace the block between the markers in README.md
    python -m evals.bench_table --check    # exit 1 if README.md differs (also run by pytest)

Only results from a clean working tree are allowed here (`provenance.git.dirty` must be false); the
function refuses anything else, so a README number can always be traced to a committed file and commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
START, END = "<!-- bench:start -->", "<!-- bench:end -->"

# Result files the README quotes. Replace with newer clean-tree runs when they exist.
GPU_RUN = "evals/results/20261024T101301Z_2131f02_gpu-fp32-4sets.json"
VALIDATION_RUN = "evals/results/20261024T100052Z_255cdb2_validate-novad.json"
NAMES = {
    "librispeech-clean": "LibriSpeech test-clean",
    "librispeech-other": "LibriSpeech test-other",
    "fleurs-en": "FLEURS en_us",
    "svarah": "Svarah (Indian-accented English)",
}


def _load(rel: str) -> dict[str, Any]:
    r = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    if r["provenance"]["git"]["dirty"] is not False:
        raise ValueError(f"{rel} was produced on a dirty working tree and cannot be quoted")
    return r


def _pct(m: dict[str, Any]) -> str:
    lo, hi = m["ci95"]
    return f"{m['value'] * 100:.2f} [{lo * 100:.1f}–{hi * 100:.1f}]"


def render() -> str:
    r, v = _load(GPU_RUN), _load(VALIDATION_RUN)
    p = r["provenance"]
    gpu = p["hardware"]["gpu"]
    commit = p["git"]["commit"][:7]
    rows = []
    for ds, g in r["summary"]["by_dataset"].items():
        lat = g["latency_ms"]["total"]
        rows.append(
            f"| {NAMES.get(ds, ds)} | {g['n']} | {_pct(g['wer_final'])} | "
            f"{lat['p50']:.0f} / {lat['p95']:.0f} | {g['rtfx']} |"
        )
    o = r["summary"]["overall"]
    lat = o["latency_ms"]["total"]
    rows.append(
        f"| **All four** | {o['n']} | {_pct(o['wer_final'])} | {lat['p50']:.0f} / {lat['p95']:.0f} | {o['rtfx']} |"
    )
    ref = v["published_reference"]
    return "\n".join(
        [
            START,
            f"Parakeet TDT 0.6B v3 fp32 on {gpu['name']} ({gpu['memory_total_mb'] / 1024:.0f} GB), driver "
            f"{gpu['driver']}, {p['hardware']['cpu']}; shipped pipeline (Silero VAD, no cleanup), "
            f"{r['summary']['overall']['n']} clips = 200 per set (seed 0). WER is corpus-level on "
            "Open ASR Leaderboard-normalized text with a 95% bootstrap CI. Latency is `Pipeline.process` "
            "per clip (no paste). From a clean commit "
            f"(`{commit}`): [`{GPU_RUN.rsplit('/', 1)[1]}`]({GPU_RUN}).",
            "",
            "| Set | Clips | WER % [95% CI] | Latency ms p50 / p95 | ASR RTFx |",
            "|---|---:|---|---|---:|",
            *rows,
            "",
            f"Harness check: the same model without VAD on the full LibriSpeech test-clean "
            f"({v['summary']['overall']['n']} clips) gives {ref['ours_wer_raw_percent']:.2f}% WER "
            f"[{ref['ours_ci95_percent'][0]:.2f}–{ref['ours_ci95_percent'][1]:.2f}], against 1.93% published on "
            f"the model card (inside our CI: {'yes' if ref['published_within_our_ci'] else 'no'}). "
            f"Result: [`{VALIDATION_RUN.rsplit('/', 1)[1]}`]({VALIDATION_RUN}).",
            END,
        ]
    )


def splice(readme: str, block: str) -> str:
    a, b = readme.find(START), readme.find(END)
    if a < 0 or b < a:
        raise ValueError(f"README.md has no {START} … {END} block")
    return readme[:a] + block + readme[b + len(END) :]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.bench_table", description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    block = render()
    if not (args.write or args.check):
        print(block)
        return 0
    current = README.read_text(encoding="utf-8")
    new = splice(current, block)
    if args.check:
        if new != current:
            print("README.md benchmark block is out of date: run python -m evals.bench_table --write")
            return 1
        return 0
    README.write_text(new, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
