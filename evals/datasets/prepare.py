"""Download eval datasets and write their manifests.

    python -m evals.datasets.prepare                      # all datasets, 200 clips each, seed 0
    python -m evals.datasets.prepare librispeech-clean --full
    python -m evals.datasets.prepare fleurs-en --n 50 --seed 1

This is the only part of Vaani that downloads data; the app itself never does.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from . import fleurs, librispeech, svarah
from .common import DEFAULT_N, DEFAULT_SEED, Clip, manifest_path, write_manifest

PREPARERS: dict[str, Callable[[str, int, int, bool], tuple[dict, list[Clip]]]] = {
    "librispeech-clean": librispeech.prepare,
    "librispeech-other": librispeech.prepare,
    "fleurs-en": fleurs.prepare,
    "svarah": svarah.prepare,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.datasets.prepare", description=__doc__.splitlines()[0])
    p.add_argument("datasets", nargs="*", default=list(PREPARERS), choices=list(PREPARERS))
    p.add_argument("--n", type=int, default=DEFAULT_N, help="clips per dataset (seeded subset)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--full", action="store_true", help="use every clip in the split")
    args = p.parse_args(argv)

    failed = []
    for name in args.datasets:
        try:
            meta, clips = PREPARERS[name](name, args.n, args.seed, args.full)
        except Exception as e:
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
            failed.append(name)
            continue
        meta.update({"n": len(clips), "seed": None if args.full else args.seed, "full": args.full})
        out = manifest_path(name, args.n, args.seed, args.full)
        write_manifest(out, meta, clips)
        hours = sum(c.duration_s for c in clips) / 3600
        print(f"[{name}] {len(clips)} clips, {hours:.2f} h -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
