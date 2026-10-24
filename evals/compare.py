"""Combine several eval results into one comparison table (Markdown + JSON).

    python -m evals.compare evals/results/A.json evals/results/B.json ... --label phase4-baselines

Every number is read from the result JSONs; nothing is recomputed except latency over speech clips
(from the per-clip timings stored in each file). Runs must share manifests (same sha256) to be compared;
a mismatch is reported and makes the command exit non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import metrics, sysinfo

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ["librispeech-clean", "librispeech-other", "fleurs-en", "svarah"]


def _pct(m: dict | None, ci: bool = True) -> str:
    if not m or m.get("value") is None:
        return "–"
    s = f"{m['value'] * 100:.2f}"
    if ci and m.get("ci95") and m["ci95"][0] is not None:
        lo, hi = m["ci95"]
        s += f" [{lo * 100:.1f}–{hi * 100:.1f}]"
    return s


def _frac(m: dict | None) -> str:
    if not m or not m.get("n"):
        return "–"
    return f"{m['k']}/{m['n']}"


def _ms(v: float | None) -> str:
    return "–" if v is None else f"{v:.0f}"


def row(r: dict[str, Any], path: Path) -> dict[str, Any]:
    p, s, sf = r["provenance"], r["summary"], r.get("safety") or {}
    speech = [u for u in r["utterances"] if u["wer_raw"][1] > 0]
    public = [u for u in speech if u["dataset"] in PUBLIC]
    by = s["by_dataset"]
    gpu = (r.get("resources") or {}).get("gpu") or {}
    cleanup = p.get("cleanup") or {}
    return {
        "label": r["label"],
        "file": sysinfo.rel_path(path),
        "dirty": p["git"].get("dirty"),
        "commit": (p["git"].get("commit") or "")[:7],
        "code": (p.get("code") or {}).get("vaani_diff_sha256"),
        "code_changed_mid_run": (p.get("code") or {}).get("changed_during_run"),
        "asr": f"{p['asr']['model']} {p['asr']['device']}/{p['asr']['quantization']}",
        "cleanup": cleanup.get("model") if r["settings"]["cleanup"] else "off",
        "wer_final": {d: by[d]["wer_final"] if d in by else None for d in PUBLIC},
        "wer_public_final": metrics.rate_with_ci([metrics.Edits(*u["wer_final"]) for u in public]),
        "dictation_clean_final": by.get("dictation", {}).get("clean_final"),
        "dictation_strict_final": by.get("dictation", {}).get("strict_final"),
        "hallucination": (sf.get("hallucination") or {}).get("rate"),
        "injection_answered": (sf.get("injection") or {}).get("answered"),
        "injection_attempted": (sf.get("injection") or {}).get("llm_attempted"),
        "regressions": (sf.get("regressions") or {}).get("regressed"),
        "vocab_recall": (sf.get("vocabulary") or {}).get("recall_final"),
        "latency_total_speech": metrics.latency_stats([u["total_ms"] for u in speech]),
        "latency_cleanup": s["overall"]["latency_ms"]["cleanup"],
        "rtfx": s["overall"]["rtfx"],
        "cold_start_ms": r["cold_start_ms"]["load_plus_warmup"],
        "gpu_peak_mb": gpu.get("peak_used_mb"),
        "gpu_baseline_mb": gpu.get("baseline_used_mb"),
        "manifests": {m["path"]: m["sha256"] for m in r["manifests"]},
    }


def render(rows: list[dict[str, Any]], label: str, mismatch: list[str]) -> str:
    out = [f"# Vaani baseline comparison: {label}", ""]
    if any(x["dirty"] for x in rows):
        out += [
            "> Some runs come from a **dirty working tree** (see the Dirty column); quote none of them.",
            "",
        ]
    if mismatch:
        out += ["> **Manifests differ between runs; rows are not comparable:** " + "; ".join(mismatch), ""]
    out += [
        "## Accuracy (WER % of the pasted text, normalized, 95% CI)",
        "",
        "| Config | ASR | Cleanup | LS clean | LS other | FLEURS | Svarah | 4 sets pooled | "
        "Dictation content WER | Dictation strict WER |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for x in rows:
        w = x["wer_final"]
        out.append(
            f"| {x['label']} | {x['asr']} | {x['cleanup']} | "
            + " | ".join(_pct(w[d], ci=False) for d in PUBLIC)
            + f" | {_pct(x['wer_public_final'])} | {_pct(x['dictation_clean_final'])} | "
            f"{_pct(x['dictation_strict_final'])} |"
        )
    out += [
        "",
        "## Safety",
        "",
        "| Config | Silence/noise pasting text | Injection answered | LLM tried to answer | "
        "Cleanup regressions | Vocabulary recall |",
        "|---|---|---|---|---|---|",
    ]
    for x in rows:
        out.append(
            f"| {x['label']} | {_frac(x['hallucination'])} | {_frac(x['injection_answered'])} | "
            f"{_frac(x['injection_attempted'])} | {_frac(x['regressions'])} | {_frac(x['vocab_recall'])} |"
        )
    out += [
        "",
        "## Latency and resources",
        "",
        "Pipeline time per clip (ASR + text + cleanup; no paste), clips with speech only. GPU memory is",
        "device-wide (includes Ollama and other apps).",
        "",
        "| Config | Total ms p50 / p95 | Cleanup ms p50 / p95 (n) | ASR RTFx | Cold start ms | "
        "GPU MB baseline → peak | Dirty | Code (vaani/ diff) | Result file |",
        "|---|---|---|---:|---:|---|---|---|---|",
    ]
    for x in rows:
        lt, lc = x["latency_total_speech"], x["latency_cleanup"]
        gpu = f"{x['gpu_baseline_mb']:.0f} → {x['gpu_peak_mb']:.0f}" if x["gpu_peak_mb"] is not None else "–"
        out.append(
            f"| {x['label']} | {_ms(lt['p50'])} / {_ms(lt['p95'])} | "
            f"{_ms(lc['p50'])} / {_ms(lc['p95'])} ({lc['n']}) | {x['rtfx'] or '–'} | "
            f"{_ms(x['cold_start_ms'])} | {gpu} | {'yes' if x['dirty'] else 'no'} ({x['commit']}) | "
            f"{(x['code'] or '?')[:8]}{' (changed mid-run)' if x['code_changed_mid_run'] else ''} | "
            f"`{x['file']}` |"
        )
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.compare", description=__doc__.splitlines()[0])
    p.add_argument("results", nargs="+", help="result JSON files, in table order")
    p.add_argument("--label", default="comparison")
    p.add_argument("--out", default=str(ROOT / "evals" / "results"))
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    rows = []
    for f in args.results:
        path = Path(f)
        rows.append(row(json.loads(path.read_text(encoding="utf-8")), path))
    ref = rows[0]["manifests"]
    mismatch = [x["label"] for x in rows[1:] if x["manifests"] != ref]
    codes = {x["code"] for x in rows}
    if len(codes) > 1 or any(x["code_changed_mid_run"] for x in rows):
        print("warning: runs used different uncommitted vaani/ code (see Code column)", file=sys.stderr)

    created = datetime.now(UTC)
    stem = f"{created:%Y%m%dT%H%M%SZ}_{args.label}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = render(rows, args.label, mismatch)
    (out / f"{stem}.md").write_text(md, encoding="utf-8", newline="\n")
    (out / f"{stem}.json").write_text(
        json.dumps({"label": args.label, "created_utc": created.isoformat(timespec="seconds"), "rows": rows},
                   indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )  # fmt: skip
    print(md)
    print(f"wrote {out / f'{stem}.md'}", file=sys.stderr)
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
