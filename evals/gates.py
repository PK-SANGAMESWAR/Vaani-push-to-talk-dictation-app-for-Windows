"""Regression gates: compare an eval result with a baseline result and fixed limits.

    python -m evals.run_eval --gates gpu ...                 # gate a fresh run; exit 1 on regression
    python -m evals.gates evals/results/X.json --profile gpu # gate an existing result file

Limits and profiles live in `evals/thresholds.toml`. Nothing here recomputes metrics: every value is read
from the result JSON (and the baseline JSON it names), so a gate failure can be traced to numbers in
committed files.

A gate that applies but has no data fails ("no data"); a typo in a dataset list must not read as a pass.
Gates that do not apply to this run (latency on a cleanup-on run, injection on a cleanup-off run, anything
the profile does not set) are reported as skipped with the reason.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from . import metrics

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = ROOT / "evals" / "thresholds.toml"

PASS, FAIL, SKIP = "pass", "fail", "skip"


def load_thresholds(path: Path = THRESHOLDS) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def profile_settings(thresholds: dict[str, Any], name: str) -> dict[str, Any]:
    """Global `[gates]` limits overridden by `[profiles.<name>]`."""
    profiles = thresholds.get("profiles", {})
    if name not in profiles:
        raise KeyError(f"unknown gate profile {name!r} (have: {', '.join(sorted(profiles)) or 'none'})")
    return {**thresholds.get("gates", {}), **profiles[name]}


def _check(gate: str, status: str, detail: str, value: Any = None, limit: Any = None) -> dict[str, Any]:
    return {"gate": gate, "status": status, "detail": detail, "value": value, "limit": limit}


def _identity(r: dict[str, Any]) -> dict[str, Any]:
    p = r["provenance"]
    cleanup_on = r["settings"]["cleanup"]
    return {
        "asr": [p["asr"]["model"], p["asr"]["device"], p["asr"]["quantization"]],
        "use_vad": r["settings"]["use_vad"],
        "cleanup": (p.get("cleanup") or {}).get("model") if cleanup_on else None,
    }


def comparable(run: dict[str, Any], base: dict[str, Any]) -> str | None:
    """Why `run` cannot be compared with `base` (None = comparable)."""
    a, b = _identity(run), _identity(base)
    if a != b:
        return f"configuration differs from the baseline: run {a} vs baseline {b}"
    return None


def _wer_gate(run: dict[str, Any], base: dict[str, Any], max_abs_pct: float) -> list[dict[str, Any]]:
    run_m = {m["meta"]["dataset"]: m["sha256"] for m in run["manifests"]}
    base_m = {m["meta"]["dataset"]: m["sha256"] for m in base["manifests"]}
    run_by, base_by = run["summary"]["by_dataset"], base["summary"]["by_dataset"]
    checks = []
    for ds in sorted(run_by):
        name = f"wer[{ds}]"
        if run_by[ds]["n_scored"] == 0:
            continue  # silence/noise sets: no reference words, covered by the silence gate
        if ds not in base_by:
            checks.append(_check(name, FAIL, "dataset is not in the baseline"))
            continue
        if run_m.get(ds) != base_m.get(ds):
            checks.append(_check(name, FAIL, "manifest differs from the baseline's: not comparable"))
            continue
        now, then = run_by[ds]["wer_final"]["value"], base_by[ds]["wer_final"]["value"]
        if now is None or then is None:
            checks.append(_check(name, FAIL, "no data"))
            continue
        delta = (now - then) * 100
        status = PASS if delta <= max_abs_pct + 1e-9 else FAIL
        checks.append(
            _check(
                name,
                status,
                f"{now * 100:.2f}% vs baseline {then * 100:.2f}% ({delta:+.2f} points, limit +{max_abs_pct})",
                round(delta, 4),
                max_abs_pct,
            )
        )
    if not checks:
        checks.append(_check("wer", FAIL, "no dataset with reference words in this run"))
    return checks


def _silence_gate(run: dict[str, Any], max_rate: float) -> dict[str, Any]:
    rate = (run.get("safety") or {}).get("hallucination", {}).get("rate")
    if not rate or not rate["n"]:
        return _check("silence", FAIL, "no data (run the negatives set: --datasets ...,negatives)")
    ok = rate["k"] <= max_rate * rate["n"]
    return _check(
        "silence",
        PASS if ok else FAIL,
        f"{rate['k']}/{rate['n']} silence/noise clips pasted text (limit {max_rate * 100:g}%)",
        rate["value"],
        max_rate,
    )


def _injection_gate(run: dict[str, Any], max_rate: float) -> dict[str, Any]:
    if not run["settings"]["cleanup"]:
        return _check("injection", SKIP, "cleanup is off in this run")
    inj = (run.get("safety") or {}).get("injection", {})
    rate = inj.get("answered")
    if not rate or not rate["n"]:
        return _check("injection", FAIL, "no data (run the safety set with cleanup on)")
    ok = rate["k"] <= max_rate * rate["n"]
    return _check(
        "injection",
        PASS if ok else FAIL,
        f"{rate['k']}/{rate['n']} dictated questions/commands answered (limit {max_rate * 100:g}%)",
        rate["value"],
        max_rate,
    )


def _latency_gate(run: dict[str, Any], max_p95_ms: float) -> dict[str, Any]:
    if run["settings"]["cleanup"]:
        return _check("latency", SKIP, "cleanup is on; the p95 limit is for the no-cleanup path")
    speech = [u["total_ms"] for u in run["utterances"] if u["wer_raw"][1] > 0]
    if not speech:
        return _check("latency", FAIL, "no clips with speech")
    p95 = metrics.latency_stats(speech)["p95"]
    return _check(
        "latency",
        PASS if p95 <= max_p95_ms else FAIL,
        f"p95 {p95:.0f} ms over {len(speech)} speech clips (limit {max_p95_ms:g} ms)",
        round(p95, 1),
        max_p95_ms,
    )


def _network_gate(run: dict[str, Any]) -> dict[str, Any]:
    net = run.get("network")
    if not net or not net.get("blocked"):
        return _check("network", FAIL, "run without --block-network: no evidence")
    attempts = net["attempts"]
    return _check(
        "network",
        PASS if not attempts else FAIL,
        f"{len(attempts)} non-local connection attempt(s)" + (f": {attempts[:3]}" if attempts else ""),
        len(attempts),
        0,
    )


def evaluate(
    run: dict[str, Any],
    profile: str,
    baseline: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Gate `run` (a result dict). `baseline` defaults to the file the profile names."""
    th = thresholds or load_thresholds()
    s = profile_settings(th, profile)
    checks: list[dict[str, Any]] = []

    base_path = s.get("baseline")
    if "wer_abs_max_pct" in s:
        if baseline is None and base_path:
            baseline = json.loads((ROOT / base_path).read_text(encoding="utf-8"))
        if baseline is None:
            checks.append(_check("wer", FAIL, "profile names no baseline file"))
        elif (why := comparable(run, baseline)) is not None:
            checks.append(_check("wer", FAIL, why))
        else:
            checks += _wer_gate(run, baseline, s["wer_abs_max_pct"])
    if "silence_hallucination_max" in s:
        checks.append(_silence_gate(run, s["silence_hallucination_max"]))
    if "injection_answered_max" in s:
        checks.append(_injection_gate(run, s["injection_answered_max"]))
    if "p95_latency_ms_max" in s:
        checks.append(_latency_gate(run, s["p95_latency_ms_max"]))
    if s.get("require_no_network"):
        checks.append(_network_gate(run))
    if not checks:
        checks.append(_check("profile", FAIL, f"profile {profile!r} enables no gates"))

    passed = all(c["status"] != FAIL for c in checks)
    return {"profile": profile, "baseline": base_path, "passed": passed, "checks": checks}


def render_text(g: dict[str, Any]) -> str:
    icon = {PASS: "PASS", FAIL: "FAIL", SKIP: "skip"}
    lines = [f"Regression gates, profile {g['profile']!r} (baseline: {g['baseline'] or '–'})"]
    lines += [f"  {icon[c['status']]}  {c['gate']}: {c['detail']}" for c in g["checks"]]
    lines.append("  => " + ("all gates passed" if g["passed"] else "REGRESSION: at least one gate failed"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.gates", description=__doc__.splitlines()[0])
    p.add_argument("result", help="result JSON written by evals.run_eval")
    p.add_argument("--profile", required=True, help="profile in evals/thresholds.toml")
    p.add_argument("--baseline", help="baseline result JSON (default: the profile's)")
    args = p.parse_args(argv)
    run = json.loads(Path(args.result).read_text(encoding="utf-8"))
    base = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None
    g = evaluate(run, args.profile, base)
    print(render_text(g))
    return 0 if g["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
