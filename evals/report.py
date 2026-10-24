"""Render an eval result (the JSON dict) as Markdown. Every number shown comes from the JSON."""

from __future__ import annotations

from typing import Any


def _pct(m: dict | None) -> str:
    if not m or m.get("value") is None:
        return "–"
    s = f"{m['value'] * 100:.2f}"
    if m.get("ci95"):
        lo, hi = m["ci95"]
        s += f" [{lo * 100:.2f}, {hi * 100:.2f}]"
    return s


def _ms(v: float | None) -> str:
    return "–" if v is None else f"{v:.0f}"


def _row(name: str, g: dict[str, Any]) -> str:
    lat = g["latency_ms"]
    return (
        f"| {name} | {g['n']} | {g['audio_hours']:.2f} | {_pct(g['wer_raw'])} | {_pct(g['wer_final'])} | "
        f"{_pct(g['cer_final'])} | {_ms(lat['asr']['p50'])} / {_ms(lat['asr']['p95'])} | "
        f"{_ms(lat['cleanup']['p50'])} / {_ms(lat['cleanup']['p95'])} (n={lat['cleanup']['n']}) | "
        f"{_ms(lat['total']['p50'])} / {_ms(lat['total']['p95'])} | {g['rtfx'] or '–'} |"
    )


HEADER = (
    "| Set | Clips | Hours | WER raw % [95% CI] | WER final % [95% CI] | CER final % [95% CI] | "
    "ASR ms p50 / p95 | Cleanup ms p50 / p95 | Total ms p50 / p95 | RTFx |\n"
    "|---|---:|---:|---|---|---|---|---|---|---:|"
)


def render_summary_table(r: dict[str, Any]) -> str:
    s = r["summary"]
    rows = [_row(k, v) for k, v in s["by_dataset"].items()]
    if len(s["by_dataset"]) > 1:
        rows.append(_row("**all**", s["overall"]))
    return HEADER + "\n" + "\n".join(rows)


def render(r: dict[str, Any]) -> str:
    p = r["provenance"]
    git = p["git"]
    title = f"# Vaani eval: {r['label']}"
    if git.get("dirty"):
        title += " (DIRTY TREE: uncommitted changes)"
    out = [title, ""]
    if r["diagnostic"]["no_vad"]:
        out += [
            "> **DIAGNOSTIC RUN, not the shipped pipeline.** VAD was bypassed (`--no-vad`): each clip went",
            "> straight into the ASR model. Use it only to compare the model with published numbers.",
            "",
        ]
    asr, hw = p["asr"], p["hardware"]
    gpu = hw.get("gpu") or {}
    out += [
        f"- Created: {r['created_utc']} · commit `{(git.get('commit') or '?')[:12]}` on `{git.get('branch')}`",
        f"- ASR: `{asr['model']}` on **{asr['device']}** ({asr['quantization']}), VAD "
        f"{'on' if r['settings']['use_vad'] else '**off**'} · cleanup "
        f"{'on (' + p['cleanup']['model'] + ')' if r['settings']['cleanup'] else 'off'}",
        f"- GPU: {gpu.get('name', gpu.get('error', '–'))}, driver {gpu.get('driver', '–')} · "
        f"CPU: {hw['cpu']} · RAM {hw['ram_gb']} GB",
        f"- Config sha256 `{p['config']['sha256'][:16]}` · onnxruntime-gpu {p['versions'].get('onnxruntime-gpu')} · "
        f"onnx-asr {p['versions'].get('onnx-asr')}",
        "",
        "## Summary",
        "",
        render_summary_table(r),
        "",
        "WER/CER are corpus-level after Open ASR Leaderboard normalization. *raw* = ASR output; *final* = "
        "what Vaani would paste (replacements, snippets, cleanup). CIs: seeded bootstrap, "
        f"{r['settings']['bootstrap']['resamples']} resamples over utterances. Latency excludes audio decode "
        "and paste.",
        "",
    ]

    ref = r.get("published_reference")
    if ref:
        lo, hi = ref["ours_ci95_percent"]
        out += [
            "## Comparison with published WER",
            "",
            "| | WER % |",
            "|---|---|",
            f"| Published | {ref['wer_percent']:.2f} |",
            f"| This run (raw) | {ref['ours_wer_raw_percent']:.2f} [{lo:.2f}, {hi:.2f}] |",
            f"| Delta (ours − published) | {ref['delta_abs_percent']:+.2f} |",
            "",
            f"Published value inside our 95% CI: **{'yes' if ref['published_within_our_ci'] else 'no'}**.  ",
            f"Source: {ref['source'] or '–'}  ",
            f"Note: {ref['note'] or '–'}",
            "",
        ]

    cs = r["cold_start_ms"]
    out += [
        "## Cold start (excluded from per-clip latency)",
        "",
        f"- Model load: {_ms(cs['model_load'])} ms · warm-up: {_ms(cs['warmup'])} ms · "
        f"total: {_ms(cs['load_plus_warmup'])} ms"
        + (f" · Ollama prewarm: {_ms(cs['llm_prewarm'])} ms" if cs.get("llm_prewarm") is not None else ""),
        "",
        "## Memory",
        "",
    ]
    res = r["resources"]
    g = res.get("gpu")
    if g:
        out.append(
            f"- GPU memory, {g['scope']}: baseline {g['baseline_used_mb']:.0f} MB → peak {g['peak_used_mb']:.0f} MB "
            f"(+{g['delta_mb']:.0f} MB) of {g['total_mb']:.0f} MB"
            + (f" · this process: {g['process_peak_mb']:.0f} MB" if g.get("process_peak_mb") else "")
        )
    else:
        out.append(f"- GPU memory: unavailable ({res.get('gpu_error')})")
    out += [f"- RAM peak (this process, {res['ram_peak_source']}): {res['ram_peak_mb']:.0f} MB", ""]

    s = r["summary"]
    for title_, key in (("By category", "by_category"), ("By tag", "by_tag")):
        if s.get(key):
            out += [f"## {title_}", "", HEADER]
            out += [_row(k, v) for k, v in s[key].items()]
            out.append("")

    o = s["overall"]
    out += [
        "## Pipeline behaviour",
        "",
        f"- Routes: {o['routes']} · cleanup status: {o['cleanup_status'] or '–'}",
        f"- Clips with speech that produced no text: {o['empty_output_on_speech']}",
        "",
    ]

    scored = [u for u in r["utterances"] if u["wer_final"][1] > 0 and u["wer_final"][0] > 0]
    worst = sorted(scored, key=lambda u: (-u["wer_final"][0] / u["wer_final"][1], u["id"]))[:10]
    if worst:
        out += [
            "## Worst 10 clips with errors (final WER)",
            "",
            "| Clip | WER % | Reference (normalized) | Output (normalized) |",
            "|---|---:|---|---|",
        ]
        for u in worst:
            e, t = u["wer_final"]
            out.append(
                f"| `{u['id']}` | {e / t * 100:.0f} | {_cell(u['ref_norm'])} | {_cell(u['final_norm'])} |"
            )
        out.append("")

    out += ["## Provenance", "", "| | |", "|---|---|"]
    for m in r["manifests"]:
        out.append(f"| Manifest | `{m['path']}` ({m['n_used']} clips, sha256 `{m['sha256'][:16]}`) |")
    out += [
        f"| Normalizer | {p['normalizer']['repo']} @ `{p['normalizer']['commit'][:12]}` |",
        f"| Versions | {', '.join(f'{k} {v}' for k, v in p['versions'].items() if v)} |",
        f"| Providers | {', '.join(asr.get('available_providers') or [])} |",
        f"| OS | {p['os']} |",
        f"| Command | `{_cmd(r['args'])}` |",
        "",
    ]
    return "\n".join(out)


def _cell(s: str, n: int = 120) -> str:
    s = s.replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def _cmd(args: dict[str, Any]) -> str:
    parts = ["python -m evals.run_eval"]
    for k, v in args.items():
        if v in (None, False, []) or k in ("out",):
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            parts.append(flag)
        elif isinstance(v, list):
            parts += [f"{flag} {x}" for x in v]
        else:
            parts.append(f"{flag} {v}")
    return " ".join(parts)
