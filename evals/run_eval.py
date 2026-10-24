"""Run the shipped Vaani pipeline over eval manifests and write a JSON + Markdown report.

    python -m evals.run_eval                                   # 4 datasets x 200 clips, config.toml
    python -m evals.run_eval --cleanup off --label gpu-fp32
    python -m evals.run_eval --datasets librispeech-clean --full --no-vad --label validate-novad
    python -m evals.run_eval --manifest my_clips.jsonl --device cpu

Audio is decoded before the timer starts; per-clip latency covers `Pipeline.process` only (ASR + text
processing + cleanup), the same span the app logs as "total". It does not include paste.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vaani.config import load_config
from vaani.pipeline import Pipeline

from . import metrics, report, safety, sysinfo
from .datasets.common import (
    DEFAULT_N,
    DEFAULT_SEED,
    SAMPLE_RATE,
    Clip,
    load_audio,
    manifest_path,
    read_manifest,
)
from .resources import PeakSampler

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASETS = ["librispeech-clean", "librispeech-other", "fleurs-en", "svarah"]
SCHEMA_VERSION = 2  # 2: adds `safety`


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m evals.run_eval", description=__doc__.splitlines()[0])
    p.add_argument(
        "--datasets", help=f"comma-separated (default: {','.join(DEFAULT_DATASETS)} unless --manifest)"
    )
    p.add_argument(
        "--manifest", action="append", default=[], help="extra/explicit manifest path (repeatable)"
    )
    p.add_argument("--full", action="store_true", help="use the full-split manifests")
    p.add_argument("--n", type=int, default=DEFAULT_N, help="subset size of the manifests to use")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="subset seed of the manifests to use")
    p.add_argument("--limit", type=int, help="only the first N clips of each manifest (smoke tests)")
    p.add_argument("--config", default=str(ROOT / "config.toml"))
    p.add_argument("--device", choices=["auto", "cuda", "cpu"])
    p.add_argument("--quantization", choices=["auto", "none", "int8"])
    p.add_argument(
        "--asr",
        choices=["config", "whisper-large-v3-turbo"],
        default="config",
        help="ASR backend: config = Parakeet as in config.toml (shipped); whisper-* = eval-only baseline",
    )
    p.add_argument("--cleanup", choices=["config", "on", "off"], default="config")
    p.add_argument("--llm-model", help="Ollama model for cleanup (default: as in config)")
    p.add_argument(
        "--gain-db",
        type=float,
        default=0.0,
        help="scale the input audio by this many dB (e.g. -30 = low voice)",
    )
    p.add_argument("--no-vad", action="store_true", help="DIAGNOSTIC: skip VAD (not the shipped pipeline)")
    p.add_argument("--label", default="run")
    p.add_argument("--out", default=str(ROOT / "evals" / "results"))
    p.add_argument(
        "--published-wer", type=float, help="published WER (%%) to compare the overall raw WER with"
    )
    p.add_argument("--published-source", help="URL the published WER was taken from")
    p.add_argument("--published-note", help="free text: model id, date fetched, leaderboard column")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _manifests(args: argparse.Namespace) -> list[Path]:
    paths = [Path(m) for m in args.manifest]
    names = args.datasets.split(",") if args.datasets else ([] if args.manifest else DEFAULT_DATASETS)
    paths += [manifest_path(n.strip(), args.n, args.seed, args.full) for n in names if n.strip()]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            "missing manifest(s): " + ", ".join(map(str, missing)) + "\nrun: python -m evals.datasets.prepare"
        )
    return paths


def _score(clip: Clip, raw: str, final: str) -> dict[str, Any]:
    ref_n = metrics.normalize(clip.ref)
    raw_n, final_n = metrics.normalize(raw), metrics.normalize(final)
    w_raw, w_fin = metrics.word_edits(ref_n, raw_n), metrics.word_edits(ref_n, final_n)
    c_raw, c_fin = metrics.char_edits(ref_n, raw_n), metrics.char_edits(ref_n, final_n)
    strict = {}
    if clip.ref_clean is not None:  # dictation sets: formatting-sensitive score vs the intended text
        s_raw, s_fin = metrics.strict_edits(clip.ref_clean, raw), metrics.strict_edits(clip.ref_clean, final)
        strict = {"ref_clean": clip.ref_clean, "strict_raw": [s_raw.errors, s_raw.total]}
        strict["strict_final"] = [s_fin.errors, s_fin.total]
        # content accuracy vs the intended text: normalized, so casing/punctuation don't count but
        # dropped fillers and resolved self-corrections are rewarded instead of penalised as in `wer_*`.
        rc = metrics.normalize(clip.ref_clean)
        c_raw_w, c_fin_w = metrics.word_edits(rc, raw_n), metrics.word_edits(rc, final_n)
        strict["clean_raw"], strict["clean_final"] = (
            [c_raw_w.errors, c_raw_w.total],
            [c_fin_w.errors, c_fin_w.total],
        )
    return strict | {
        "ref_norm": ref_n,
        "raw_norm": raw_n,
        "final_norm": final_n,
        "wer_raw": [w_raw.errors, w_raw.total],
        "wer_final": [w_fin.errors, w_fin.total],
        "cer_raw": [c_raw.errors, c_raw.total],
        "cer_final": [c_fin.errors, c_fin.total],
    }


def group_stats(utts: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [u for u in utts if u["wer_raw"][1] > 0]  # empty normalized ref: not part of WER

    def edits(key: str) -> list[metrics.Edits]:
        return [metrics.Edits(*u[key]) for u in scored]

    audio_s = sum(u["duration_s"] for u in utts)
    asr_s = sum(u["asr_ms"] for u in utts) / 1000
    routes: dict[str, int] = defaultdict(int)
    statuses: dict[str, int] = defaultdict(int)
    for u in utts:
        routes[u["route"]] += 1
        if u["cleanup_status"]:
            statuses[u["cleanup_status"]] += 1
    empty_ref = [u for u in utts if u["wer_raw"][1] == 0]
    strict = [u for u in utts if "strict_final" in u]
    return {
        "n": len(utts),
        "n_scored": len(scored),
        "audio_hours": round(audio_s / 3600, 4),
        "wer_raw": metrics.rate_with_ci(edits("wer_raw")),
        "wer_final": metrics.rate_with_ci(edits("wer_final")),
        "cer_raw": metrics.rate_with_ci(edits("cer_raw")),
        "cer_final": metrics.rate_with_ci(edits("cer_final")),
        "latency_ms": {
            "asr": metrics.latency_stats([u["asr_ms"] for u in utts]),
            "cleanup": metrics.latency_stats([u["cleanup_ms"] for u in utts if u["route"] == "cleanup"]),
            "total": metrics.latency_stats([u["total_ms"] for u in utts]),
        },
        "rtfx": round(audio_s / asr_s, 1) if asr_s > 0 else None,
        "routes": dict(routes),
        "cleanup_status": dict(statuses),
        "empty_output_on_speech": sum(1 for u in scored if not u["raw"]),
        # Silence/noise clips (empty reference): any text at all is a false positive.
        "empty_ref": {
            "n": len(empty_ref),
            "nonempty_raw": sum(1 for u in empty_ref if u["raw"]),
            "nonempty_output": sum(1 for u in empty_ref if u["final"]),
        },
        "strict_raw": metrics.rate_with_ci([metrics.Edits(*u["strict_raw"]) for u in strict]),
        "strict_final": metrics.rate_with_ci([metrics.Edits(*u["strict_final"]) for u in strict]),
        "clean_raw": metrics.rate_with_ci([metrics.Edits(*u["clean_raw"]) for u in strict]),
        "clean_final": metrics.rate_with_ci([metrics.Edits(*u["clean_final"]) for u in strict]),
        "n_strict": len(strict),
    }


def summarize(utts: list[dict[str, Any]]) -> dict[str, Any]:
    by: dict[str, dict[str, list]] = {"dataset": defaultdict(list), "category": defaultdict(list)}
    by_tag: dict[str, list] = defaultdict(list)
    for u in utts:
        by["dataset"][u["dataset"]].append(u)
        by["category"][u["category"]].append(u)
        for k, v in u["tags"].items():
            if k in ("l1", "gender"):
                by_tag[f"{u['dataset']}/{k}={v}"].append(u)
    out: dict[str, Any] = {"overall": group_stats(utts)}
    out["by_dataset"] = {k: group_stats(v) for k, v in sorted(by["dataset"].items())}
    cats = {k: v for k, v in by["category"].items() if k not in by["dataset"]}
    out["by_category"] = {k: group_stats(v) for k, v in sorted(cats.items())}
    out["by_tag"] = {k: group_stats(v) for k, v in sorted(by_tag.items())}
    return out


def _load_pipeline(cfg, args: argparse.Namespace) -> Pipeline:
    if args.asr == "config":
        return Pipeline.load(cfg, use_vad=not args.no_vad)
    from vaani.pipeline import asr_device, make_cleaner
    from vaani.textproc import TextProcessor

    from .whisper_backend import WhisperTranscriber

    cfg.asr_model = args.asr  # so provenance and the config hash name the model that ran
    text = TextProcessor(cfg.replacements, cfg.snippets)
    stt = WhisperTranscriber(
        args.asr,
        asr_device(cfg),
        cfg.quantization,
        cfg.vad_threshold,
        cfg.vad_min_silence_ms,
        cfg.vad_speech_pad_ms,
    )
    return Pipeline(cfg, stt, text, make_cleaner(cfg, text.vocabulary), use_vad=not args.no_vad)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    paths = _manifests(args)
    code_at_start = sysinfo.code_state()  # before any vaani module is imported by the pipeline
    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.quantization:
        cfg.quantization = args.quantization
    if args.cleanup != "config":
        cfg.llm_enabled = args.cleanup == "on"
    if args.llm_model:
        cfg.llm_model = args.llm_model

    manifests = []
    clips: list[Clip] = []
    for p in paths:
        meta, cs = read_manifest(p)
        cs = cs[: args.limit] if args.limit else cs
        rel = sysinfo.rel_path(p)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        manifests.append({"path": rel, "sha256": sha, "n_used": len(cs), "meta": meta})
        clips += cs
    missing_audio = [c.audio for c in clips if not c.path.exists()]
    if missing_audio:
        raise SystemExit(f"{len(missing_audio)} audio files missing (e.g. {missing_audio[0]}); run prepare")

    sampler = PeakSampler().start()  # baseline is taken before any model loads
    t0 = time.perf_counter()
    pipeline = _load_pipeline(cfg, args)
    load_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    pipeline.stt.warmup(SAMPLE_RATE)
    warmup_ms = (time.perf_counter() - t1) * 1000
    llm_prewarm_ms = None
    if cfg.llm_enabled:
        t2 = time.perf_counter()
        pipeline.cleaner.prewarm()
        llm_prewarm_ms = (time.perf_counter() - t2) * 1000

    print(f"{len(clips)} clips on {pipeline.stt.device}/{pipeline.stt.quantization}", file=sys.stderr)
    utts = []
    t_run = time.perf_counter()
    for i, clip in enumerate(clips, 1):
        audio = load_audio(clip.path)
        if args.gain_db:
            audio = audio * np.float32(10 ** (args.gain_db / 20))
        r = pipeline.process(audio, SAMPLE_RATE)
        utts.append(
            {
                "id": clip.id,
                "dataset": clip.dataset,
                "category": clip.category,
                "tags": clip.tags,
                "duration_s": clip.duration_s,
                "ref": clip.ref,
                "speaker": clip.speaker,
                "mic": clip.mic,
                "raw": r.raw,
                "final": r.text,
                "route": r.route,
                "cleanup_status": r.cleanup_status,
                "asr_ms": round(r.asr_ms, 2),
                "cleanup_ms": round(r.cleanup_ms, 2),
                "total_ms": round(r.total_ms, 2),
            }
            | _score(clip, r.raw, r.text)
        )
        if i % 50 == 0 or i == len(clips):
            print(f"  {i}/{len(clips)}", file=sys.stderr)
    run_s = time.perf_counter() - t_run
    resources = sampler.stop()

    created = datetime.now(UTC)
    result = {
        "schema": SCHEMA_VERSION,
        "label": args.label,
        "created_utc": created.isoformat(timespec="seconds"),
        "diagnostic": {"no_vad": args.no_vad},
        "args": vars(args) | {"config": sysinfo.rel_path(args.config), "out": sysinfo.rel_path(args.out)},
        "settings": {
            "cleanup": cfg.llm_enabled,
            "use_vad": not args.no_vad,
            "bootstrap": {
                "resamples": metrics.BOOTSTRAP_RESAMPLES,
                "seed": metrics.BOOTSTRAP_SEED,
                "ci": 0.95,
            },
            "percentile_method": metrics.PERCENTILE_METHOD,
            "wer": "corpus-level, EnglishTextNormalizer on ref and hyp; empty normalized refs excluded",
            "cer": "corpus-level on normalized text, spaces counted",
        },
        "provenance": sysinfo.collect(cfg, pipeline.stt, cfg.llm_enabled)
        | {"code": code_at_start | {"changed_during_run": sysinfo.code_state() != code_at_start}},
        "manifests": manifests,
        "cold_start_ms": {
            "model_load": round(load_ms, 1),
            "warmup": round(warmup_ms, 1),
            "load_plus_warmup": round(load_ms + warmup_ms, 1),
            "llm_prewarm": round(llm_prewarm_ms, 1) if llm_prewarm_ms is not None else None,
        },
        "run_wall_s": round(run_s, 1),
        "resources": resources,
        "summary": summarize(utts),
        "safety": safety.summarize(utts, pipeline.text.vocabulary),
        "published_reference": None,
        "utterances": utts,
    }
    if args.published_wer is not None:
        ours = result["summary"]["overall"]["wer_raw"]
        lo, hi = (x * 100 for x in ours["ci95"])
        result["published_reference"] = {
            "wer_percent": args.published_wer,
            "source": args.published_source,
            "note": args.published_note,
            "ours_wer_raw_percent": ours["value"] * 100,
            "ours_ci95_percent": [lo, hi],
            "delta_abs_percent": ours["value"] * 100 - args.published_wer,
            "published_within_our_ci": lo <= args.published_wer <= hi,
        }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = (result["provenance"]["git"]["commit"] or "nogit")[:7]
    stem = f"{created:%Y%m%dT%H%M%SZ}_{sha}_{args.label}"
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    md_path.write_text(report.render(result), encoding="utf-8", newline="\n")
    print(report.render_summary_table(result), file=sys.stderr)
    print(f"\nwrote {json_path}\nwrote {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
