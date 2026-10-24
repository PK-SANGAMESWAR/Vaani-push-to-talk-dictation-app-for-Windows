"""E2E harness pieces that need no Windows UI, mic, or Vaani process."""

from __future__ import annotations

import tomllib

from evals.datasets import common
from evals.e2e import run_e2e
from evals.run_eval import _score
from vaani.config import load_config


def test_write_config_applies_overrides_and_round_trips(tmp_path):
    src = tmp_path / "in.toml"
    src.write_text(
        '[hotkey]\nkey = "caps lock"\nmode = "hold"\n[audio]\nsounds = true\nmin_duration_s = 0.3\n'
        '[replacements]\n"pie torch" = "PyTorch"\n[snippets]\n' + r'"sig" = "Best,\nMe"' + "\n",
        encoding="utf-8",
    )
    dst = tmp_path / "out.toml"
    run_e2e.write_config(
        src, dst, {"hotkey": {"key": "f9"}, "audio": {"input_device": "CABLE Output", "sounds": False},
                   "cleanup": {"llm_enabled": None}},
    )  # fmt: skip
    d = tomllib.loads(dst.read_text(encoding="utf-8"))
    assert d["hotkey"] == {"key": "f9", "mode": "hold"}
    assert d["audio"] == {"sounds": False, "min_duration_s": 0.3, "input_device": "CABLE Output"}
    assert d["replacements"] == {"pie torch": "PyTorch"} and d["snippets"]["sig"] == "Best,\nMe"
    assert "llm_enabled" not in d["cleanup"]  # None = keep config's value
    cfg = load_config(dst)
    assert cfg.hotkey == "f9" and cfg.input_device == "CABLE Output" and not cfg.sounds


def test_log_events_cover_final_streamed_and_silent_outcomes():
    log = (
        "2026-10-24 10:00:00,000 INFO    vaani.app: Ready on CUDA. Hold f9 to dictate.\n"
        "2026-10-24 10:00:01,000 INFO    vaani.streaming: phrase 1: audio 2.0s | asr 40 ms\n"
        "2026-10-24 10:00:02,000 INFO    vaani.app: audio 3.1s | asr 52 ms | total 60 ms | 9 words\n"
        "2026-10-24 10:00:03,000 INFO    vaani.app: no speech detected (1.5s audio)\n"
        "2026-10-24 10:00:04,000 INFO    vaani.app: ignored 0.10s tap\n"
        "2026-10-24 10:00:05,000 INFO    vaani.cleanup: cleanup 300 ms\n"
    )
    kinds = [m.group("kind") for m in run_e2e.EVENT.finditer(log)]
    assert kinds == ["phrase 1:", "audio 3.1s | asr", "no speech detected", "ignored 0.10s tap"]
    assert run_e2e.READY.search(log).group(1) == "CUDA"


def _row(cid, ref, pasted, ms, handled=True, dur=3.0, ref_clean=None):
    clip = common.Clip(cid, "d", f"{cid}.wav", dur, ref, "c", ref_clean=ref_clean)
    return {"id": cid, "ref": ref, "duration_s": dur, "pasted": pasted, "release_to_paste_ms": ms,
            "handled": handled, "events": [], "read_ms_p50": 8.0} | _score(clip, pasted, pasted)  # fmt: skip


def test_score_rows_counts_latency_drops_and_silence():
    rows = [
        _row("a", "send the report", "Send the report.", 300.0, ref_clean="Send the report."),
        _row("b", "hello world", "", None, handled=False),
        _row("c", "", "thank you", 200.0),
        _row("d", "a long one here", "a long one here", 900.0, dur=20.0),
    ]
    s = run_e2e.score_rows(rows)
    assert s["dropped_hotkeys"] == 1 and s["speech_with_nothing_pasted"] == 1
    assert s["silence_pasted"] == {"k": 1, "n": 1}
    assert s["release_to_paste_ms"]["n"] == 2 and s["release_to_paste_ms"]["max"] == 900.0
    assert set(s["latency_by_length"]) == {"<5 s", ">=15 s"} and s["strict_wer"]["value"] == 0.0

    result = {"label": "t", "created_utc": "x", "app_device": "cuda", "provenance": {
        "git": {"dirty": True, "commit": "abc"}, "code": {"vaani_diff_sha256": "f" * 64}},
        "settings": {"mic": "m", "play": "p", "hotkey": "f9", "cleanup": False, "streaming": True, "lead_ms": 300},
        "accuracy": {"notepad": {"summary": s, "clips": rows}},
        "soak": {"stats": {"done": 3, "wall_s": 9, "crashed": False, "dropped": 0, "speech_no_text": 0,
                 "silence_pasted": 0}, "growth_after_warmup": None,
                 "memory_samples": [{"i": 0, "rss_mb": 900.0, "private_mb": 1000.0, "handles": 400, "threads": 20}]},
    }  # fmt: skip
    md = run_e2e.render(result)
    assert "DIRTY TREE" in md and "## notepad" in md and "Dropped hotkeys | 1" in md and "## Soak" in md
    assert "`b`" in md and "`c`" in md and "`a`" not in md
