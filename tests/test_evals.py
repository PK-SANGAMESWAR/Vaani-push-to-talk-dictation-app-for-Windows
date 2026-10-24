"""Eval harness tests: normalizer, WER/CER, CIs, percentiles, manifests, provenance (no GPU/network)."""

from __future__ import annotations

import numpy as np
import pytest

from evals import metrics, report, run_eval, sysinfo
from evals.datasets import common
from vaani.config import Config

# ---------- normalizer ----------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", "hello world"),
        ("HE WON'T GO", "he will not go"),
        ("Um, I think so.", "i think so"),
        ("It costs twenty five dollars", "it costs $25"),
        ("The colour of the aeroplane", "the color of the airplane"),
        ("Mr. Smith", "mister smith"),
    ],
)
def test_normalizer_matches_leaderboard_behaviour(text, expected):
    assert metrics.normalize(text) == expected


def test_normalize_collapses_whitespace_and_handles_empty():
    assert metrics.normalize("  red   green \n blue ") == "red green blue"
    assert metrics.normalize("") == ""


# ---------- edit counts ----------


def test_word_edits_counts_sub_del_ins():
    assert metrics.word_edits("a b c d", "a x c d") == metrics.Edits(1, 4)  # substitution
    assert metrics.word_edits("a b c d", "a c d") == metrics.Edits(1, 4)  # deletion
    assert metrics.word_edits("a b c d", "a b c d e f") == metrics.Edits(2, 4)  # insertions
    assert metrics.word_edits("a b", "a b") == metrics.Edits(0, 2)


def test_word_edits_empty_sides():
    assert metrics.word_edits("a b c", "") == metrics.Edits(3, 3)
    assert metrics.word_edits("", "a b") == metrics.Edits(2, 0)
    assert metrics.word_edits("", "") == metrics.Edits(0, 0)


def test_char_edits_count_spaces():
    assert metrics.char_edits("ab cd", "ab cd") == metrics.Edits(0, 5)
    assert metrics.char_edits("ab cd", "abcd") == metrics.Edits(1, 5)  # dropped space
    assert metrics.char_edits("cat", "cut") == metrics.Edits(1, 3)


def test_corpus_rate_is_pooled_not_averaged():
    edits = [metrics.Edits(1, 1), metrics.Edits(0, 9)]
    assert metrics.corpus_rate(edits) == pytest.approx(0.1)  # mean of per-utt rates would be 0.5
    assert metrics.corpus_rate([metrics.Edits(3, 0)]) is None


# ---------- bootstrap + percentiles ----------


def test_bootstrap_ci_is_seeded_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    edits = [metrics.Edits(int(e), 20) for e in rng.integers(0, 4, size=300)]
    a, b = metrics.bootstrap_ci(edits), metrics.bootstrap_ci(edits)
    assert a == b  # deterministic
    rate = metrics.corpus_rate(edits)
    assert a[0] < rate < a[1] and a[1] - a[0] < 0.05
    assert metrics.bootstrap_ci(edits, seed=2) != a


def test_bootstrap_ci_zero_variance_and_empty():
    assert metrics.bootstrap_ci([metrics.Edits(0, 5)] * 10) == (0.0, 0.0)
    assert metrics.bootstrap_ci([]) is None
    assert metrics.rate_with_ci([])["value"] is None


def test_latency_stats():
    s = metrics.latency_stats(list(range(1, 101)))
    assert s["n"] == 100 and s["p50"] == pytest.approx(50.5) and s["p95"] == pytest.approx(95.05)
    assert metrics.latency_stats([])["p50"] is None


# ---------- manifests + subsets ----------


def test_subset_is_deterministic_and_order_independent():
    ids = [f"u{i:04d}" for i in range(1000)]
    a = common.select_subset(ids, 50, 0)
    assert a == common.select_subset(list(reversed(ids)), 50, 0) and a == sorted(a) and len(set(a)) == 50
    assert common.select_subset(ids, 50, 1) != a
    assert common.select_subset(ids[:10], 50, 0) == ids[:10]


def test_manifest_round_trip_is_byte_stable(tmp_path):
    clips = [
        common.Clip("b", "ds", "ds/b.wav", 1.5, "Hello there", "ds", tags={"l1": "Tamil"}),
        common.Clip("a", "ds", "ds/a.wav", 2.0, "", "silence", speaker="s1", ref_clean=""),
    ]
    p1, p2 = tmp_path / "m1.jsonl", tmp_path / "m2.jsonl"
    common.write_manifest(p1, {"seed": 0, "n": 2}, clips)
    meta, back = common.read_manifest(p1)
    common.write_manifest(p2, meta, list(reversed(back)))
    assert p1.read_bytes() == p2.read_bytes()
    assert [c.id for c in back] == ["a", "b"] and back[1].tags == {"l1": "Tamil"}
    assert b"\r\n" not in p1.read_bytes()


def test_manifest_names():
    assert common.manifest_path("fleurs-en").name == "fleurs-en.n200.s0.jsonl"
    assert common.manifest_path("fleurs-en", full=True).name == "fleurs-en.full.jsonl"


# ---------- provenance ----------


def test_config_hash_tracks_values_not_path(tmp_path):
    a, b = Config(path=tmp_path / "a.toml"), Config(path=tmp_path / "b.toml")
    assert sysinfo.config_hash(a) == sysinfo.config_hash(b)
    b.llm_enabled = not b.llm_enabled
    assert sysinfo.config_hash(a) != sysinfo.config_hash(b)
    c = Config(replacements={"x": "y", "a": "b"})
    d = Config(replacements={"a": "b", "x": "y"})
    assert sysinfo.config_hash(c) == sysinfo.config_hash(d)


def test_rel_path_never_leaks_absolute_paths(tmp_path):
    assert sysinfo.rel_path(sysinfo.ROOT / "config.toml") == "config.toml"
    assert sysinfo.rel_path(tmp_path / "x" / "cfg.toml") == "cfg.toml"


# ---------- scoring + report on a synthetic run ----------


def _utt(uid, ref, hyp, dataset="ds", route="replace", **tags):
    clip = common.Clip(uid, dataset, f"{uid}.wav", 2.0, ref, dataset, tags=tags)
    return {
        "id": uid, "dataset": dataset, "category": dataset, "tags": tags, "duration_s": 2.0, "ref": ref,
        "raw": hyp, "final": hyp, "route": route, "cleanup_status": None,
        "asr_ms": 50.0, "cleanup_ms": 0.0, "total_ms": 51.0,
    } | run_eval._score(clip, hyp, hyp)  # fmt: skip


def test_summarize_and_render():
    utts = [
        _utt("1", "the cat sat", "the cat sat", l1="Tamil"),
        _utt("2", "on the mat", "on a mat", l1="Hindi"),
        _utt("3", "hello world", "", dataset="other", route="empty"),
        _utt("4", "", "", dataset="other", route="empty"),
    ]
    s = run_eval.summarize(utts)
    assert s["overall"]["n"] == 4 and s["overall"]["n_scored"] == 3
    assert s["by_dataset"]["ds"]["wer_raw"]["value"] == pytest.approx(1 / 6)
    assert s["by_dataset"]["other"]["wer_raw"]["value"] == 1.0
    assert s["overall"]["empty_output_on_speech"] == 1 and s["overall"]["empty_ref"]["n"] == 1
    assert set(s["by_tag"]) == {"ds/l1=Hindi", "ds/l1=Tamil"}
    assert s["overall"]["rtfx"] == pytest.approx(8.0 / 0.2)

    result = {
        "label": "t", "created_utc": "2026-01-01T00:00:00+00:00", "diagnostic": {"no_vad": True},
        "args": {"label": "t", "no_vad": True, "manifest": []},
        "settings": {"use_vad": False, "cleanup": False, "bootstrap": {"resamples": 1000}},
        "provenance": {
            "git": {"commit": "abc", "branch": "main", "dirty": True},
            "asr": {"model": "m", "device": "cuda", "quantization": "fp32", "available_providers": []},
            "cleanup": None, "hardware": {"gpu": {"name": "G", "driver": "1"}, "cpu": "C", "ram_gb": 8},
            "config": {"sha256": "0" * 64}, "versions": {"onnx-asr": "1"},
            "normalizer": {"repo": "r", "commit": "c" * 40}, "os": "w",
        },
        "manifests": [], "cold_start_ms": {"model_load": 1, "warmup": 2, "load_plus_warmup": 3, "llm_prewarm": None},
        "resources": {"gpu": None, "gpu_error": "none", "ram_peak_mb": 1.0, "ram_peak_source": "sampled_rss"},
        "summary": s, "published_reference": None, "utterances": utts,
    }  # fmt: skip
    md = report.render(result)
    assert "DIRTY TREE" in md and "DIAGNOSTIC RUN" in md and "| ds | 2 |" in md
    assert "Worst 10 clips" in md and "`2`" in md and "`1`" not in md


# ---------- phase 2: negatives and dictation formatting ----------


def test_strict_edits_count_case_punctuation_and_digits():
    assert metrics.strict_tokens("Hi, Priya. It's 10:30!") == [
        "Hi",
        ",",
        "Priya",
        ".",
        "It's",
        "10",
        ":",
        "30",
        "!",
    ]
    assert metrics.strict_edits("Hi, Priya.", "Hi, Priya.") == metrics.Edits(0, 4)
    assert metrics.strict_edits("Hi, Priya.", "hi priya") == metrics.Edits(4, 4)  # 2 case, comma, period
    assert metrics.strict_edits("It costs $5.", "It costs 5.").errors == 1


def test_negatives_are_deterministic_and_quiet():
    from evals.datasets import negatives

    for kind in negatives.KINDS:
        a, b = negatives.synth(kind, 1.5, 0), negatives.synth(kind, 1.5, 0)
        assert a.dtype == np.float32 and len(a) == 24_000 and np.array_equal(a, b)
        assert np.abs(a).max() <= 1.0
    assert not np.any(negatives.synth("silence-digital", 1.0, 0))
    assert not np.array_equal(negatives.synth("noise-room", 1.0, 0), negatives.synth("noise-room", 1.0, 1))
    rms = float(np.sqrt(np.mean(negatives.synth("noise-fan", 2.0, 0) ** 2)))
    assert 20 * np.log10(rms) == pytest.approx(-45.0, abs=0.5)


def test_fixed_datasets_have_single_manifest_name():
    assert common.manifest_path("negatives", 5, 3).name == "negatives.fixed.jsonl"
    assert common.manifest_path("fleurs-en", 5, 3).name == "fleurs-en.n5.s3.jsonl"


def test_dictation_script_is_well_formed():
    from evals.datasets import dictation

    rows = dictation.load_script()
    assert len({r["id"] for r in rows}) == len(rows) >= 20
    assert all(r["ref"] and r["ref_clean"] and r["ref"] == r["ref"].lower() for r in rows)


def test_silence_false_positives_and_strict_score_in_report():
    clip = common.Clip("n1", "negatives", "n1.wav", 2.0, "", "noise-fan")
    leak = _utt("n1", "", "thank you", dataset="negatives")
    assert leak["wer_raw"][1] == 0
    dclip = common.Clip("d1", "dictation", "d1.wav", 2.0, "hi priya", "email", ref_clean="Hi, Priya.")
    sc = run_eval._score(dclip, "hi priya", "Hi, Priya.")
    assert sc["strict_raw"] == [4, 4] and sc["strict_final"] == [0, 4]
    assert clip.ref == "" and "strict_final" not in run_eval._score(clip, "", "")

    s = run_eval.summarize([leak, _utt("n2", "", "", dataset="negatives")])
    assert s["overall"]["empty_ref"] == {"n": 2, "nonempty_raw": 1, "nonempty_output": 1}


def test_content_wer_rewards_removed_fillers_that_verbatim_wer_penalises():
    clip = common.Clip("d2", "dictation", "d2.wav", 2.0, "it is you know done", "fillers", ref_clean="It is done.")
    sc = run_eval._score(clip, "it is you know done", "It is done.")
    assert sc["wer_final"] == [2, 5]  # verbatim reference expects the filler
    assert sc["clean_final"] == [0, 3] and sc["clean_raw"] == [2, 3]
