"""Repository-level checks: README numbers match committed results, CI pins match, notices exist."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from evals import bench_table

ROOT = Path(__file__).resolve().parent.parent


def _pins(name: str) -> dict[str, str]:
    out = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s#]+)", line)
        if m:
            out[m.group(1).lower()] = m.group(2)
    return out


def test_readme_benchmark_block_matches_the_committed_result_file():
    assert bench_table.main(["--check"]) == 0


def test_bench_table_refuses_dirty_results(monkeypatch):
    monkeypatch.setattr(
        bench_table, "GPU_RUN", "evals/results/20261024T150526Z_ac31804_p4-parakeet-gpu-fp32.json"
    )
    with pytest.raises(ValueError, match="dirty"):
        bench_table.render()


def test_ci_requirements_share_exact_pins_with_eval_requirements():
    ci, ev = _pins("requirements-ci.txt"), _pins("requirements-eval.txt")
    shared = set(ci) & set(ev)
    assert {"jiwer", "regex", "soundfile", "pyarrow", "huggingface-hub", "psutil"} <= shared
    assert {k: (ci[k], ev[k]) for k in shared if ci[k] != ev[k]} == {}


def test_ci_never_installs_the_gpu_runtime():
    lines = (ROOT / "requirements-ci.txt").read_text(encoding="utf-8").splitlines()
    text = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    assert "onnxruntime-gpu" not in text and re.search(r"^onnxruntime>=", text, re.M)


def test_lockfile_pins_every_eval_requirement_with_hashes():
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    pins = _pins("requirements.lock")
    for name, version in _pins("requirements-eval.txt").items():
        assert pins.get(name) == version, name
    assert "--hash=sha256:" in lock and "onnxruntime-gpu" in pins and "onnxruntime" not in pins


def test_open_source_files_exist_and_notice_covers_third_party_assets():
    for f in ("SECURITY.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "CHANGELOG.md", "LICENSE", "NOTICE"):
        assert (ROOT / f).is_file(), f
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for needle in (
        "CC-BY-4.0",
        "Parakeet",
        "Silero",
        "onnx-asr",
        "Ollama",
        "Apache License 2.0",
        "LibriSpeech",
    ):
        assert needle in notice, needle
    for t in ("bug_report.yml", "feature_request.yml"):
        assert (ROOT / ".github" / "ISSUE_TEMPLATE" / t).is_file()
    bug = (ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml").read_text(encoding="utf-8")
    assert "vaani.log" in bug and "driver" in bug.lower() and "GPU" in bug


def test_no_phase6_todos_left():
    for f in ("evals/README.md", "evals/normalizer/README.md"):
        assert "TODO(phase6" not in (ROOT / f).read_text(encoding="utf-8"), f
