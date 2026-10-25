# Changelog

All notable changes to Vaani. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). There are
no tagged releases yet, so everything is under Unreleased; entries are grouped by the project's phases.

## [Unreleased]

### Added
- Streaming phrases while the key is held, auto-gain for quiet voices, and a release tail so the last word
  is not clipped (`[streaming]`, `auto_gain`, `release_tail_ms` in `config.toml`).
- `Pipeline` (`vaani/pipeline.py`): the audio-to-text path shared by the app and the eval suite.
- Eval suite (`evals/`): LibriSpeech, FLEURS and Svarah accuracy, synthetic silence/noise, dictation
  formatting, injection and vocabulary safety checks, baseline comparison (Parakeet GPU/CPU, Whisper,
  cleanup models), end-to-end Notepad/browser test through a virtual microphone, and a 1000-dictation soak.
- Regression gates (`evals/thresholds.toml`, `run_eval --gates PROFILE`): WER +0.5 points at most per
  dataset, no text from silence/noise, at most 1% injection answers, p95 latency 600 ms without cleanup.
  `run_eval` exits non-zero on regression.
- `run_eval --block-network` and `evals/netguard.py`: refuse every non-loopback connection and record attempts.
- GitHub Actions: ruff, pytest, and a CPU int8 mini-eval with the network blocked and gates applied.
- `requirements.lock` (hash-pinned), `requirements-ci.txt`, NOTICE, SECURITY, CONTRIBUTING, CODE_OF_CONDUCT,
  issue and pull-request templates, README benchmark table generated from committed results
  (`python -m evals.bench_table`).
- README: tray-menu screenshot and a troubleshooting note for a wrong or silent microphone.

### Changed
- `llm_url` must point at loopback (127.0.0.0/8, ::1, localhost); anything else is rejected when the config loads.
- Cleanup prompt and plausibility guard hardened against dictated questions and commands.
- Mic names can be matched by a part of the name.

### Removed
- Stale `uv.lock` (an unrelated `uv init` artifact); `requirements.lock` replaces it.
