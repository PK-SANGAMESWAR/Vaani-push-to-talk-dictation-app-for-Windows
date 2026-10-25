# Contributing to Vaani

Thanks for helping. Read [CLAUDE.md](CLAUDE.md) first: it lists the project's non-negotiable principles
(fully local, latency is the product, models stay resident, silence produces no text, never block the
keyboard hook thread). Changes that break one of them will not be merged.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
```

Exact versions known to work are pinned in `requirements.lock`.

## Before you open a pull request

```powershell
.\.venv\Scripts\python -m ruff check . ; .\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m pytest -q
```

Unit tests must run without GPU, microphone, Ollama, or network. Mock HTTP for `Cleaner`.

If you touch the audio-to-text path (`vaani/pipeline.py`, `stt.py`, `cleanup.py`, `textproc.py`, `audio.py`),
also run the evals (see [evals/README.md](evals/README.md)) and put the gate output in the PR description:

```powershell
.\.venv\Scripts\python -m evals.datasets.prepare librispeech-clean negatives --n 40
.\.venv\Scripts\python -m evals.run_eval --datasets librispeech-clean,negatives --n 40 --device cpu --quantization int8 --cleanup off --block-network --gates ci --label my-change
```

CI runs this on every pull request. If you changed accuracy on purpose, say so, and commit the new result
file together with the `baseline` update in `evals/thresholds.toml`.

## Rules of thumb

- Put audio-to-text logic in `Pipeline`, never in `App`, so the evals measure what ships.
- Heavy imports (`onnx_asr`, `onnxruntime`, `keyboard`, `sounddevice`) stay inside the functions that use them.
- New settings: add to `Config` with a default, to `_SECTIONS` in `config.py`, and document in `config.toml`.
- Never commit model files, recordings, or `evals/data/`.
- Numbers in docs must come from a committed file in `evals/results/`, from a clean commit.
- Don't add telemetry, update checks, or cloud calls, even optional ones.

## Reporting bugs

Use the bug-report template; it asks for `%LOCALAPPDATA%\Vaani\vaani.log`, your GPU and driver version.
Transcripts are not logged at INFO level, but check the log for anything private before attaching it.
