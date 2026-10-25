# CLAUDE.md: Vaani

Vaani (വാണി, "speech") is a **fully local, push-to-talk dictation app for Windows**: hold a key, speak, release, and clean text is pasted into whatever app has focus. It is an offline, open-source take on Wispr Flow / Suki (Suki is macOS-only). No audio or text ever leaves the machine.

Target machine: Windows 11 laptop, **NVIDIA RTX 4050 Laptop GPU (6 GB VRAM)**, driver 591.66, Python 3.12. With 6 GB, Parakeet fp32 and the Ollama cleanup model share tight VRAM.

## Pipeline

```
hotkey (keyboard hook) → control queue → Recorder (sounddevice, 16 kHz mono float32)
  → jobs queue → worker thread:
      Pipeline.process (vaani/pipeline.py, shared with evals/):
        Silero VAD (CPU) segments speech, drops silence
        → Parakeet TDT 0.6B v3 (onnx-asr, CUDA EP, fp32)
        → snippet match | replacements → optional Ollama cleanup (localhost) → replacements again
      → paste via clipboard + Ctrl+V, restore clipboard (App, not Pipeline)
```

`Pipeline.process(audio, sr)` returns a `Result` (raw, final text, route, cleanup status, timings). It never pastes, logs transcripts, or touches the clipboard. The app and the eval suite both call it, so evals measure the code that ships: put any audio-to-text logic change there, not in `App`.

## File map

| File | Responsibility |
|---|---|
| `vaani/__main__.py` | CLI entry (`--config`, `--file`, `--download`, `-v`), logging, single-instance mutex |
| `vaani/app.py` | Orchestration: threads, queues, hotkey → start/stop, calls `Pipeline`, paste, timing log, history, state |
| `vaani/pipeline.py` | `Pipeline` / `Result`: audio → text (VAD + ASR → snippets/replacements → optional cleanup) with timings; `make_transcriber` / `make_cleaner` factories |
| `vaani/audio.py` | `Recorder`: mic capture, optional always-open stream with pre-roll buffer |
| `vaani/stt.py` | `Transcriber`: provider selection (CUDA → CPU fallback), VAD + Parakeet, warm-up; `transcribe(..., vad=False)` is eval-only |
| `vaani/cleanup.py` | `Cleaner`: Ollama `/api/chat` call, prompt, `is_plausible` output guard; `clean_with_status` → `"ok"` / `"error"` / `"rejected"` |
| `vaani/textproc.py` | `TextProcessor`: custom vocabulary replacements, voice snippets |
| `vaani/output.py` | `insert_text`: clipboard paste or simulated typing |
| `vaani/tray.py` | pystray icon (state colours) + menu |
| `vaani/config.py` | `config.toml` → `Config` dataclass, validation |
| `config.toml` | User settings (hotkey, models, vocabulary, snippets) |
| `tests/` | pytest unit tests (no GPU/mic needed) |
| `evals/` | Accuracy/latency/memory eval suite over public datasets; see `evals/README.md` |
| `evals/datasets/` | Dataset downloaders (`prepare.py`) and committed JSONL manifests (`manifests/`) |
| `evals/run_eval.py` | Runs `Pipeline` over manifests → `evals/results/*.json` + `.md` report |
| `evals/normalizer/` | Vendored Open ASR Leaderboard text normalizer (Apache-2.0, unmodified; ruff-excluded) |
| `evals/thresholds.toml`, `evals/gates.py` | Regression gates and profiles (`gpu`, `cleanup`, `ci`); `run_eval --gates PROFILE` exits 1 on regression |
| `evals/netguard.py` | Blocks non-loopback sockets and records attempts (`run_eval --block-network`, `tests/test_privacy.py`) |
| `evals/bench_table.py` | Generates the README benchmark block from clean-tree result files |
| `requirements-eval.txt` | Exact-pinned eval-only dependencies (not needed to run Vaani) |
| `requirements-ci.txt` | CI install: CPU `onnxruntime`, no CUDA wheels; shared pins must equal `requirements-eval.txt` (tested) |
| `requirements.lock` | Hash-pinned full set for Windows/py3.12 (regenerate with the `uv pip compile` line in its header) |
| `.github/workflows/ci.yml` | ruff + pytest, then CPU int8 mini-eval with network blocked and `--gates ci` |
| `NOTICE`, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md` | Open-source files; update NOTICE when adding a model, dataset or vendored code |

## Commands (PowerShell, from the repo root)

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1   # venv + deps + model download
.\run.bat                                            # run in tray (pythonw, no console)
.\run-debug.bat                                      # run with console + verbose logs
.\.venv\Scripts\python -m vaani --file sample.wav    # transcribe a 16-bit WAV, print latency/RTFx
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check . ; .\.venv\Scripts\python -m ruff format .

# evals (optional)
.\.venv\Scripts\python -m pip install -r requirements-eval.txt
.\.venv\Scripts\python -m evals.datasets.prepare                        # download + write manifests (once)
.\.venv\Scripts\python -m evals.run_eval --cleanup off --label gpu-fp32
.\.venv\Scripts\python -m evals.run_eval --limit 5 --cleanup off        # quick smoke test
```

Logs: `%LOCALAPPDATA%\Vaani\vaani.log` (each dictation logs audio length, ASR ms, total ms).

## Non-negotiable principles

1. **Fully local.** The only network call allowed is to Ollama on `127.0.0.1`. Never add cloud APIs, telemetry, or update checks by default. Model downloads happen once (Hugging Face cache) during setup. Eval datasets are downloaded only by `evals.datasets.prepare`, never by the app.
2. **Latency is the product.** Budget on the target GPU (set for an RTX 4060; to be re-validated on the RTX 4050 in Phase 4 of the eval work): release-to-paste **< 400 ms** for a 10 s clip without cleanup, **< 1.2 s** with cleanup. Measure before and after any change that touches the hot path (use the timing log line or `--file`).
3. **Models load once and stay resident.** Never load a model per dictation. Warm up after loading.
4. **Never block the keyboard hook thread.** Hook callbacks only `ctrl.put(...)`. Windows silently removes low-level hooks that exceed `LowLevelHooksTimeout`, and the hotkey would just stop working.
5. **Silence must produce no text.** That's why we use a transducer model (Parakeet) plus VAD rather than Whisper, which hallucinates on silence. Don't remove the VAD stage.
6. **Fail safe to raw text.** If Ollama is down, slow, or returns something implausible, paste the raw transcript. Never lose a dictation.
7. **Privacy:** never write audio to disk; history is opt-in (`save_history`); never log transcripts at INFO level except via opt-in history.

## Conventions

- Python 3.11+ (uses `tomllib`), type hints everywhere, `from __future__ import annotations`.
- ruff (config in `pyproject.toml`, line length 110). Keep modules small and single-purpose.
- Heavy imports (`onnx_asr`, `onnxruntime`, `keyboard`, `sounddevice`) stay inside functions/classes that need them, so `config`/`textproc`/`cleanup` stay importable and testable on any OS without a GPU.
- New settings: add to `Config` (with default), to `_SECTIONS` in `config.py`, and document in `config.toml`.
- Threads: `main` = tray; `init` = model loading; `control` = start/stop in hotkey order; `worker` = ASR + cleanup + paste; PortAudio callback thread = audio frames only (no allocation-heavy work, no logging per frame).

## Key technical facts

- `onnx-asr` API: `onnx_asr.load_model(name, quantization=None|"int8", providers=[...])`, `.with_vad(onnx_asr.load_vad("silero"), threshold=..., min_silence_duration_ms=..., speech_pad_ms=..., max_speech_duration_s=...)`, `.recognize(np.float32 array, sample_rate=16000)`. With VAD it returns an iterator of segments (`.text`, `.start`, `.end`).
- Parakeet's max input is ~20–30 s; the VAD adapter splits longer speech, so long dictation works.
- On the CUDA EP use **fp32** (int8 ops fall back to CPU and get slower); on CPU use **int8**.
- `onnxruntime-gpu[cuda,cudnn]` (≥1.27) ships CUDA 13 + cuDNN 9 as pip packages; `onnxruntime.preload_dlls()` loads them. Requires a recent NVIDIA driver (R580+). Don't install the CPU `onnxruntime` package alongside it; they conflict.
- Parakeet v3 covers English + 24 European languages, not Malayalam/Hindi. For Indic languages a Whisper backend would be needed (see roadmap).
- `keyboard` lib: `on_press_key` repeats while held (guarded by `_recording`); hold mode supports a single key only; combos need toggle mode. On most laptops the Fn key is invisible to Windows.
- Clipboard restore only preserves text (pyperclip). Pasting into elevated (admin) windows fails unless Vaani also runs elevated (UIPI).

## Testing

- Unit tests (`tests/`) cover config parsing, replacements/snippets, the cleanup plausibility guard, `Pipeline` routing/cleanup status/timings (`test_pipeline.py`, models faked), and the eval harness: normalizer, WER/CER, CIs, manifests, provenance (`test_evals.py`). They must run without GPU, mic, Ollama, or network. Mock HTTP for `Cleaner`.
- Evals (`evals/`) measure accuracy and latency on real models; see `evals/README.md`. Numbers quoted in the README must come from a run on a clean commit (dirty-tree runs are flagged in the report).
- `test_gates.py` (gate logic on synthetic results), `test_privacy.py` (non-loopback sockets blocked, pipeline still works, `llm_url` must be loopback), `test_repo_files.py` (README table equals the committed result, CI pins match, notices exist).
- Manual checklist after changes to the hot path: (1) short phrase in Notepad, (2) 45 s dictation (VAD split), (3) silence-only press → nothing pasted, (4) Ollama stopped → raw text still pasted, (5) rapid press/release taps don't wedge state, (6) dictate into a browser text box and VS Code, (7) check timing lines in the log.

## Evals (summary; details in `evals/README.md`)

- Layout: `evals/datasets/` (downloaders + committed manifests), `evals/run_eval.py` (runs `Pipeline`), `report.py`, `compare.py`, `safety.py`, `gates.py`, `e2e/` (real app through a virtual mic), results in `evals/results/`.
- Gates (`evals/thresholds.toml`): WER at most +0.5 points per dataset vs the profile baseline, 0 silence/noise clips pasting text, injection answered ≤ 1%, p95 latency ≤ 600 ms without cleanup (`gpu` profile only), 0 non-loopback connection attempts (`ci`). A gate with no data fails. Baselines are committed result files; change one only together with the code change that moved it.
- Before a release, on the laptop: `run_eval --gates gpu` (cleanup off) and `--gates cleanup`, plus the e2e and soak runs; commit the results. CI runs only the CPU mini-eval (`--gates ci`).
- Never quote a number in docs unless it is in a committed result from a clean tree; `python -m evals.bench_table --write` regenerates the README table and refuses dirty runs.
- `llm_url` must be loopback; config loading rejects anything else.

## Roadmap (ideas, in rough priority)

1. Streaming partial results while holding the key (Nemotron/Parakeet streaming models) to hide latency on long dictations.
2. TensorRT EP option for Parakeet (onnx-asr supports it; much higher RTFx on RTX GPUs; needs engine cache).
3. Optional Whisper large-v3-turbo backend (faster-whisper/CTranslate2) for Malayalam/Hindi, selectable per hotkey.
4. Per-app profiles (e.g. code mode in VS Code: no trailing period, camelCase helpers; casual mode in WhatsApp).
5. Small settings window (hotkey picker, mic picker, model picker) instead of editing TOML.
6. Package as a single `.exe` (Nuitka or PyInstaller) with an installer; start-with-Windows option.
7. Learn replacements from user corrections.

## What not to do

- Don't swap in Whisper as the default ASR or remove VAD.
- Don't add cloud fallbacks "for convenience".
- Don't move work into hook callbacks or the PortAudio callback.
- Don't make the LLM step mandatory or let it run on short utterances by default.
- Don't commit model files, `.venv`, or recordings, or eval audio (`evals/data/` is gitignored; only manifests and results are committed).
- Don't duplicate audio-to-text logic outside `Pipeline`, or the evals stop measuring what ships.
- Don't edit the vendored files in `evals/normalizer/`; update them from upstream as its README describes.
