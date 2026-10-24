# CLAUDE.md: Vaani

Vaani (വാണി, "speech") is a **fully local, push-to-talk dictation app for Windows**: hold a key, speak, release, and clean text is pasted into whatever app has focus. It is an offline, open-source take on Wispr Flow / Suki (Suki is macOS-only). No audio or text ever leaves the machine.

Target machine: Windows 11 laptop, **NVIDIA RTX 4060 (8 GB VRAM)**, Python 3.12.

## Pipeline

```
hotkey (keyboard hook) → control queue → Recorder (sounddevice, 16 kHz mono float32)
  → jobs queue → worker thread:
      Silero VAD (CPU) segments speech, drops silence
      → Parakeet TDT 0.6B v3 (onnx-asr, CUDA EP, fp32)
      → snippet match | replacements → optional Ollama cleanup (localhost) → replacements again
      → paste via clipboard + Ctrl+V, restore clipboard
```

## File map

| File | Responsibility |
|---|---|
| `vaani/__main__.py` | CLI entry (`--config`, `--file`, `--download`, `-v`), logging, single-instance mutex |
| `vaani/app.py` | Orchestration: threads, queues, hotkey → start/stop, processing, state |
| `vaani/audio.py` | `Recorder`: mic capture, optional always-open stream with pre-roll buffer |
| `vaani/stt.py` | `Transcriber`: provider selection (CUDA → CPU fallback), VAD + Parakeet, warm-up |
| `vaani/cleanup.py` | `Cleaner`: Ollama `/api/chat` call, prompt, `is_plausible` output guard |
| `vaani/textproc.py` | `TextProcessor`: custom vocabulary replacements, voice snippets |
| `vaani/output.py` | `insert_text`: clipboard paste or simulated typing |
| `vaani/tray.py` | pystray icon (state colours) + menu |
| `vaani/config.py` | `config.toml` → `Config` dataclass, validation |
| `config.toml` | User settings (hotkey, models, vocabulary, snippets) |
| `tests/` | pytest unit tests (no GPU/mic needed) |

## Commands (PowerShell, from the repo root)

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1   # venv + deps + model download
.\run.bat                                            # run in tray (pythonw, no console)
.\run-debug.bat                                      # run with console + verbose logs
.\.venv\Scripts\python -m vaani --file sample.wav    # transcribe a 16-bit WAV, print latency/RTFx
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check . ; .\.venv\Scripts\python -m ruff format .
```

Logs: `%LOCALAPPDATA%\Vaani\vaani.log` (each dictation logs audio length, ASR ms, total ms).

## Non-negotiable principles

1. **Fully local.** The only network call allowed is to Ollama on `127.0.0.1`. Never add cloud APIs, telemetry, or update checks by default. Model downloads happen once (Hugging Face cache) during setup.
2. **Latency is the product.** Budget on the RTX 4060: release-to-paste **< 400 ms** for a 10 s clip without cleanup, **< 1.2 s** with cleanup. Measure before and after any change that touches the hot path (use the timing log line or `--file`).
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

- Unit tests (`tests/`) cover config parsing, replacements/snippets, and the cleanup plausibility guard. They must run without GPU, mic, or Ollama. Mock HTTP for `Cleaner`.
- Manual checklist after changes to the hot path: (1) short phrase in Notepad, (2) 45 s dictation (VAD split), (3) silence-only press → nothing pasted, (4) Ollama stopped → raw text still pasted, (5) rapid press/release taps don't wedge state, (6) dictate into a browser text box and VS Code, (7) check timing lines in the log.

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
- Don't commit model files, `.venv`, or recordings.
