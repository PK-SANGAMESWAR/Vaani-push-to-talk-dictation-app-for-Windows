# Vaani

**Fully local push-to-talk dictation for Windows.** Hold a key, speak, release, and clean text appears in whatever app you're typing in. No cloud, no account, no word limits.

- **ASR:** NVIDIA Parakeet TDT 0.6B v3 on your GPU (CUDA), with Silero VAD so silence never turns into text
- **Cleanup (optional):** a small local LLM via Ollama fixes punctuation, removes "um"s and handles self-corrections
- **Custom vocabulary + voice snippets** in `config.toml`
- Tray icon: 🟦 loading · 🟩 ready · 🟥 listening · 🟨 transcribing

## Requirements

- Windows 10/11
- NVIDIA GPU with a recent driver (R580 or newer, for CUDA 13). Works on CPU too, just slower.
- [Python 3.12](https://www.python.org/downloads/) (tick "Add to PATH" / install the `py` launcher)
- [Ollama](https://ollama.com) (optional, for AI cleanup)

## Setup

```powershell
cd path\to\vaani
powershell -ExecutionPolicy Bypass -File setup.ps1
```

This creates `.venv`, installs dependencies (~1 GB because of the CUDA libraries), downloads the Parakeet and Silero models, and pulls `gemma3:4b` in Ollama if it's installed.

## Use

Double-click `run.bat`. When the tray icon turns green, **hold Right Ctrl, speak, release.**

- Short dictations (under 8 words) skip the LLM and paste almost instantly.
- Toggle AI cleanup from the tray menu.
- Debug with `run-debug.bat`; logs live in `%LOCALAPPDATA%\Vaani\vaani.log`.
- Start with Windows: press `Win+R`, type `shell:startup`, and put a shortcut to `run.bat` there.
- Benchmark: `.venv\Scripts\python -m vaani --file some.wav`

## Configure

Edit `config.toml` (tray → Open config), then restart. Highlights:

| Setting | What it does |
|---|---|
| `[hotkey] key / mode` | Push-to-talk key, or toggle mode with combos |
| `[audio] keep_mic_open` | Zero start delay + pre-roll so your first word is never clipped (mic indicator stays on) |
| `[cleanup] llm_model` | Any Ollama model; small 1–4B instruct models are the sweet spot for an 8 GB GPU |
| `[replacements]` | "pie torch" → "PyTorch" style fixes |
| `[snippets]` | Say "insert my signature" to paste a block of text |

## Notes

- Parakeet v3 handles English and 24 European languages. It does not support Malayalam or Hindi yet.
- Pasting into apps running as Administrator requires running Vaani as Administrator too.
- The clipboard is restored after pasting, but only if it held text.

MIT License.
