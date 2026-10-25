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

## Benchmarks

<!-- bench:start -->
Parakeet TDT 0.6B v3 fp32 on NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), driver 591.66, AMD Ryzen 7 7435HS; shipped pipeline (Silero VAD, no cleanup), 800 clips = 200 per set (seed 0). WER is corpus-level on Open ASR Leaderboard-normalized text with a 95% bootstrap CI. Latency is `Pipeline.process` per clip (no paste). From a clean commit (`2131f02`): [`20261024T101301Z_2131f02_gpu-fp32-4sets.json`](evals/results/20261024T101301Z_2131f02_gpu-fp32-4sets.json).

| Set | Clips | WER % [95% CI] | Latency ms p50 / p95 | ASR RTFx |
|---|---:|---|---|---:|
| FLEURS en_us | 200 | 7.22 [5.9–8.6] | 170 / 229 | 55.8 |
| LibriSpeech test-clean | 200 | 2.04 [1.5–2.6] | 143 / 259 | 45.6 |
| LibriSpeech test-other | 200 | 3.79 [2.9–4.8] | 133 / 227 | 43.6 |
| Svarah (Indian-accented English) | 200 | 10.14 [7.7–12.5] | 133 / 222 | 35.6 |
| **All four** | 800 | 5.33 [4.7–6.0] | 145 / 238 | 45.7 |

Harness check: the same model without VAD on the full LibriSpeech test-clean (2620 clips) gives 1.89% WER [1.74–2.05], against 1.93% published on the model card (inside our CI: yes). Result: [`20261024T100052Z_255cdb2_validate-novad.json`](evals/results/20261024T100052Z_255cdb2_validate-novad.json).
<!-- bench:end -->

Reproduce (needs the eval dependencies, a free Hugging Face login for the gated Svarah set, ~3 GB of disk):

```powershell
.\.venv\Scripts\python -m pip install -r requirements-eval.txt ; .\.venv\Scripts\python -m evals.datasets.prepare ; .\.venv\Scripts\python -m evals.run_eval --device cuda --cleanup off --label repro
```

The README table is generated from the committed result file (`python -m evals.bench_table --check` runs in
the tests), and only runs from a clean commit are quoted. Later runs (silence/noise, injection safety,
CPU vs GPU vs Whisper, cleanup models, end-to-end release-to-paste latency, 1000-dictation soak) are in
[`evals/results/`](evals/results); they were taken on a working tree with uncommitted code, so their
reports are marked dirty and are not quoted here until re-run on a commit. Method, datasets and gates:
[evals/README.md](evals/README.md).

## Privacy

Audio and text never leave your machine. The only network call is to Ollama on `127.0.0.1`, and
`llm_url` is rejected at startup if it points anywhere else. The test suite blocks every non-loopback
socket and checks the pipeline still works (`tests/test_privacy.py`), and CI repeats that with the real
models (`--block-network`). Models are downloaded once, during setup. See [SECURITY.md](SECURITY.md).

## Limitations

- **Languages:** Parakeet v3 covers English and 24 European languages. No Malayalam or Hindi.
- **Accuracy is model-bound:** the table above is read speech. Accented English is measurably harder
  (Svarah), and your microphone, room and speaking style will differ. The only dictation-style test set
  is one speaker on one laptop microphone, so treat it as a smoke test, not a benchmark.
- **Cleanup can be wrong:** a small local LLM sometimes rewrites more than asked or tries to answer a
  dictated question instead of cleaning it. Vaani falls back to the raw transcript when its output looks
  implausible, but cleanup is optional for a reason; compare models with the eval suite.
- **Latency is measured on one machine** (RTX 4050 Laptop, 6 GB). CPU-only is slower; the GPU shares
  VRAM with the cleanup model, so `device = "auto"` runs ASR on the CPU when cleanup is on.
- **Windows only**, and the Fn key is invisible to Windows on most laptops. Pasting into
  Administrator windows needs Vaani to run as Administrator. Only text on the clipboard is restored.
- CI cannot test the GPU or the real keyboard/microphone path; those run locally before a release.

## Notes

- Parakeet v3 handles English and 24 European languages. It does not support Malayalam or Hindi yet.
- Pasting into apps running as Administrator requires running Vaani as Administrator too.
- The clipboard is restored after pasting, but only if it held text.

MIT License. Third-party models, libraries and datasets: see [NOTICE](NOTICE). Contributing: [CONTRIBUTING.md](CONTRIBUTING.md).
