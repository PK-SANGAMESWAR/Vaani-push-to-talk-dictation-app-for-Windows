# Vaani eval suite

Measures the code that ships: every clip goes through `vaani.pipeline.Pipeline`, the same object the
app uses (VAD + Parakeet → snippets / replacements → optional Ollama cleanup). Nothing is pasted.

## Quick start (PowerShell, repo root)

```powershell
.\.venv\Scripts\python -m pip install -r requirements-eval.txt
.\.venv\Scripts\python -m evals.datasets.prepare            # download + write manifests (once)
.\.venv\Scripts\python -m evals.run_eval --cleanup off --label gpu-fp32
```

Each run writes `evals/results/<UTC time>_<commit>_<label>.json` (everything, including every clip's
reference, output, and timings) and a `.md` report rendered only from that JSON.

## Datasets

| Name | Source | What it tests | License |
|---|---|---|---|
| `librispeech-clean` | LibriSpeech test-clean ([openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) on HF, or openslr.org) | read audiobook speech, clean | CC-BY-4.0 |
| `librispeech-other` | LibriSpeech test-other | harder speakers/acoustics | CC-BY-4.0 |
| `fleurs-en` | [google/fleurs](https://huggingface.co/datasets/google/fleurs) en_us test | read Wikipedia sentences, many speakers | CC-BY-4.0 |
| `svarah` | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | Indian-accented English, tagged by L1 | CC-BY-4.0 |

- Default: a seeded subset of **200 clips per dataset** (seed 0). `--full` uses the whole split.
  Options: `--n 50 --seed 1`, or a dataset name list (`prepare fleurs-en svarah`).
- Every source is pinned (HF revision or openslr md5) and recorded in the manifest's `_meta` line.
- **Svarah is gated:** run `hf auth login` and accept the terms on its dataset page first.
- LibriSpeech source: HF by default (faster); set `VAANI_LIBRISPEECH_SOURCE=openslr` for the original
  tarballs (md5-checked). Both produce the same manifest rows.
- Audio and archives go to `evals/data/` (gitignored). **Only manifests are committed.**
- This is the only part of the project that downloads data. The app never does.

### Manifest format (JSONL)

Line 1: `{"_meta": {...}}` (source, revision, checksums, seed, n, population). Then one clip per line:

| Field | Meaning |
|---|---|
| `id` | unique clip id |
| `dataset` | dataset name |
| `audio` | path relative to `evals/data/` (16 kHz) |
| `duration_s` | clip length |
| `ref` | verbatim reference (`""` for silence/noise-only clips) |
| `ref_clean` | intended clean text (dictation sets); `null` for public ASR sets |
| `category` | breakdown key (public sets: the dataset name) |
| `speaker`, `mic` | optional |
| `tags` | free-form, e.g. Svarah `{"l1": "Malayalam", "gender": "female", ...}` |

## run_eval options

| Flag | Meaning |
|---|---|
| `--datasets a,b` / `--manifest path.jsonl` | what to run (default: all four subsets) |
| `--full`, `--n`, `--seed` | which manifests to pick |
| `--limit N` | first N clips per manifest (smoke test) |
| `--config` | config file (default `config.toml`); overrides below apply on top |
| `--device auto/cuda/cpu`, `--quantization auto/none/int8` | ASR runtime |
| `--cleanup config/on/off` | LLM cleanup (default: as in config) |
| `--no-vad` | **diagnostic only**: feed whole clips to the model, bypassing VAD (not the shipped pipeline) |
| `--label NAME` | suffix for the result file names (default `run`) |
| `--out DIR` | where results go (default `evals/results/`) |
| `--published-wer X --published-source URL --published-note TEXT` | compare overall raw WER with a published figure |
| `-v` | verbose logging |

## What the report contains

- **WER / CER** (corpus-level: total edits / total reference words or characters), with reference and
  output both normalized by the Open ASR Leaderboard's `EnglishTextNormalizer` (vendored in
  `evals/normalizer/`, pinned commit). CER counts characters of the normalized text, spaces included.
  - *raw* = ASR output. *final* = what Vaani would paste (after replacements / snippets / cleanup).
  - 95% CI: seeded bootstrap, 1000 resamples over utterances, percentile interval.
  - Clips whose normalized reference is empty are excluded from WER and counted separately.
- **Breakdowns** by dataset, category, and tag (Svarah L1 and gender, FLEURS gender).
- **Latency** p50/p95/mean/max for ASR, cleanup (only clips where it ran), and total, per clip. This
  is `Pipeline.process` only: audio decoding and paste are excluded. True release-to-paste latency is
  measured by the end-to-end test (later phase). **RTFx** = audio seconds / ASR seconds.
- **Cold start**: model load + warm-up (+ Ollama prewarm), reported separately and excluded from
  per-clip latency.
- **Memory**: GPU memory used, device-wide (all processes, including Ollama), as baseline before model
  load → peak during the run; per-process GPU memory only if the driver reports it. RAM = this
  process's peak working set.
- **Provenance**: git commit + dirty flag, config sha256 (effective config after overrides) and values,
  ASR model / device / quantization, VAD params, Ollama version + model digest (when cleanup is on),
  package versions, GPU name + driver, CPU, RAM, OS, manifest sha256s, normalizer commit, the command.

A run on a dirty working tree is marked "DIRTY TREE" in the report title. Results meant for the README
must come from a clean commit.

## Harness validation

`--no-vad` on full LibriSpeech test-clean should land close to the published Parakeet v3 number
(model card table "Huggingface Open-ASR-Leaderboard": 1.93% WER). A gap outside the CI means the
harness or the ONNX export differs from the reference setup and must be explained before other
numbers are trusted.

```powershell
.\.venv\Scripts\python -m evals.datasets.prepare librispeech-clean --full
.\.venv\Scripts\python -m evals.run_eval --datasets librispeech-clean --full --no-vad --cleanup off --label validate-novad --published-wer 1.93 --published-source https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3 --published-note "model card commit 541d1f99, table 'Huggingface Open-ASR-Leaderboard', LS test-clean"
```

## Tests

`tests/test_evals.py` covers the harness itself (normalizer behaviour, WER/CER, bootstrap CIs,
percentiles, manifest reading/writing, provenance) and `tests/test_pipeline.py` covers `Pipeline`
routing and timings. Both run with plain `pytest`: no GPU, audio, datasets, or network needed.

## Attribution

LibriSpeech (Panayotov et al., 2015), FLEURS (Conneau et al., 2022), and Svarah (Javed et al., 2023)
are CC-BY-4.0; manifests contain their reference transcripts, and result files contain references
plus Vaani's output for each clip.

TODO(phase6-notice): add to NOTICE: the vendored normalizer (`evals/normalizer/`, Apache-2.0,
copyright The OpenAI team and The HuggingFace Team) and the dataset attributions above.
