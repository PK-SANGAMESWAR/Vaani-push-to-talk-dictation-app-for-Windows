# Vaani eval: ci-baseline

- Created: 2026-10-25T05:01:07+00:00 · commit `ace27dd65e86` on `main`
- ASR: `nemo-parakeet-tdt-0.6b-v3` on **cpu** (int8), VAD on · cleanup off
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, driver 591.66 · CPU: AMD Ryzen 7 7435HS · RAM 23.7 GB
- Config sha256 `4ff758e91e1741ef` · onnxruntime-gpu 1.31.0 · onnx-asr 0.12.0

## Summary

| Set | Clips | Hours | WER raw % [95% CI] | WER final % [95% CI] | CER final % [95% CI] | ASR ms p50 / p95 | Cleanup ms p50 / p95 | Total ms p50 / p95 | RTFx |
|---|---:|---:|---|---|---|---|---|---|---:|
| librispeech-clean | 40 | 0.09 | 1.52 [0.60, 2.79] | 1.52 [0.60, 2.79] | 0.54 [0.23, 0.97] | 367 / 948 | – / – (n=0) | 368 / 948 | 17.8 |
| negatives | 56 | 0.08 | – | – | – | 23 / 51 | – / – (n=0) | 23 / 51 | 192.7 |
| **all** | 96 | 0.17 | 1.52 [0.60, 2.79] | 1.52 [0.60, 2.79] | 0.54 [0.23, 0.97] | 50 / 719 | – / – (n=0) | 50 / 719 | 31.2 |

WER/CER are corpus-level after Open ASR Leaderboard normalization. *raw* = ASR output; *final* = what Vaani would paste (replacements, snippets, cleanup). CIs: seeded bootstrap, 1000 resamples over utterances. Latency excludes audio decode and paste.

## Network

All non-loopback connections were blocked for the whole run (`--block-network`): **none were attempted.**

## Cold start (excluded from per-clip latency)

- Model load: 4569 ms · warm-up: 124 ms · total: 4693 ms

## Memory

- GPU memory, device-wide (all processes): baseline 4983 MB → peak 4984 MB (+1 MB) of 6141 MB
- RAM peak (this process, peak_wset): 1223 MB

## By category

| Set | Clips | Hours | WER raw % [95% CI] | WER final % [95% CI] | CER final % [95% CI] | ASR ms p50 / p95 | Cleanup ms p50 / p95 | Total ms p50 / p95 | RTFx |
|---|---:|---:|---|---|---|---|---|---|---:|
| hum-60hz | 8 | 0.01 | – | – | – | 25 / 66 | – / – (n=0) | 25 / 66 | 166.4 |
| keyboard-clicks | 8 | 0.01 | – | – | – | 22 / 50 | – / – (n=0) | 22 / 50 | 200.0 |
| noise-fan | 8 | 0.01 | – | – | – | 23 / 49 | – / – (n=0) | 23 / 49 | 198.5 |
| noise-room | 8 | 0.01 | – | – | – | 24 / 50 | – / – (n=0) | 24 / 50 | 192.8 |
| noise-white-loud | 8 | 0.01 | – | – | – | 23 / 52 | – / – (n=0) | 23 / 52 | 192.8 |
| silence-digital | 8 | 0.01 | – | – | – | 21 / 50 | – / – (n=0) | 21 / 50 | 203.6 |
| silence-dither | 8 | 0.01 | – | – | – | 23 / 49 | – / – (n=0) | 23 / 49 | 200.1 |

## Pipeline behaviour

- Routes: {'replace': 40, 'empty': 56} · cleanup status: –
- Clips with speech that produced no text: 0

## Silence / noise (reference is empty)

- Clips: 56 · text out of ASR: **0** · text that would be pasted: **0** (target: 0)

| Kind | Clips | Pasted text |
|---|---:|---:|
| hum-60hz | 8 | 0 |
| keyboard-clicks | 8 | 0 |
| noise-fan | 8 | 0 |
| noise-room | 8 | 0 |
| noise-white-loud | 8 | 0 |
| silence-digital | 8 | 0 |
| silence-dither | 8 | 0 |


## Pipeline safety

| Check | Result | Target |
|---|---|---|
| Silence/noise clips that paste text (hallucination) | **0.0%** (0/56, 95% CI 0.0–6.4) | 0% |

## Worst 10 clips with errors (final WER)

| Clip | WER % | Reference (normalized) | Output (normalized) |
|---|---:|---|---|
| `61-70970-0034` | 11 | nay nay lording answered warrenton with a half laugh | nay nay lording answered warrington with a half laugh |
| `4507-16021-0021` | 11 | the painter who says my grinder the notary who says my skip the gutter the hairdresser who says my mealyback the cobble… | the painter who says my grinder the notary who says my skip 0 the gutter the hairdresser who says my mealy back the cob… |
| `4970-29093-0014` | 8 | well i am going as an engineer you can go as one | well i am going as an engineer you could go as one |
| `8555-284449-0006` | 8 | do not worry sizzle dear it will all come right pretty soon | do not worry sizzle dear it will all come right pretty soon but |
| `4446-2273-0016` | 8 | her hair is still like flax and her blue eyes are just like a baby s and she has the same 3 freckles on her little nose… | her hair is still like flax and her blue eyes are just like a baby s and she has the same 3 freckles on her little nose… |
| `2300-131720-0000` | 8 | the paris plant like that at the crystal palace was a temporary exhibit | the paris plant like that of the crystal palace was a temporary exhibit |
| `2094-142345-0059` | 7 | this missus poyser said blushing and believing that the captain was really interested in her milk pans and would adjust… | this misses poiser said blushing and believing that the captain was really interested in her milk pans and would adjust… |
| `6829-68771-0019` | 5 | she was dressed in the regulation costume of the maids at elmhurst a plain black gown with white apron and cap | she was dressed in the regulation costume of the maids at elmhurst a plain black gown with a white apron and cap |

## Provenance

| | |
|---|---|
| Manifest | `evals/datasets/manifests/librispeech-clean.n40.s0.jsonl` (40 clips, sha256 `c48f0b247a836f3a`) |
| Manifest | `evals/datasets/manifests/negatives.fixed.jsonl` (56 clips, sha256 `924e74ef0f4be4b6`) |
| Normalizer | https://github.com/huggingface/open_asr_leaderboard @ `7e8d1a1003c2` |
| Versions | onnxruntime-gpu 1.31.0, onnx-asr 0.12.0, numpy 2.5.4, jiwer 4.0.0, soundfile 0.14.0, faster-whisper 1.2.1, ctranslate2 4.8.3, python 3.12.11 |
| Providers | TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider |
| OS | Windows-11-10.0.26300-SP0 |
| Command | `python -m evals.run_eval --datasets librispeech-clean,negatives --n 40 --config config.toml --device cpu --quantization int8 --asr config --cleanup off --block-network --label ci-baseline` |
