# Vaani eval: phase2-cleanup-off

- Created: 2026-10-24T10:47:37+00:00 · commit `2d55cfcd138b` on `main`
- ASR: `nemo-parakeet-tdt-0.6b-v3` on **cuda** (fp32), VAD on · cleanup off
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, driver 591.66 · CPU: AMD Ryzen 7 7435HS · RAM 23.7 GB
- Config sha256 `d5a153a3c3341f30` · onnxruntime-gpu 1.31.0 · onnx-asr 0.12.0

## Summary

| Set | Clips | Hours | WER raw % [95% CI] | WER final % [95% CI] | CER final % [95% CI] | ASR ms p50 / p95 | Cleanup ms p50 / p95 | Total ms p50 / p95 | RTFx |
|---|---:|---:|---|---|---|---|---|---|---:|
| dictation | 24 | 0.04 | 7.83 [4.29, 11.84] | 7.47 [3.93, 11.37] | 3.74 [1.59, 5.74] | 130 / 261 | – / – (n=0) | 130 / 261 | 42.7 |
| negatives | 56 | 0.08 | – | – | – | 20 / 45 | – / – (n=0) | 20 / 45 | 225.1 |
| **all** | 80 | 0.12 | 7.83 [4.29, 11.84] | 7.47 [3.93, 11.37] | 3.74 [1.59, 5.74] | 28 / 191 | – / – (n=0) | 28 / 191 | 89.2 |

WER/CER are corpus-level after Open ASR Leaderboard normalization. *raw* = ASR output; *final* = what Vaani would paste (replacements, snippets, cleanup). CIs: seeded bootstrap, 1000 resamples over utterances. Latency excludes audio decode and paste.

## Cold start (excluded from per-clip latency)

- Model load: 6127 ms · warm-up: 524 ms · total: 6651 ms

## Memory

- GPU memory, device-wide (all processes): baseline 4534 MB → peak 6086 MB (+1552 MB) of 6141 MB
- RAM peak (this process, peak_wset): 1117 MB

## By category

| Set | Clips | Hours | WER raw % [95% CI] | WER final % [95% CI] | CER final % [95% CI] | ASR ms p50 / p95 | Cleanup ms p50 / p95 | Total ms p50 / p95 | RTFx |
|---|---:|---:|---|---|---|---|---|---|---:|
| email | 2 | 0.00 | 12.20 [5.26, 18.18] | 12.20 [5.26, 18.18] | 6.80 [4.04, 9.35] | 207 / 256 | – / – (n=0) | 207 / 257 | 42.6 |
| fillers | 3 | 0.01 | 6.67 [0.00, 28.57] | 6.67 [0.00, 28.57] | 2.84 [0.00, 13.79] | 131 / 206 | – / – (n=0) | 131 / 206 | 40.1 |
| hum-60hz | 8 | 0.01 | – | – | – | 20 / 44 | – / – (n=0) | 20 / 44 | 228.8 |
| keyboard-clicks | 8 | 0.01 | – | – | – | 20 / 43 | – / – (n=0) | 20 / 43 | 224.8 |
| long | 2 | 0.01 | 6.25 [0.00, 12.20] | 6.25 [0.00, 12.20] | 4.34 [0.00, 8.41] | 305 / 351 | – / – (n=0) | 305 / 351 | 59.4 |
| noise-fan | 8 | 0.01 | – | – | – | 20 / 44 | – / – (n=0) | 20 / 44 | 222.3 |
| noise-room | 8 | 0.01 | – | – | – | 19 / 47 | – / – (n=0) | 19 / 47 | 223.2 |
| noise-white-loud | 8 | 0.01 | – | – | – | 21 / 44 | – / – (n=0) | 21 / 44 | 222.7 |
| numbers | 3 | 0.00 | 5.56 [0.00, 11.11] | 5.56 [0.00, 11.11] | 4.60 [0.00, 10.00] | 130 / 139 | – / – (n=0) | 130 / 139 | 42.0 |
| plain | 5 | 0.01 | 8.70 [2.04, 17.50] | 8.70 [2.04, 17.50] | 3.24 [0.44, 6.64] | 127 / 177 | – / – (n=0) | 127 / 177 | 38.3 |
| self-correction | 2 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 136 / 138 | – / – (n=0) | 136 / 139 | 40.4 |
| short | 3 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 102 / 104 | – / – (n=0) | 102 / 104 | 30.6 |
| silence-digital | 8 | 0.01 | – | – | – | 19 / 42 | – / – (n=0) | 19 / 42 | 234.9 |
| silence-dither | 8 | 0.01 | – | – | – | 20 / 47 | – / – (n=0) | 20 / 47 | 219.5 |
| technical | 4 | 0.01 | 11.90 [4.76, 19.05] | 9.52 [4.76, 15.91] | 1.90 [0.89, 3.43] | 135 / 177 | – / – (n=0) | 135 / 177 | 40.5 |

## Pipeline behaviour

- Routes: {'replace': 24, 'empty': 56} · cleanup status: –
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


## Dictation formatting (vs intended clean text)

Strict WER counts casing, punctuation and digits (24 clips). Lower is better.

| | Strict WER % [95% CI] | Content WER vs clean text % [95% CI] |
|---|---|---|
| raw ASR | 27.74 [18.18, 41.40] | 13.38 [7.44, 20.98] |
| final (what is pasted) | 27.42 [17.74, 41.25] | 13.01 [7.14, 20.47] |

Content WER normalizes both sides like the main WER but compares with the intended clean text,
so removing fillers and resolving self-corrections counts as correct. The main WER columns use
the verbatim speech as reference and count those same edits as errors.

| Clip | Intended | Pasted |
|---|---|---|
| `dict-001` | Hey, can you send me the report by end of day? | Hey, can you send me the report by end of the day? |
| `dict-003` | The build finished and all tests passed. | The bill finished and all tests passed. |
| `dict-004` | Thanks for the quick reply. I will take a look tonight. | Thanks for the quick reply, I'll take a look tonight. |
| `dict-005` | Let's move the standup to 10:30 tomorrow. | Let's move the stand-up to ten thirty tomorrow. |
| `dict-009` | So I was thinking we could push the release to Friday. | Hmm. So I was thinking we could uh push the release to Friday. |
| `dict-010` | It is basically done, just needs review. | Like I mean it's basically done, you know, just needs review. |
| `dict-011` | Can you check the logs for me? | Can you m check the locks for me? |
| `dict-012` | Let's meet at four o'clock. | Let's meet at three. No, wait, four o'clock. |
| `dict-013` | Send it to Jane. | Send it to John, I mean to Jane. |
| `dict-014` | The invoice total is $1,250. | The invoice total is one thousand two hundred and fifty dollars. |

## Worst 10 clips with errors (final WER)

| Clip | WER % | Reference (normalized) | Output (normalized) |
|---|---:|---|---|
| `dict-011` | 29 | can you check the logs for me | can you m check the locks for me |
| `dict-005` | 25 | let us move the standup to 1030 tomorrow | let us move the stand up to 1030 tomorrow |
| `dict-018` | 18 | the api returns a 404 when the user id is missing | the ap returns a 44 when the user id is missing |
| `dict-021` | 18 | hi priya thanks for sending the draft i have a few comments inline let me know if tomorrow works for a call | hi priya thanks for sending the draft i have a few comments in line let me know if tomorrow works well for your call |
| `dict-003` | 14 | the build finished and all tests passed | the bill finished and all tests passed |
| `dict-023` | 12 | so the plan for the quarter is to finish the dictation evals 1st then look at streaming partial results and after that … | so the plan for the quarter is to finish the detection evals 1st then look at the streaming partial results and after t… |
| `dict-016` | 11 | we shipped 42 orders on the 5th of march | we shipped 42 orders on 5th of march |
| `dict-020` | 10 | parakeet runs on the gpu while ollama handles the cleanup | arakeet runs on the gpu while ollama handles the cleanup |
| `dict-001` | 9 | hey can you send me the report by end of day | hey can you send me the report by end of the day |
| `dict-019` | 9 | set the cuda device to 0 and rerun the onnx model | set the cuda device to 0 and rerun the onx model |

## Provenance

| | |
|---|---|
| Manifest | `evals/datasets/manifests/dictation.fixed.jsonl` (24 clips, sha256 `b493c4783f2263eb`) |
| Manifest | `evals/datasets/manifests/negatives.fixed.jsonl` (56 clips, sha256 `924e74ef0f4be4b6`) |
| Normalizer | https://github.com/huggingface/open_asr_leaderboard @ `7e8d1a1003c2` |
| Versions | onnxruntime-gpu 1.31.0, onnx-asr 0.12.0, numpy 2.5.4, jiwer 4.0.0, soundfile 0.14.0, python 3.12.11 |
| Providers | TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider |
| OS | Windows-11-10.0.26300-SP0 |
| Command | `python -m evals.run_eval --datasets dictation,negatives --n 200 --config config.toml --cleanup off --label phase2-cleanup-off` |
