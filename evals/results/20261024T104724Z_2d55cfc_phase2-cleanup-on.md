# Vaani eval: phase2-cleanup-on

- Created: 2026-10-24T10:47:24+00:00 · commit `2d55cfcd138b` on `main`
- ASR: `nemo-parakeet-tdt-0.6b-v3` on **cpu** (int8), VAD on · cleanup on (gemma3:4b)
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, driver 591.66 · CPU: AMD Ryzen 7 7435HS · RAM 23.7 GB
- Config sha256 `6c6bf7bec3adeb9f` · onnxruntime-gpu 1.31.0 · onnx-asr 0.12.0

## Summary

| Set | Clips | Hours | WER raw % [95% CI] | WER final % [95% CI] | CER final % [95% CI] | ASR ms p50 / p95 | Cleanup ms p50 / p95 | Total ms p50 / p95 | RTFx |
|---|---:|---:|---|---|---|---|---|---|---:|
| dictation | 24 | 0.04 | 7.12 [3.57, 10.80] | 12.10 [6.41, 19.00] | 6.97 [3.08, 11.55] | 224 / 675 | 330 / 879 (n=20) | 549 / 1505 | 24.6 |
| negatives | 56 | 0.08 | – | – | – | 26 / 60 | – / – (n=0) | 26 / 60 | 169.6 |
| **all** | 80 | 0.12 | 7.12 [3.57, 10.80] | 12.10 [6.41, 19.00] | 6.97 [3.08, 11.55] | 38 / 325 | 330 / 879 (n=20) | 38 / 709 | 54.6 |

WER/CER are corpus-level after Open ASR Leaderboard normalization. *raw* = ASR output; *final* = what Vaani would paste (replacements, snippets, cleanup). CIs: seeded bootstrap, 1000 resamples over utterances. Latency excludes audio decode and paste.

## Cold start (excluded from per-clip latency)

- Model load: 3789 ms · warm-up: 117 ms · total: 3907 ms · Ollama prewarm: 3812 ms

## Memory

- GPU memory, device-wide (all processes): baseline 809 MB → peak 4562 MB (+3752 MB) of 6141 MB
- RAM peak (this process, peak_wset): 1005 MB

## By category

| Set | Clips | Hours | WER raw % [95% CI] | WER final % [95% CI] | CER final % [95% CI] | ASR ms p50 / p95 | Cleanup ms p50 / p95 | Total ms p50 / p95 | RTFx |
|---|---:|---:|---|---|---|---|---|---|---:|
| email | 2 | 0.00 | 12.20 [5.26, 18.18] | 12.20 [5.26, 18.18] | 6.80 [4.04, 9.35] | 382 / 385 | 573 / 584 (n=2) | 955 / 963 | 23.1 |
| fillers | 3 | 0.01 | 3.33 [0.00, 9.09] | 23.33 [0.00, 50.00] | 17.73 [0.00, 37.29] | 269 / 286 | 279 / 323 (n=3) | 548 / 609 | 25.3 |
| hum-60hz | 8 | 0.01 | – | – | – | 26 / 62 | – / – (n=0) | 26 / 62 | 164.2 |
| keyboard-clicks | 8 | 0.01 | – | – | – | 27 / 57 | – / – (n=0) | 27 / 57 | 169.3 |
| long | 2 | 0.01 | 6.25 [0.00, 12.20] | 6.25 [0.00, 12.20] | 4.34 [0.00, 8.41] | 764 / 798 | 922 / 966 (n=2) | 1686 / 1764 | 23.7 |
| noise-fan | 8 | 0.01 | – | – | – | 25 / 57 | – / – (n=0) | 25 / 57 | 172.9 |
| noise-room | 8 | 0.01 | – | – | – | 25 / 61 | – / – (n=0) | 25 / 61 | 169.7 |
| noise-white-loud | 8 | 0.01 | – | – | – | 27 / 59 | – / – (n=0) | 27 / 59 | 166.9 |
| numbers | 3 | 0.00 | 5.56 [0.00, 11.11] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 239 / 247 | 326 / 342 (n=3) | 550 / 571 | 24.2 |
| plain | 5 | 0.01 | 6.52 [0.00, 15.79] | 10.87 [2.22, 23.81] | 3.70 [0.95, 7.11] | 191 / 237 | 324 / 423 (n=4) | 491 / 619 | 26.4 |
| self-correction | 2 | 0.00 | 0.00 [0.00, 0.00] | 47.06 [44.44, 50.00] | 41.27 [33.33, 50.00] | 184 / 203 | 238 / 280 (n=2) | 422 / 482 | 29.9 |
| short | 3 | 0.00 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 139 / 143 | – / – (n=0) | 139 / 143 | 24.1 |
| silence-digital | 8 | 0.01 | – | – | – | 26 / 61 | – / – (n=0) | 26 / 61 | 168.9 |
| silence-dither | 8 | 0.01 | – | – | – | 25 / 57 | – / – (n=0) | 25 / 57 | 176.0 |
| technical | 4 | 0.01 | 11.90 [4.76, 19.05] | 9.52 [4.76, 15.91] | 2.38 [0.93, 3.77] | 235 / 310 | 348 / 372 (n=4) | 583 / 682 | 23.2 |

## Pipeline behaviour

- Routes: {'cleanup': 20, 'replace': 4, 'empty': 56} · cleanup status: {'ok': 20}
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
| raw ASR | 25.48 [15.69, 37.46] | 13.38 [7.16, 20.57] |
| final (what is pasted) | 20.00 [11.61, 29.91] | 9.67 [4.56, 15.00] |

Content WER normalizes both sides like the main WER but compares with the intended clean text,
so removing fillers and resolving self-corrections counts as correct. The main WER columns use
the verbatim speech as reference and count those same edits as errors.

| Clip | Intended | Pasted |
|---|---|---|
| `dict-001` | Hey, can you send me the report by end of day? | Hey, can you send me the report by end of the day? |
| `dict-004` | Thanks for the quick reply. I will take a look tonight. | Thanks for the quick reply. I’ll take a look tonight. |
| `dict-005` | Let's move the standup to 10:30 tomorrow. | Let’s move the stand up to ten thirty tomorrow. |
| `dict-009` | So I was thinking we could push the release to Friday. | I was thinking we could push the release to Friday. |
| `dict-010` | It is basically done, just needs review. | It’s basically done, just needs review. |
| `dict-012` | Let's meet at four o'clock. | Let’s meet at four o’clock. |
| `dict-014` | The invoice total is $1,250. | The invoice total is one thousand two hundred and fifty dollars. |
| `dict-015` | Call me on 9876543210. | Call me on nine eight seven six five four three two one zero |
| `dict-016` | We shipped 42 orders on the 5th of March. | We shipped forty-two orders on the fifth of March. |
| `dict-018` | The API returns a 404 when the user ID is missing. | The APA returns a four, four when the user ID is missing. |

## Worst 10 clips with errors (final WER)

| Clip | WER % | Reference (normalized) | Output (normalized) |
|---|---:|---|---|
| `dict-010` | 50 | like i mean it is basically done you know just needs review | it s basically done just needs review |
| `dict-013` | 50 | send it to john i mean to jane | send it to jane |
| `dict-012` | 44 | let us meet at 3 no wait 40 clock | let s meet at 40 clock |
| `dict-005` | 38 | let us move the standup to 1030 tomorrow | let s move the stand up to 1030 tomorrow |
| `dict-018` | 18 | the api returns a 404 when the user id is missing | the apa returns a 44 when the user id is missing |
| `dict-021` | 18 | hi priya thanks for sending the draft i have a few comments inline let me know if tomorrow works for a call | hi priya thanks for sending the draft i have a few comments in line let me know if tomorrow works well for your call |
| `dict-023` | 12 | so the plan for the quarter is to finish the dictation evals 1st then look at streaming partial results and after that … | so the plan for the quarter is to finish the detection evaluation 1st then look at the streaming partial results and af… |
| `dict-020` | 10 | parakeet runs on the gpu while ollama handles the cleanup | arakeep runs on the gpu while ollama handles the cleanup |
| `dict-001` | 9 | hey can you send me the report by end of day | hey can you send me the report by end of the day |
| `dict-004` | 9 | thanks for the quick reply i will take a look tonight | thanks for the quick reply i ll take a look tonight |

## Provenance

| | |
|---|---|
| Manifest | `evals/datasets/manifests/dictation.fixed.jsonl` (24 clips, sha256 `b493c4783f2263eb`) |
| Manifest | `evals/datasets/manifests/negatives.fixed.jsonl` (56 clips, sha256 `924e74ef0f4be4b6`) |
| Normalizer | https://github.com/huggingface/open_asr_leaderboard @ `7e8d1a1003c2` |
| Versions | onnxruntime-gpu 1.31.0, onnx-asr 0.12.0, numpy 2.5.4, jiwer 4.0.0, soundfile 0.14.0, python 3.12.11 |
| Providers | TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider |
| OS | Windows-11-10.0.26300-SP0 |
| Command | `python -m evals.run_eval --datasets dictation,negatives --n 200 --config config.toml --cleanup on --label phase2-cleanup-on` |
