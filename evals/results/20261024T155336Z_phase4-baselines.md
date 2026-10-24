# Vaani baseline comparison: phase4-baselines

> Some runs come from a **dirty working tree** (see the Dirty column); quote none of them.

## Accuracy (WER % of the pasted text, normalized, 95% CI)

| Config | ASR | Cleanup | LS clean | LS other | FLEURS | Svarah | 4 sets pooled | Dictation content WER | Dictation strict WER |
|---|---|---|---|---|---|---|---|---|---|
| p4-parakeet-gpu-fp32 | nemo-parakeet-tdt-0.6b-v3 cuda/fp32 | off | 2.04 | 3.79 | 6.36 | 10.14 | 5.06 [4.4–5.7] | 13.01 [7.1–20.5] | 27.42 [17.7–41.3] |
| p4-parakeet-cpu-int8 | nemo-parakeet-tdt-0.6b-v3 cpu/int8 | off | 2.09 | 4.66 | 7.50 | 11.46 | 5.85 [5.2–6.6] | 13.01 [6.9–20.4] | 25.16 [15.5–37.2] |
| p4-whisper-turbo-gpu | whisper-large-v3-turbo cuda/float16 | off | 1.81 | 3.45 | 5.50 | 8.63 | 4.41 [3.8–5.0] | 12.64 [6.9–20.1] | 20.00 [12.8–28.6] |
| p4-cleanup-gemma3-4b | nemo-parakeet-tdt-0.6b-v3 cpu/int8 | gemma3:4b | 2.83 | 5.03 | 7.29 | 11.95 | 6.16 [5.4–6.9] | 9.29 [4.3–14.4] | 21.61 [12.4–32.0] |
| p4-cleanup-llama3.2-3b | nemo-parakeet-tdt-0.6b-v3 cpu/int8 | llama3.2:3b | 5.84 | 10.15 | 9.83 | 18.28 | 10.06 [9.0–11.1] | 9.67 [5.8–12.7] | 19.68 [12.0–28.0] |
| p4-cleanup-qwen2.5-7b | nemo-parakeet-tdt-0.6b-v3 cpu/int8 | qwen2.5:7b | 2.75 | 5.46 | 7.52 | 14.04 | 6.63 [5.8–7.4] | 10.41 [5.7–15.8] | 20.65 [11.9–31.4] |

## Safety

| Config | Silence/noise pasting text | Injection answered | LLM tried to answer | Cleanup regressions | Vocabulary recall |
|---|---|---|---|---|---|
| p4-parakeet-gpu-fp32 | 0/56 | – | – | – | 12/17 |
| p4-parakeet-cpu-int8 | 0/56 | – | – | – | 13/17 |
| p4-whisper-turbo-gpu | 0/56 | – | – | – | 13/17 |
| p4-cleanup-gemma3-4b | 0/56 | 0/23 | 2/23 | 56/698 | 16/17 |
| p4-cleanup-llama3.2-3b | 0/56 | 3/23 | 11/23 | 229/698 | 17/17 |
| p4-cleanup-qwen2.5-7b | 0/56 | 1/23 | 2/23 | 86/698 | 17/17 |

## Latency and resources

Pipeline time per clip (ASR + text + cleanup; no paste), clips with speech only. GPU memory is
device-wide (includes Ollama and other apps).

| Config | Total ms p50 / p95 | Cleanup ms p50 / p95 (n) | ASR RTFx | Cold start ms | GPU MB baseline → peak | Dirty | Code (vaani/ diff) | Result file |
|---|---|---|---:|---:|---|---|---|---|
| p4-parakeet-gpu-fp32 | 268 / 444 | – / – (0) | 25.9 | 7247 | 1449 → 5187 | yes (ac31804) | da0d7cee | `evals/results/20261024T150526Z_ac31804_p4-parakeet-gpu-fp32.json` |
| p4-parakeet-cpu-int8 | 277 / 660 | – / – (0) | 22.5 | 3985 | 1101 → 1338 | yes (ac31804) | c3ae1a2d | `evals/results/20261024T140305Z_ac31804_p4-parakeet-cpu-int8.json` |
| p4-whisper-turbo-gpu | 340 / 449 | – / – (0) | 20.7 | 6336 | 1337 → 3675 | yes (ac31804) | c3ae1a2d | `evals/results/20261024T140820Z_ac31804_p4-whisper-turbo-gpu.json` |
| p4-cleanup-gemma3-4b | 965 / 2054 | 628 / 1141 (698) | 15.7 | 5864 | 1365 → 5705 | yes (ac31804) | da0d7cee | `evals/results/20261024T152032Z_ac31804_p4-cleanup-gemma3-4b.json` |
| p4-cleanup-llama3.2-3b | 762 / 1836 | 416 / 899 (698) | 15.6 | 6946 | 1492 → 4454 | yes (ac31804) | da0d7cee | `evals/results/20261024T153302Z_ac31804_p4-cleanup-llama3.2-3b.json` |
| p4-cleanup-qwen2.5-7b | 1325 / 2831 | 1060 / 2032 (698) | 15.8 | 6647 | 1244 → 5754 | yes (ac31804) | da0d7cee | `evals/results/20261024T155311Z_ac31804_p4-cleanup-qwen2.5-7b.json` |
