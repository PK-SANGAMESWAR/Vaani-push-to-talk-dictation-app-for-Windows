# Vaani end-to-end: e2e-nostream-cleanup-off (DIRTY TREE)

- Created 2026-10-24T16:43:37+00:00 · commit `0ed53bfc1141` · code `b1bd894b23db` · ASR on cuda
- Mic `CABLE Output` ← playback `CABLE Input` · hotkey `f9` · cleanup False · streaming False · lead 300 ms

## notepad

| Metric | Value |
|---|---|
| Clips | 15 |
| Release-to-paste ms p50 / p95 / max (speech clips) | 662 / 757 / 760 |
| WER of pasted text % [95% CI] | 3.85 [0.00, 8.70] |
| Content WER vs intended text % | 5.71 [0.00, 14.43] |
| Strict WER vs intended text % | 10.08 [3.20, 21.49] |
| Speech clips with nothing pasted | 0 |
| Silence/noise clips that pasted text | 0/5 |
| Dropped hotkeys | 0 |
| Clips redone because another window took focus / still unfocused after redo | 0 / 0 |
| UI Automation read time p50 (latency resolution) | 0 ms |

| Clip length | n | p50 ms | p95 ms |
|---|---:|---:|---:|
| 5-15 s | 6 | 666 | 747 |
| <5 s | 4 | 620 | 740 |

| Clip | Reference | Pasted | ms |
|---|---|---|---:|
| `dict-001` | hey can you send me the report by end of day | Hey, can you send me the report by end of the day? | 760 |
| `dict-003` | the build finished and all tests passed | The bill finished and all tests passed. | 706 |
| `dict-005` | let's move the standup to ten thirty tomorrow | Let's move the stand-up to ten thirty tomorrow. | 754 |
