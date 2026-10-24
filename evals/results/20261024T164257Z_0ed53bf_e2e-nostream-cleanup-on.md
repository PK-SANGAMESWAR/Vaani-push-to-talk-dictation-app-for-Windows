# Vaani end-to-end: e2e-nostream-cleanup-on (DIRTY TREE)

- Created 2026-10-24T16:41:08+00:00 · commit `0ed53bfc1141` · code `b1bd894b23db` · ASR on cpu
- Mic `CABLE Output` ← playback `CABLE Input` · hotkey `f9` · cleanup True · streaming False · lead 300 ms

## notepad

| Metric | Value |
|---|---|
| Clips | 15 |
| Release-to-paste ms p50 / p95 / max (speech clips) | 1103 / 1372 / 1489 |
| WER of pasted text % [95% CI] | 2.88 [0.00, 7.37] |
| Content WER vs intended text % | 4.76 [0.00, 13.13] |
| Strict WER vs intended text % | 9.24 [1.67, 20.01] |
| Speech clips with nothing pasted | 0 |
| Silence/noise clips that pasted text | 0/5 |
| Dropped hotkeys | 0 |
| UI Automation read time p50 (latency resolution) | 0 ms |

| Clip length | n | p50 ms | p95 ms |
|---|---:|---:|---:|
| 5-15 s | 6 | 1053 | 1416 |
| <5 s | 4 | 1155 | 1220 |

| Clip | Reference | Pasted | ms |
|---|---|---|---:|
| `dict-001` | hey can you send me the report by end of day | Hey, can you send me the report by end of the day? | 1489 |
| `dict-005` | let's move the standup to ten thirty tomorrow | Let’s move the stand up to ten thirty tomorrow. | 1043 |
