# Vaani end-to-end: e2e-nostream-cleanup-on-full (DIRTY TREE)

- Created 2026-10-24T19:30:42+00:00 · commit `0ed53bfc1141` · code `b1bd894b23db` · ASR on cpu
- Mic `CABLE Output` ← playback `CABLE Input` · hotkey `f9` · cleanup True · streaming False · lead 300 ms

## notepad

| Metric | Value |
|---|---|
| Clips | 114 |
| Release-to-paste ms p50 / p95 / max (speech clips) | 1126 / 1530 / 2162 |
| WER of pasted text % [95% CI] | 8.61 [5.60, 11.51] |
| Content WER vs intended text % | 4.45 [2.36, 6.95] |
| Strict WER vs intended text % | 11.01 [6.79, 16.12] |
| Speech clips with nothing pasted | 0 |
| Silence/noise clips that pasted text | 0/56 |
| Dropped hotkeys | 0 |
| Clips redone because another window took focus / still unfocused after redo | 0 / 0 |
| UI Automation read time p50 (latency resolution) | 0 ms |

| Clip length | n | p50 ms | p95 ms |
|---|---:|---:|---:|
| 5-15 s | 29 | 1164 | 1446 |
| <5 s | 27 | 1083 | 1260 |
| >=15 s | 2 | 2084 | 2154 |

| Clip | Reference | Pasted | ms |
|---|---|---|---:|
| `dict-001` | hey can you send me the report by end of day | Hey, can you send me the report by end of the day? | 1154 |
| `dict-005` | let's move the standup to ten thirty tomorrow | Let’s move the stand up to ten thirty tomorrow. | 1083 |
| `dict-009` | um so i was thinking we could uh push the release to friday | I was thinking we could push the release to Friday. | 1199 |
| `dict-010` | like i mean it is basically done you know just needs review | Like, it’s basically done, just needs review. | 1249 |
| `dict-012` | let's meet at three no wait four o'clock | Let’s meet at four o’clock. | 1114 |
| `dict-013` | send it to john i mean to jane | Send it to Jane. | 965 |
| `dict-018` | the api returns a four oh four when the user id is missing | The APA returns a four, four, when the user ID is missing. | 1330 |
| `dict-019` | set the cuda device to zero and rerun the onnx model | Set the CUDA device to zero and rerun the ONX model. | 1072 |
| `dict-020` | parakeet runs on the gpu while ollama handles the cleanup | Arakeep runs on the GPU while Ollama handles the cleanup. | 1306 |
| `dict-021` | hi priya thanks for sending the draft i have a few comments inline let me know if tomorro… | Hi Priya, thanks for sending the draft. I have a few comments in line. Let me know if tom… | 1647 |
| `dict-022` | dear team i am writing to confirm the schedule for next week please reply if you have any… | Dear team, I am writing to confirm the schedule for the next week. Please reply if you ha… | 1510 |
| `dict-023` | so the plan for the quarter is to finish the dictation evals first then look at streaming… | So the plan for the quarter is to finish the detection events first. Then look at the str… | 2162 |
| `dict-024` | yesterday i tried the new build on my laptop and the latency felt much better especially … | Yesterday I tried the new build on my laptop and the latency felt much better, especially… | 2006 |
| `voc-01` | i trained the model with pie torch on my laptop last night | I trained the model with PyTorch on my laptop last night. | 1030 |
| `voc-02` | we use pie torch geometric for all of the graph layers | We use PyTorch Geometric for all of the graph layers. | 1104 |
