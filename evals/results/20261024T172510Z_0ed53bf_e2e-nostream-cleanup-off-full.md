# Vaani end-to-end: e2e-nostream-cleanup-off-full (DIRTY TREE)

- Created 2026-10-24T17:11:58+00:00 · commit `0ed53bfc1141` · code `b1bd894b23db` · ASR on cuda
- Mic `CABLE Output` ← playback `CABLE Input` · hotkey `f9` · cleanup False · streaming False · lead 300 ms

## notepad

| Metric | Value |
|---|---|
| Clips | 114 |
| Release-to-paste ms p50 / p95 / max (speech clips) | 658 / 781 / 786 |
| WER of pasted text % [95% CI] | 6.68 [4.06, 9.53] |
| Content WER vs intended text % | 5.99 [3.35, 9.20] |
| Strict WER vs intended text % | 14.27 [9.59, 20.42] |
| Speech clips with nothing pasted | 0 |
| Silence/noise clips that pasted text | 0/56 |
| Dropped hotkeys | 0 |
| Clips redone because another window took focus / still unfocused after redo | 0 / 0 |
| UI Automation read time p50 (latency resolution) | 1 ms |

| Clip length | n | p50 ms | p95 ms |
|---|---:|---:|---:|
| 5-15 s | 29 | 638 | 783 |
| <5 s | 27 | 665 | 747 |
| >=15 s | 2 | 739 | 747 |

| Clip | Reference | Pasted | ms |
|---|---|---|---:|
| `dict-001` | hey can you send me the report by end of day | Hey, can you send me the report by end of the day? | 698 |
| `dict-003` | the build finished and all tests passed | The bill finished and all tests passed. | 637 |
| `dict-005` | let's move the standup to ten thirty tomorrow | Let's move the stand-up to ten thirty tomorrow. | 734 |
| `dict-016` | we shipped forty two orders on the fifth of march | We shipped forty-two orders on fifth of March. | 784 |
| `dict-018` | the api returns a four oh four when the user id is missing | The AP returns a four, four, when the user ID is missing. | 641 |
| `dict-019` | set the cuda device to zero and rerun the onnx model | Set the CUDA device to zero and rerun the ONEX model. | 759 |
| `dict-020` | parakeet runs on the gpu while ollama handles the cleanup | Arakeet runs on the GPU while Ollama handles the cleanup. | 607 |
| `dict-021` | hi priya thanks for sending the draft i have a few comments inline let me know if tomorro… | Hi Priya, thanks for sending the draft. I have a few comments in line. Let me know if tom… | 628 |
| `dict-022` | dear team i am writing to confirm the schedule for next week please reply if you have any… | Dear team, I am writing to confirm the schedule for the next week. Please reply if you ha… | 617 |
| `dict-023` | so the plan for the quarter is to finish the dictation evals first then look at streaming… | So the plan for the quarter is to finish the detection evals first, then look at the stre… | 748 |
| `voc-01` | i trained the model with pie torch on my laptop last night | I trained the model with PyTorch on my laptop last night. | 588 |
| `voc-02` | we use pie torch geometric for all of the graph layers | We use PyTorch geometric for all of the graph layers. | 737 |
| `voc-03` | the agent is built with lang chain and the workflow uses lang graph | The agent is built with LangChain and the workflow uses LangGraph. | 656 |
| `voc-04` | the backend is a fast api service running behind nginx | The backend is a FastAPI service running behind links. | 603 |
| `voc-05` | olama runs the cleanup model locally on the laptop gpu | Alama runs the cleanup model locally on the laptop GPU. | 717 |
