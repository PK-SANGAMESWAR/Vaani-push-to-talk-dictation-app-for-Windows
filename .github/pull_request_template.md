## What and why

## Checklist
- [ ] `ruff check .`, `ruff format --check .` and `pytest -q` pass
- [ ] No new network calls, telemetry, or cloud fallbacks
- [ ] Hot-path change? Eval gate output pasted below (`--gates ci`), and latency measured before/after
- [ ] Docs/CHANGELOG updated; any numbers come from a committed `evals/results/` file
