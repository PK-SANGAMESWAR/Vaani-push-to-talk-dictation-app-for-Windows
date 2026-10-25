# Security policy

## Scope

Vaani is a local desktop app: it listens to a hotkey and the microphone, pastes into other apps, and
(optionally) talks to an Ollama server on `127.0.0.1`. Reports we care about most:

- anything that makes Vaani send audio or text off the machine (the design rule is: no network except loopback);
- audio or transcripts written to disk or logs when the user has not opted in (`save_history`);
- ways to make Vaani paste into or read from a window the user did not intend;
- code execution through `config.toml`, snippets, or model files.

Not in scope: attacks that already require running code as the same Windows user, and the licensing or
safety of models you choose to pull into Ollama.

## Reporting

Please **do not open a public issue** for a vulnerability. Use GitHub's private reporting: on the
repository page, **Security → Report a vulnerability**. Include the Vaani commit, Windows version, and
steps to reproduce. Do not attach recordings of other people.

You can expect an acknowledgement within a week. Vaani is maintained by volunteers, so fixes are
best-effort; a confirmed privacy issue gets priority over features.

## Supported versions

Only the latest commit on `main` is supported. There are no tagged releases yet.

## Verifying the privacy claim yourself

`tests/test_privacy.py` blocks every non-loopback socket and checks the pipeline still works, and CI runs
the real models with `python -m evals.run_eval --block-network`. See `evals/README.md`.
