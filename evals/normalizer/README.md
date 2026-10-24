# Vendored text normalizer

`normalizer.py` and `english_abbreviations.py` are copied unmodified from
[huggingface/open_asr_leaderboard](https://github.com/huggingface/open_asr_leaderboard/tree/7e8d1a1003c201324e9003dd5bc34244340a63b0/normalizer)
at commit `7e8d1a1003c201324e9003dd5bc34244340a63b0` (2026-10-21). They derive from OpenAI Whisper's
`EnglishTextNormalizer`. License: Apache-2.0 (see `LICENSE` in this folder).

`__init__.py` is Vaani's own: it re-exports `EnglishTextNormalizer` only, because the upstream
`__init__` also imports modules we don't vendor.

We vendor it (instead of installing `openai-whisper` or `transformers`) so the eval suite uses the
exact normalizer the Open ASR Leaderboard uses, with a single small dependency (`regex`), and so
results can't drift when an upstream package changes.

To update: download the two files at a newer commit, update `SOURCE_COMMIT` in `__init__.py` and this
README, and re-run the evals (a normalizer change can move WER).

TODO(phase6-notice): list this vendored code (Apache-2.0, copyright The OpenAI team and The
HuggingFace Team) in the repo's NOTICE file.

SHA-256 of the vendored files:

```
52997cc963e0bd6568d15554ad8d5fb0759f20f361483632a22f90d2d0e07db1  english_abbreviations.py
490b56393484ef386b486f3679cc2261264af280b3512154431e0bdcf2778295  normalizer.py
5ee13882fce0975f0ad3c3d5c2042af4896c2a669be2d6df5b3d256811267c85  LICENSE
```
