"""Deterministic text post-processing: custom vocabulary replacements and voice snippets."""

from __future__ import annotations

import re

_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _SPACES.sub(" ", _PUNCT.sub("", text.lower())).strip()


class TextProcessor:
    def __init__(self, replacements: dict[str, str] | None = None, snippets: dict[str, str] | None = None):
        self.replacements = replacements or {}
        self.snippets = {_normalize(k): v for k, v in (snippets or {}).items()}
        # Longest phrases first so "pie torch geometric" wins over "pie torch".
        self._patterns = [
            (re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.IGNORECASE), dst)
            for src, dst in sorted(self.replacements.items(), key=lambda kv: -len(kv[0]))
        ]

    def snippet(self, text: str) -> str | None:
        """If the whole utterance is a snippet trigger ("insert my email"), return its expansion."""
        return self.snippets.get(_normalize(text))

    def replace(self, text: str) -> str:
        for pattern, dst in self._patterns:
            text = pattern.sub(lambda _m, dst=dst: dst, text)  # callable = literal; no \n / \1 expansion
        return text

    @property
    def vocabulary(self) -> list[str]:
        """Preferred spellings, handed to the LLM so it doesn't 'fix' them back."""
        return sorted(set(self.replacements.values()))


def word_count(text: str) -> int:
    return len(text.split())
