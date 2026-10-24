# Vaani: minimal re-export. Upstream's __init__ also imports modules we don't vendor.
from .normalizer import EnglishTextNormalizer

SOURCE_REPO = "https://github.com/huggingface/open_asr_leaderboard"
SOURCE_COMMIT = "7e8d1a1003c201324e9003dd5bc34244340a63b0"

__all__ = ["SOURCE_COMMIT", "SOURCE_REPO", "EnglishTextNormalizer"]
