"""Load config.toml into a typed Config object (missing keys fall back to defaults)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

# TOML section -> Config field names that live in it.
_SECTIONS: dict[str, tuple[str, ...]] = {
    "hotkey": ("hotkey", "mode"),
    "audio": (
        "input_device",
        "sample_rate",
        "keep_mic_open",
        "preroll_ms",
        "min_duration_s",
        "max_duration_s",
        "sounds",
    ),
    "asr": (
        "asr_model",
        "device",
        "quantization",
        "vad_threshold",
        "vad_min_silence_ms",
        "vad_speech_pad_ms",
    ),
    "cleanup": (
        "llm_enabled",
        "llm_url",
        "llm_model",
        "llm_min_words",
        "llm_timeout_s",
        "llm_keep_alive",
        "llm_think_flag",
    ),
    "output": (
        "paste_method",
        "restore_clipboard",
        "paste_delay_ms",
        "trailing_space",
        "save_history",
    ),
}

# Friendlier TOML names -> Config field names.
_ALIASES: dict[tuple[str, str], str] = {("hotkey", "key"): "hotkey"}


@dataclass
class Config:
    # [hotkey]
    hotkey: str = "right ctrl"
    mode: str = "hold"  # "hold" (push-to-talk) | "toggle" (tap to start, tap to stop)

    # [audio]
    input_device: str | int | None = None  # None = Windows default mic
    sample_rate: int = 16_000
    keep_mic_open: bool = False  # True = zero start latency + pre-roll, but mic indicator stays on
    preroll_ms: int = 300
    min_duration_s: float = 0.3
    max_duration_s: float = 300.0
    sounds: bool = True

    # [asr]
    asr_model: str = "nemo-parakeet-tdt-0.6b-v3"
    device: str = "auto"  # auto | cuda | cpu
    quantization: str = "auto"  # auto (fp32 on GPU, int8 on CPU) | none | int8
    vad_threshold: float = 0.5
    vad_min_silence_ms: int = 600
    vad_speech_pad_ms: int = 200

    # [cleanup]
    llm_enabled: bool = True
    llm_url: str = "http://127.0.0.1:11434"
    llm_model: str = "gemma3:4b"
    llm_min_words: int = 8  # short utterances skip the LLM entirely
    llm_timeout_s: float = 2.5
    llm_keep_alive: int | str = -1  # keep the model resident in VRAM
    llm_think_flag: bool = False  # send "think": false (only for thinking models like qwen3)

    # [output]
    paste_method: str = "clipboard"  # clipboard | type
    restore_clipboard: bool = True
    paste_delay_ms: int = 150
    trailing_space: bool = True
    save_history: bool = False

    # free-form tables
    replacements: dict[str, str] = field(default_factory=dict)
    snippets: dict[str, str] = field(default_factory=dict)

    path: Path | None = None


def load_config(path: str | Path) -> Config:
    path = Path(path)
    cfg = Config(path=path)
    if not path.exists():
        return cfg

    with path.open("rb") as f:
        data = tomllib.load(f)

    valid = {f.name for f in fields(Config)}
    for section, keys in _SECTIONS.items():
        for key, value in data.get(section, {}).items():
            key = _ALIASES.get((section, key), key)
            if key in keys and key in valid:
                setattr(cfg, key, value)

    cfg.replacements = {str(k): str(v) for k, v in data.get("replacements", {}).items()}
    cfg.snippets = {str(k): str(v) for k, v in data.get("snippets", {}).items()}

    if cfg.mode not in ("hold", "toggle"):
        raise ValueError(f"[hotkey] mode must be 'hold' or 'toggle', got {cfg.mode!r}")
    if cfg.mode == "hold" and "+" in cfg.hotkey:
        raise ValueError("hold mode needs a single key (e.g. 'right ctrl'); use toggle mode for combos")
    if cfg.device not in ("auto", "cuda", "cpu"):
        raise ValueError(f"[asr] device must be auto/cuda/cpu, got {cfg.device!r}")
    if cfg.paste_method not in ("clipboard", "type"):
        raise ValueError(f"[output] paste_method must be clipboard/type, got {cfg.paste_method!r}")
    return cfg
