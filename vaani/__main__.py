"""Entry point: `python -m vaani [--config PATH] [--file WAV] [--download]`."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .config import load_config

ROOT = Path(__file__).resolve().parent.parent


def _data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    d = Path(base) / "Vaani"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _setup_logging(data_dir: Path, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.FileHandler(data_dir / "vaani.log", encoding="utf-8")]
    if sys.stdout is not None:  # pythonw has no console
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
    )


def _single_instance() -> bool:
    """Windows named mutex so two copies don't fight over the hotkey."""
    if sys.platform != "win32":
        return True
    import ctypes

    ctypes.windll.kernel32.CreateMutexW(None, False, "Local\\VaaniDictation")
    return ctypes.windll.kernel32.GetLastError() != 183  # ERROR_ALREADY_EXISTS


def main() -> None:
    p = argparse.ArgumentParser(prog="vaani", description="Local push-to-talk dictation")
    p.add_argument("--config", default=str(ROOT / "config.toml"))
    p.add_argument("--file", help="transcribe a WAV file, print text + timings, and exit")
    p.add_argument("--download", action="store_true", help="download/cache models and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    data_dir = _data_dir()
    _setup_logging(data_dir, args.verbose)
    cfg = load_config(args.config)

    if args.download or args.file:
        from .stt import Transcriber

        stt = Transcriber(
            cfg.asr_model,
            cfg.device,
            cfg.quantization,
            cfg.vad_threshold,
            cfg.vad_min_silence_ms,
            cfg.vad_speech_pad_ms,
        )
        if args.download:
            print(f"Models cached. ASR runs on: {stt.device}")
            return
        import wave

        import numpy as np

        with wave.open(args.file, "rb") as w:
            sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
            if width != 2:
                sys.exit("--file expects 16-bit PCM WAV")
            pcm = np.frombuffer(w.readframes(n), dtype=np.int16).reshape(-1, ch).mean(axis=1)
        audio = (pcm / 32768.0).astype(np.float32)
        stt.warmup(sr)
        t0 = time.perf_counter()
        text = stt.transcribe(audio, sr)
        dt = time.perf_counter() - t0
        print(text)
        print(
            f"\n[{stt.device}] {len(audio) / sr:.1f}s audio in {dt * 1000:.0f} ms (RTFx {len(audio) / sr / dt:.0f})"
        )
        return

    if not _single_instance():
        logging.getLogger("vaani").warning("Vaani is already running")
        return

    from .app import App

    App(cfg, data_dir).run()


if __name__ == "__main__":
    main()
