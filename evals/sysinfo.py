"""Provenance recorded in every eval report: code, config, models, runtime, hardware."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from vaani.config import Config

ROOT = Path(__file__).resolve().parent.parent


def rel_path(path: str | Path | None) -> str | None:
    """Repo-relative posix path when possible, so committed reports don't leak local user paths."""
    if path is None:
        return None
    p = Path(path).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.name


def config_hash(cfg: Config) -> str:
    """sha256 of the effective config (after CLI overrides). Independent of file path and key order."""
    d = dataclasses.asdict(cfg)
    d.pop("path", None)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_info() -> dict[str, Any]:
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def gpu_info() -> dict[str, Any] | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(h)
            driver = pynvml.nvmlSystemGetDriverVersion()
            return {
                "name": name.decode() if isinstance(name, bytes) else name,
                "driver": driver.decode() if isinstance(driver, bytes) else driver,
                "memory_total_mb": round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**20),
                "cuda_driver_version": pynvml.nvmlSystemGetCudaDriverVersion(),
            }
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def ollama_info(url: str, model: str) -> dict[str, Any]:
    """Ollama version + model digest (localhost only, per CLAUDE.md)."""
    import urllib.request

    info: dict[str, Any] = {"url": url, "model": model}
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/version", timeout=3) as r:
            info["version"] = json.loads(r.read()).get("version")
        req = urllib.request.Request(
            url.rstrip("/") + "/api/show",
            data=json.dumps({"model": model}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            show = json.loads(r.read())
        details = show.get("details", {})
        info.update(
            {
                "quantization": details.get("quantization_level"),
                "parameter_size": details.get("parameter_size"),
                "license_head": (show.get("license") or "").strip().splitlines()[:1],
            }
        )
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=3) as r:
            for m in json.loads(r.read()).get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    info["digest"] = m.get("digest")
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def collect(cfg: Config, stt: Any, cleanup_on: bool) -> dict[str, Any]:
    import psutil

    from .normalizer import SOURCE_COMMIT, SOURCE_REPO

    providers = None
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except Exception:
        pass
    return {
        "git": git_info(),
        "config": {
            "path": rel_path(cfg.path),
            "sha256": config_hash(cfg),
            "values": {k: v for k, v in dataclasses.asdict(cfg).items() if k != "path"},
        },
        "asr": {
            "model": cfg.asr_model,
            "device": getattr(stt, "device", None),
            "quantization": getattr(stt, "quantization", None),
            "vad": {
                "model": "silero",
                "threshold": cfg.vad_threshold,
                "min_silence_ms": cfg.vad_min_silence_ms,
                "speech_pad_ms": cfg.vad_speech_pad_ms,
            },
            "available_providers": providers,
        },
        "cleanup": ollama_info(cfg.llm_url, cfg.llm_model) if cleanup_on else None,
        "normalizer": {"name": "EnglishTextNormalizer", "repo": SOURCE_REPO, "commit": SOURCE_COMMIT},
        "versions": {
            p: _version(p)
            for p in ("onnxruntime-gpu", "onnxruntime", "onnx-asr", "numpy", "jiwer", "soundfile")
        }
        | {"python": platform.python_version()},
        "hardware": {
            "gpu": gpu_info(),
            "cpu": cpu_name(),
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        },
        "os": platform.platform(),
    }
