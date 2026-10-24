"""Peak GPU memory (NVML, device-wide) and RAM (this process) during an eval run.

Device-wide numbers include every process on the GPU (desktop, browser, Ollama), so the report
records the baseline before model load and the peak during the run. Per-process GPU memory is added
only when the driver reports it (often unavailable under Windows WDDM).
"""

from __future__ import annotations

import contextlib
import os
import threading

MB = 1024 * 1024


class PeakSampler:
    def __init__(self, interval_s: float = 0.05, gpu_index: int = 0):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None
        self.gpu_error: str | None = None
        self.baseline_used = self.peak_used = self.total = None
        self.peak_process: int | None = None
        self.peak_rss = 0

    def start(self) -> PeakSampler:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.total, self.baseline_used, self.peak_used = mem.total, mem.used, mem.used
        except Exception as e:  # no NVIDIA GPU / driver: RAM-only report
            self.gpu_error = f"{type(e).__name__}: {e}"
            self._nvml = None
        self._sample()
        self._thread = threading.Thread(target=self._run, name="peak-sampler", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def _sample(self) -> None:
        import psutil

        mi = psutil.Process().memory_info()
        self.peak_rss = max(self.peak_rss, mi.rss)
        if self._nvml is None:
            return
        try:
            used = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used
            self.peak_used = max(self.peak_used, used)
            for proc in self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                if proc.pid == os.getpid() and proc.usedGpuMemory:
                    self.peak_process = max(self.peak_process or 0, proc.usedGpuMemory)
        except Exception:
            pass

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(2)
        self._sample()
        import psutil

        mi = psutil.Process().memory_info()
        peak_wset = getattr(mi, "peak_wset", None)  # Windows only: true OS-tracked peak
        out = {
            "ram_peak_mb": round((peak_wset or self.peak_rss) / MB, 1),
            "ram_peak_source": "peak_wset" if peak_wset else "sampled_rss",
            "gpu": None,
        }
        if self._nvml is not None:
            out["gpu"] = {
                "scope": "device-wide (all processes)",
                "total_mb": round(self.total / MB, 1),
                "baseline_used_mb": round(self.baseline_used / MB, 1),
                "peak_used_mb": round(self.peak_used / MB, 1),
                "delta_mb": round((self.peak_used - self.baseline_used) / MB, 1),
                "process_peak_mb": round(self.peak_process / MB, 1) if self.peak_process else None,
                "sample_interval_ms": int(self.interval_s * 1000),
            }
            with contextlib.suppress(Exception):
                self._nvml.nvmlShutdown()
        else:
            out["gpu_error"] = self.gpu_error
        return out
