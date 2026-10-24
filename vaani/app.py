"""Main application: hotkey -> record -> VAD + Parakeet -> (Ollama cleanup) -> paste."""

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import Config
from .pipeline import Pipeline, make_cleaner, make_transcriber
from .textproc import TextProcessor, word_count

log = logging.getLogger(__name__)


def _beep(freq: int, ms: int) -> None:
    def run():
        try:
            import winsound

            winsound.Beep(freq, ms)
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


class App:
    def __init__(self, cfg: Config, data_dir: Path):
        self.cfg = cfg
        self.data_dir = data_dir
        self.state = "loading"
        self.device = "…"
        self.text = TextProcessor(cfg.replacements, cfg.snippets)
        self.jobs: queue.Queue[np.ndarray | None] = queue.Queue()
        self.ctrl: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._recording = False
        self._timer: threading.Timer | None = None
        self.pipeline = Pipeline(cfg, None, self.text, None)
        self.recorder = None
        self.tray = None

    # Models live on the pipeline (shared with the eval suite); these keep App's attribute API.
    @property
    def stt(self):
        return self.pipeline.stt

    @stt.setter
    def stt(self, value) -> None:
        self.pipeline.stt = value

    @property
    def cleaner(self):
        return self.pipeline.cleaner

    @cleaner.setter
    def cleaner(self, value) -> None:
        self.pipeline.cleaner = value

    # ---------- lifecycle ----------

    def run(self) -> None:
        from .tray import Tray

        self.tray = Tray(self)
        threading.Thread(target=self._init, name="init", daemon=True).start()
        threading.Thread(target=self._worker, name="worker", daemon=True).start()
        self.tray.run()  # blocks on the main thread until Quit

    def _init(self) -> None:
        try:
            import keyboard

            from .audio import Recorder

            c = self.cfg
            self.stt = make_transcriber(c)
            self.device = self.stt.device
            self.recorder = Recorder(c.sample_rate, c.input_device, c.keep_mic_open, c.preroll_ms)
            self.stt.warmup(self.recorder.sample_rate)

            self.cleaner = make_cleaner(c, self.text.vocabulary)
            if c.llm_enabled:
                threading.Thread(target=self.cleaner.prewarm, name="ollama-warm", daemon=True).start()

            # Hook callbacks only enqueue; a dedicated thread does the work. Windows silently
            # drops low-level hooks that take too long, so nothing slow may run on the hook thread.
            threading.Thread(target=self._control_loop, name="control", daemon=True).start()
            if c.mode == "hold":
                keyboard.on_press_key(c.hotkey, lambda _e: self.ctrl.put("start"))
                keyboard.on_release_key(c.hotkey, lambda _e: self.ctrl.put("stop"))
            else:
                keyboard.add_hotkey(c.hotkey, lambda: self.ctrl.put("toggle"))

            self._set_state("idle")
            verb = "Hold" if c.mode == "hold" else "Press"
            self._notify(f"Ready on {self.device.upper()}. {verb} {c.hotkey} to dictate.")
        except Exception as e:
            log.exception("startup failed")
            self._set_state("error")
            self._notify(f"Startup failed: {e}")

    def shutdown(self) -> None:
        try:
            import keyboard

            keyboard.unhook_all()
        except Exception:
            pass
        if self.recorder:
            self.recorder.close()
        self.ctrl.put(None)
        self.jobs.put(None)

    # ---------- recording control (runs on the "control" thread, in hotkey order) ----------

    def _control_loop(self) -> None:
        actions = {"start": self._start, "stop": self._stop, "toggle": self._toggle}
        while True:
            action = self.ctrl.get()
            if action is None:
                return
            try:
                actions[action]()
            except Exception:
                log.exception("control action %s failed", action)

    def _toggle(self) -> None:
        if self._recording:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        with self._lock:
            if self._recording or self.recorder is None:  # key auto-repeat fires press repeatedly
                return
            self._recording = True
        try:
            self.recorder.start()
        except Exception:
            log.exception("could not open microphone")
            self._recording = False
            self._notify("Could not open the microphone")
            return
        if self.cfg.sounds:
            _beep(880, 40)
        self._set_state("recording")
        self._timer = threading.Timer(self.cfg.max_duration_s, lambda: self.ctrl.put("stop"))
        self._timer.daemon = True
        self._timer.start()

    def _stop(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
        if self._timer:
            self._timer.cancel()
        audio = self.recorder.stop()
        if self.cfg.sounds:
            _beep(660, 40)
        duration = audio.size / self.recorder.sample_rate
        if duration < self.cfg.min_duration_s:
            log.info("ignored %.2fs tap", duration)
            self._set_state("idle")
            return
        self._set_state("busy")
        self.jobs.put(audio)

    # ---------- processing ----------

    def _worker(self) -> None:
        while True:
            audio = self.jobs.get()
            if audio is None:
                return
            try:
                self._process(audio)
            except Exception:
                log.exception("processing failed")
            finally:
                if self.jobs.empty() and not self._recording:
                    self._set_state("idle")

    def _process(self, audio: np.ndarray) -> None:
        from .output import insert_text

        c = self.cfg
        sr = self.recorder.sample_rate
        r = self.pipeline.process(audio, sr)
        if not r.raw:
            log.info("no speech detected (%.1fs audio)", r.audio_s)
            return

        final = r.text
        if c.trailing_space and not final.endswith(("\n", " ")):
            final += " "
        insert_text(final, c.paste_method, c.restore_clipboard, c.paste_delay_ms)
        log.info(
            "audio %.1fs | asr %.0f ms | total %.0f ms | %d words",
            r.audio_s,
            r.asr_ms,
            r.total_ms,
            word_count(final),
        )
        if c.save_history:
            with (self.data_dir / "history.txt").open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{final.strip()}\n")

    # ---------- helpers ----------

    def _set_state(self, state: str) -> None:
        self.state = state
        if self.tray:
            self.tray.set_state(state)

    def _notify(self, msg: str) -> None:
        log.info(msg)
        if self.tray:
            self.tray.notify(msg)
