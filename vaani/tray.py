"""System tray icon showing state, with a small menu."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import pystray
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from .app import App

COLORS = {
    "loading": (90, 140, 230),
    "idle": (70, 175, 110),
    "recording": (225, 60, 60),
    "busy": (235, 170, 40),
    "error": (130, 130, 130),
}
LABELS = {
    "loading": "Loading models…",
    "idle": "Ready",
    "recording": "Listening…",
    "busy": "Transcribing…",
    "error": "Error (see log)",
}


def _icon(color: tuple[int, int, int]) -> Image.Image:
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((2, 2, size - 2, size - 2), fill=color)
    # simple microphone glyph
    d.rounded_rectangle((24, 12, 40, 38), radius=8, fill="white")
    d.arc((18, 22, 46, 46), start=0, end=180, fill="white", width=3)
    d.line((32, 46, 32, 52), fill="white", width=3)
    d.line((24, 52, 40, 52), fill="white", width=3)
    return img


class Tray:
    def __init__(self, app: App):
        self.app = app
        self._images = {state: _icon(c) for state, c in COLORS.items()}
        self.icon = pystray.Icon(
            "vaani",
            self._images["loading"],
            "Vaani: " + LABELS["loading"],
            menu=pystray.Menu(
                pystray.MenuItem(lambda _: f"Vaani · {LABELS[self.app.state]}", None, enabled=False),
                pystray.MenuItem(lambda _: f"ASR on {self.app.device}", None, enabled=False),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(
                    "AI cleanup (Ollama)",
                    self._toggle_cleanup,
                    checked=lambda _: self.app.cfg.llm_enabled,
                ),
                pystray.MenuItem("Open config", self._open_config),
                pystray.MenuItem("Open log folder", self._open_logs),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self._quit),
            ),
        )

    def set_state(self, state: str) -> None:
        self.icon.icon = self._images[state]
        self.icon.title = "Vaani: " + LABELS[state]
        self.icon.update_menu()

    def notify(self, message: str) -> None:
        with contextlib.suppress(Exception):  # notifications are best-effort
            self.icon.notify(message, "Vaani")

    def _toggle_cleanup(self, *_):
        self.app.cfg.llm_enabled = not self.app.cfg.llm_enabled

    def _open_config(self, *_):
        if self.app.cfg.path:
            os.startfile(self.app.cfg.path)  # type: ignore[attr-defined]  # Windows only

    def _open_logs(self, *_):
        os.startfile(self.app.data_dir)  # type: ignore[attr-defined]

    def _quit(self, *_):
        self.app.shutdown()
        self.icon.stop()

    def run(self) -> None:
        self.icon.run()
