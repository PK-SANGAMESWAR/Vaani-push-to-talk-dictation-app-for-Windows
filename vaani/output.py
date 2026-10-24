"""Insert text into whatever window has focus."""

from __future__ import annotations

import logging
import time

import keyboard
import pyperclip

log = logging.getLogger(__name__)


def insert_text(text: str, method: str = "clipboard", restore: bool = True, delay_ms: int = 150) -> None:
    if not text:
        return
    if method == "type":
        keyboard.write(text, delay=0)
        return

    previous: str | None = None
    if restore:
        try:
            previous = pyperclip.paste()  # text only; images/files on the clipboard are not preserved
        except Exception:
            previous = None

    pyperclip.copy(text)
    time.sleep(0.02)
    keyboard.send("ctrl+v")

    if restore and previous is not None:
        time.sleep(delay_ms / 1000)  # give the target app time to read the clipboard
        try:
            pyperclip.copy(previous)
        except Exception:
            log.debug("could not restore clipboard")
