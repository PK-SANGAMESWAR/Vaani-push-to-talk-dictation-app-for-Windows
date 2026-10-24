"""Text targets the e2e test dictates into: a Notepad window, or a <textarea> in Edge. Windows only.

Each target can focus itself, clear its text, and read it back through UI Automation (pywinauto), so
the harness sees exactly what a user would see after Vaani pastes.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path


def _send(keys: str) -> None:
    import keyboard

    keyboard.send(keys)


class NotepadTarget:
    name = "notepad"

    def __init__(self) -> None:
        from pywinauto import Desktop

        # A uniquely named empty file, so we attach to our own window (Win11 Notepad is one process
        # for all windows; never kill it or we'd close the user's notes).
        self.dir = Path(tempfile.mkdtemp(prefix="vaani-e2e-"))
        self.file = self.dir / f"vaani-e2e-{int(time.time())}.txt"
        self.file.write_text("", encoding="utf-8")
        subprocess.Popen(["notepad.exe", str(self.file)])
        title = self.file.stem
        deadline = time.time() + 15
        while time.time() < deadline:
            wins = [w for w in Desktop(backend="uia").windows() if title in w.window_text()]
            if wins:
                self.window = wins[0]
                break
            time.sleep(0.3)
        else:
            raise RuntimeError("Notepad window did not appear")
        self.doc = self._find_doc()

    def _find_doc(self):
        for ct in ("Document", "Edit"):
            found = self.window.descendants(control_type=ct)
            if found:
                return found[0]
        raise RuntimeError("could not find Notepad's text area via UI Automation")

    def focus(self) -> None:
        self.window.set_focus()
        self.doc.set_focus()

    def clear(self) -> None:
        self.focus()
        _send("ctrl+a")
        _send("delete")
        time.sleep(0.05)

    def read(self) -> str:
        try:
            return self.doc.iface_text.DocumentRange.GetText(-1)
        except Exception:
            return self.doc.window_text()

    def close(self) -> None:
        try:
            self.clear()
            _send("ctrl+s")  # saved empty file: closing the tab won't prompt
            time.sleep(0.3)
            _send("ctrl+w")
        except Exception:
            pass


TEXTAREA_HTML = """<!doctype html><meta charset="utf-8"><title>vaani-e2e-textarea</title>
<textarea id="t" aria-label="vaani-e2e" style="width:95vw;height:80vh" autofocus></textarea>"""


class BrowserTarget:
    """A local page with one <textarea>, opened in a new Edge window (no network)."""

    name = "browser"

    def __init__(self) -> None:
        from pywinauto import Desktop

        self.dir = Path(tempfile.mkdtemp(prefix="vaani-e2e-"))
        page = self.dir / "textarea.html"
        page.write_text(TEXTAREA_HTML, encoding="utf-8")
        subprocess.Popen(["cmd", "/c", "start", "", "msedge", "--new-window", page.as_uri()])
        deadline = time.time() + 20
        while time.time() < deadline:
            wins = [w for w in Desktop(backend="uia").windows() if "vaani-e2e-textarea" in w.window_text()]
            if wins:
                self.window = wins[0]
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Edge window with the test page did not appear")
        time.sleep(1.0)
        edits = [e for e in self.window.descendants(control_type="Edit") if "vaani-e2e" in e.window_text()]
        edits = edits or self.window.descendants(control_type="Edit")
        if not edits:
            raise RuntimeError("could not find the textarea via UI Automation")
        self.box = edits[-1]

    def focus(self) -> None:
        self.window.set_focus()
        self.box.set_focus()

    def clear(self) -> None:
        self.focus()
        _send("ctrl+a")
        _send("delete")
        time.sleep(0.05)

    def read(self) -> str:
        try:
            return self.box.iface_value.CurrentValue
        except Exception:
            return self.box.get_value() or ""

    def close(self) -> None:
        try:
            self.focus()
            _send("ctrl+w")
        except Exception:
            pass


TARGETS = {"notepad": NotepadTarget, "browser": BrowserTarget}
