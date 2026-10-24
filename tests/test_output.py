"""insert_text tests with fake keyboard/clipboard (nothing is pasted into the real desktop)."""

from __future__ import annotations

import pytest

from vaani import output


class FakeClipboard:
    def __init__(self, content="old", fail_paste=False):
        self.content, self.fail_paste, self.history = content, fail_paste, []

    def paste(self):
        if self.fail_paste:
            raise RuntimeError("no clipboard")
        return self.content

    def copy(self, text):
        self.content = text
        self.history.append(text)


class FakeKeyboard:
    def __init__(self, clip: FakeClipboard):
        self.clip, self.events = clip, []

    def send(self, keys):
        self.events.append(("send", keys, self.clip.content))  # clipboard at the moment of Ctrl+V

    def write(self, text, delay=0):
        self.events.append(("write", text, delay))


@pytest.fixture
def env(monkeypatch):
    clip = FakeClipboard()
    kb = FakeKeyboard(clip)
    monkeypatch.setattr(output, "pyperclip", clip)
    monkeypatch.setattr(output, "keyboard", kb)
    monkeypatch.setattr(output.time, "sleep", lambda s: None)
    return clip, kb


def test_clipboard_paste_then_restore(env):
    clip, kb = env
    output.insert_text("hello")
    assert kb.events == [("send", "ctrl+v", "hello")]  # text was on the clipboard when pasted
    assert clip.content == "old"  # previous clipboard text restored


def test_no_restore_leaves_dictation_on_clipboard(env):
    clip, _ = env
    output.insert_text("hello", restore=False)
    assert clip.content == "hello" and clip.history == ["hello"]


def test_paste_still_happens_if_clipboard_unreadable(env):
    clip, kb = env
    clip.fail_paste = True
    output.insert_text("hello")
    assert kb.events == [("send", "ctrl+v", "hello")]
    assert clip.content == "hello"  # nothing to restore


def test_type_method_uses_simulated_keystrokes(env):
    clip, kb = env
    output.insert_text("hello", method="type")
    assert kb.events == [("write", "hello", 0)] and clip.history == []


def test_empty_text_is_a_noop(env):
    clip, kb = env
    output.insert_text("")
    assert kb.events == [] and clip.history == []
