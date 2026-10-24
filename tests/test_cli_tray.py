"""CLI entry point and tray tests (no model, no real tray icon shown)."""

from __future__ import annotations

import sys
import types
import wave

import numpy as np
import pytest

from vaani import __main__ as cli
from vaani import tray


def write_wav(path, samples: np.ndarray, channels: int = 1, width: int = 2, sr: int = 16_000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_setup_logging", lambda *a: None)
    monkeypatch.setattr(cli, "_data_dir", lambda: tmp_path)
    created: list = []

    class FakeTranscriber:
        device = "cuda"

        def __init__(self, *args):
            self.args = args
            self.heard = None
            created.append(self)

        def warmup(self, sr):
            pass

        def transcribe(self, audio, sr):
            self.heard = (audio, sr)
            return "hello from file"

    stt = types.ModuleType("vaani.stt")
    stt.Transcriber = FakeTranscriber
    monkeypatch.setitem(sys.modules, "vaani.stt", stt)
    return created


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["vaani", *argv])
    cli.main()


def test_file_mode_transcribes_wav_and_prints_timing(cli_env, tmp_path, monkeypatch, capsys):
    wav = tmp_path / "a.wav"
    write_wav(wav, (np.ones(16_000) * 16384).astype(np.int16))
    run_cli(monkeypatch, "--file", str(wav))
    out = capsys.readouterr().out
    assert "hello from file" in out and "RTFx" in out and "1.0s audio" in out
    audio, sr = cli_env[0].heard
    assert sr == 16_000 and audio.dtype == np.float32 and abs(audio[0] - 0.5) < 1e-6


def test_file_mode_downmixes_stereo(cli_env, tmp_path, monkeypatch, capsys):
    wav = tmp_path / "s.wav"
    stereo = np.stack([np.full(1600, 8192), np.full(1600, 16384)], axis=1).astype(np.int16)
    write_wav(wav, stereo, channels=2)
    run_cli(monkeypatch, "--file", str(wav))
    audio, _ = cli_env[0].heard
    assert audio.shape == (1600,) and abs(audio[0] - 0.375) < 1e-6


def test_file_mode_rejects_non_16bit(cli_env, tmp_path, monkeypatch):
    wav = tmp_path / "b.wav"
    write_wav(wav, np.zeros(100, dtype=np.uint8), width=1)
    with pytest.raises(SystemExit):
        run_cli(monkeypatch, "--file", str(wav))


def test_download_mode_loads_models_and_exits(cli_env, monkeypatch, capsys):
    run_cli(monkeypatch, "--download")
    assert "ASR runs on: cuda" in capsys.readouterr().out


def test_config_flag_is_honoured(cli_env, tmp_path, monkeypatch):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[asr]\ndevice = "cpu"\nvad_threshold = 0.7\n')
    run_cli(monkeypatch, "--download", "--config", str(cfg))
    assert cli_env[0].args[1] == "cpu" and cli_env[0].args[3] == 0.7


def test_second_instance_exits_without_starting_app(cli_env, monkeypatch):
    started = []
    app_mod = types.ModuleType("vaani.app")
    app_mod.App = lambda *a: types.SimpleNamespace(run=lambda: started.append(1))
    monkeypatch.setitem(sys.modules, "vaani.app", app_mod)
    monkeypatch.setattr(cli, "_single_instance", lambda: False)
    run_cli(monkeypatch)
    assert started == []
    monkeypatch.setattr(cli, "_single_instance", lambda: True)
    run_cli(monkeypatch)
    assert started == [1]


def test_data_dir_is_created_under_localappdata(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    d = cli._data_dir()
    assert d == tmp_path / "Vaani" and d.is_dir()


# ---------- tray ----------


def test_every_app_state_has_colour_and_label():
    states = {"loading", "idle", "recording", "busy", "error"}  # states App can set
    assert set(tray.COLORS) == set(tray.LABELS) == states


def test_icon_image_is_rgba_64():
    img = tray._icon(tray.COLORS["idle"])
    assert img.size == (64, 64) and img.mode == "RGBA"


def test_tray_menu_actions(tmp_path, monkeypatch):
    stopped = []
    app = types.SimpleNamespace(
        cfg=types.SimpleNamespace(llm_enabled=True, path=tmp_path / "config.toml"),
        state="idle",
        device="cuda",
        data_dir=tmp_path,
        shutdown=lambda: stopped.append("app"),
    )
    t = tray.Tray(app)
    t._toggle_cleanup()
    assert app.cfg.llm_enabled is False
    t._toggle_cleanup()
    assert app.cfg.llm_enabled is True

    opened: list = []
    monkeypatch.setattr(tray.os, "startfile", opened.append, raising=False)
    t._open_config()
    t._open_logs()
    assert opened == [app.cfg.path, tmp_path]

    t.icon.stop = lambda: stopped.append("icon")
    t._quit()
    assert stopped == ["app", "icon"]
    t.set_state("recording")
    assert t.icon.title == "Vaani: Listening…"
