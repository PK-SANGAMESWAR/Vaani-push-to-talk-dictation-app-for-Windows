"""Privacy: with every non-loopback socket blocked, the pipeline still works and attempts none.

CLAUDE.md principle 1: the only network call allowed is to Ollama on 127.0.0.1. Models are faked here;
the CI mini-eval (`run_eval --block-network`) repeats the check with the real models and real load.
"""

from __future__ import annotations

import json
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

from evals.netguard import Guard, NetworkBlocked
from vaani.cleanup import Cleaner
from vaani.config import Config
from vaani.pipeline import Pipeline

SR = 16_000
AUDIO = np.ones(SR, np.float32)
SENTENCE = "so um we should ship the pie torch model on friday okay"  # 12 words: above llm_min_words


class FakeStt:
    def transcribe(self, audio, sr, vad=True):
        return SENTENCE


class FakeOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"message": {"content": "We should ship the PyTorch model on Friday."}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def ollama():
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def guard():
    with Guard() as g:
        g.self_test()
        yield g


def test_guard_blocks_and_records_remote_connections_dns_and_udp(guard):
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("example.com", 443), timeout=1)
    with pytest.raises(NetworkBlocked):
        socket.socket().connect(("93.184.216.34", 80))
    with pytest.raises(NetworkBlocked):
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"x", ("8.8.8.8", 53))
    assert guard.attempts == ["dns example.com", "connect 93.184.216.34:80", "sendto 8.8.8.8:53"]


def test_guard_allows_loopback_and_restores_sockets_afterwards(ollama):
    real_connect = socket.socket.connect
    with Guard() as g:
        host, port = ollama.removeprefix("http://").split(":")
        socket.create_connection((host, int(port)), timeout=2).close()
        socket.create_connection(("localhost", int(port)), timeout=2).close()
        assert g.attempts == []
    assert socket.socket.connect is real_connect


def test_pipeline_with_cleanup_works_with_network_blocked(ollama, guard):
    cfg = Config(llm_enabled=True, replacements={"pie torch": "PyTorch"})
    pipeline = Pipeline(cfg, FakeStt(), cleaner=Cleaner(url=ollama, vocabulary=["PyTorch"]))
    r = pipeline.process(AUDIO, SR)
    assert (r.route, r.cleanup_status) == ("cleanup", "ok")
    assert r.text == "We should ship the PyTorch model on Friday."
    assert guard.attempts == []


def test_pipeline_without_ollama_falls_back_to_raw_text_and_stays_local(guard):
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()  # nothing listens here: connection refused, on loopback
    cfg = Config(llm_enabled=True, replacements={"pie torch": "PyTorch"})
    r = Pipeline(cfg, FakeStt(), cleaner=Cleaner(url=f"http://127.0.0.1:{port}")).process(AUDIO, SR)
    assert (r.route, r.cleanup_status) == ("cleanup", "error")
    assert "PyTorch" in r.text  # raw transcript + replacements, never lost
    assert guard.attempts == []


def test_a_remote_ollama_url_is_refused_not_silently_used(guard):
    # Even if someone points cleanup at a remote host, the guard shows the attempt and the pipeline
    # still returns the raw text (fail safe). The config validation test below stops it earlier.
    cfg = Config(llm_enabled=True)
    r = Pipeline(cfg, FakeStt(), cleaner=Cleaner(url="http://ollama.example.com:11434")).process(AUDIO, SR)
    assert r.cleanup_status == "error" and r.text
    assert guard.attempts == ["dns ollama.example.com"]


@pytest.mark.parametrize(
    "url",
    ["http://example.com:11434", "https://api.openai.com", "http://192.168.1.20:11434", "http://10.0.0.5"],
)
def test_config_rejects_non_loopback_llm_url(tmp_path, url):
    from vaani.config import load_config

    p = tmp_path / "c.toml"
    p.write_text(f'[cleanup]\nllm_url = "{url}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="loopback"):
        load_config(p)


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434"])
def test_config_accepts_loopback_llm_url(tmp_path, url):
    from vaani.config import load_config

    p = tmp_path / "c.toml"
    p.write_text(f'[cleanup]\nllm_url = "{url}"\n', encoding="utf-8")
    assert load_config(p).llm_url == url


def test_app_source_contains_no_remote_urls():
    """Static check: the only URL literal in the app code is the local Ollama default."""
    pkg = Path(__file__).resolve().parent.parent / "vaani"
    urls = {
        (f.name, u)
        for f in pkg.glob("*.py")
        for u in re.findall(r"https?://[^\s\"'`)>\]]+", f.read_text(encoding="utf-8"))
    }
    assert {u for _, u in urls if not re.match(r"https?://(127\.0\.0\.1|localhost|\[::1\])", u)} == set()
