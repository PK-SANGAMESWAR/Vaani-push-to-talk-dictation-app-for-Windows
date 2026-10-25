"""Block every non-loopback network connection in this process and record the attempts.

Used by the privacy test (tests/test_privacy.py) and by `run_eval --block-network` to show that the
shipped pipeline never needs the network: not for transcription, not for model loading, and for cleanup
only the local Ollama server. The only traffic allowed is loopback (127.0.0.0/8, ::1, localhost).

Blocked calls raise `NetworkBlocked` (an OSError, like a real connection failure) and are also recorded
in `Guard.attempts`. Code that swallows the error (the cleanup step falls back to raw text on OSError)
still leaves a record, so callers assert `attempts == []`, not just "nothing crashed".
"""

from __future__ import annotations

import ipaddress
import socket
import threading
from typing import Any


class NetworkBlocked(OSError):
    pass


def _is_loopback_host(host: Any) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    if not isinstance(host, str):
        return False
    if host.lower() in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


class Guard:
    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._lock = threading.Lock()
        self._saved: list[tuple[Any, str, Any]] = []

    def _record(self, what: str) -> None:
        with self._lock:
            self.attempts.append(what)

    def _check_addr(self, op: str, address: Any) -> None:
        # AF_UNIX paths are strs and have no host; only (host, port, ...) tuples are network addresses.
        if not isinstance(address, tuple):
            return
        host = address[0]
        if not _is_loopback_host(host):
            what = f"{op} {host}:{address[1] if len(address) > 1 else ''}"
            self._record(what)
            raise NetworkBlocked(f"network blocked: {what}")

    def _patch(self, obj: Any, name: str, replacement: Any) -> None:
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, replacement)

    def install(self) -> Guard:
        guard = self
        real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
        real_sendto = socket.socket.sendto
        real_getaddrinfo = socket.getaddrinfo

        def connect(sock, address):
            guard._check_addr("connect", address)
            return real_connect(sock, address)

        def connect_ex(sock, address):
            guard._check_addr("connect", address)
            return real_connect_ex(sock, address)

        def sendto(sock, data, *args):
            guard._check_addr("sendto", args[-1])
            return real_sendto(sock, data, *args)

        def getaddrinfo(host, *args, **kwargs):
            # Name resolution of anything but a loopback name is itself a network request (DNS).
            if host is not None and not _is_loopback_host(host):
                try:
                    ipaddress.ip_address(host)  # literal IPs need no DNS; connect() judges them
                except ValueError:
                    what = f"dns {host}"
                    guard._record(what)
                    raise NetworkBlocked(f"network blocked: {what}") from None
            return real_getaddrinfo(host, *args, **kwargs)

        self._patch(socket.socket, "connect", connect)
        self._patch(socket.socket, "connect_ex", connect_ex)
        self._patch(socket.socket, "sendto", sendto)
        self._patch(socket, "getaddrinfo", getaddrinfo)
        return self

    def self_test(self) -> None:
        """Prove the guard is live: a connect to a documentation-range address (RFC 5737) must be refused
        before any packet is sent. The probe is not counted as an attempt."""
        before = len(self.attempts)
        try:
            socket.socket().connect(("203.0.113.1", 9))
        except NetworkBlocked:
            del self.attempts[before:]
            return
        except OSError:
            pass
        raise RuntimeError("netguard is not blocking connections")

    def uninstall(self) -> None:
        while self._saved:
            obj, name, original = self._saved.pop()
            setattr(obj, name, original)

    def __enter__(self) -> Guard:
        return self.install()

    def __exit__(self, *exc: object) -> None:
        self.uninstall()
