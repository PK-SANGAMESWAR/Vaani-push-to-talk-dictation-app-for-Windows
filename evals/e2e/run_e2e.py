"""End-to-end Windows test: the real Vaani app, a virtual mic, a real text box.

    python -m evals.e2e.run_e2e                                   # dictation+safety+negatives into Notepad
    python -m evals.e2e.run_e2e --target both --limit 10           # Notepad and an Edge <textarea>
    python -m evals.e2e.run_e2e --soak 1000 --label soak           # 1000 dictations: crashes, memory, drops

For every clip: focus the target and clear it, press the hotkey (SendInput), play the WAV into
VB-Audio Virtual Cable ("CABLE Input"), which Vaani records as its mic ("CABLE Output"), release the
hotkey, then poll the target's text through UI Automation until it settles. Measures:

- release-to-paste: key release -> last change of the target's text (includes Vaani's release tail,
  ASR, cleanup, the paste delay, and the target app rendering the text). With streaming on, phrases
  can land before release; if nothing changes after release, the latency is 0.
- what the user got: the target's text vs the reference (WER, strict WER vs intended text) and, for
  silence/noise clips, whether anything was pasted at all.
- dropped hotkeys: a press/release with no sign in Vaani's log that it handled the dictation.

Vaani runs as a child process with a temporary config (your config.toml plus the overrides below) and
its own log folder, so your settings and %LOCALAPPDATA%\\Vaani are untouched. See evals/e2e/README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import metrics, sysinfo
from ..datasets.common import SAMPLE_RATE, Clip, load_audio, manifest_path, read_manifest
from ..run_eval import _score

ROOT = Path(__file__).resolve().parents[2]
EVENT = re.compile(
    r"vaani\.(?:app|streaming): (?P<kind>audio [\d.]+s \| asr|no speech detected|ignored [\d.]+s tap|"
    r"processing failed|phrase \d+:)"
)
READY = re.compile(r"Ready on (\w+)|Startup failed: (.*)")


# ---------- app process ----------


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return repr(v)
    return json.dumps(str(v), ensure_ascii=False)  # TOML basic strings use JSON-compatible escapes


def write_config(src: Path, dst: Path, overrides: dict[str, dict[str, Any]]) -> dict[str, Any]:
    data = tomllib.loads(src.read_text(encoding="utf-8"))
    for section, values in overrides.items():
        data.setdefault(section, {}).update({k: v for k, v in values.items() if v is not None})
    lines = []
    for section, values in data.items():
        lines.append(f"[{section}]")
        lines += [f"{json.dumps(k)} = {_toml_value(v)}" for k, v in values.items()]
        lines.append("")
    dst.write_text("\n".join(lines), encoding="utf-8")
    return data


class VaaniProcess:
    def __init__(self, config: Path, workdir: Path, startup_timeout_s: float = 180):
        self.workdir = workdir
        self.log = workdir / "Vaani" / "vaani.log"
        env = os.environ | {"LOCALAPPDATA": str(workdir), "PYTHONIOENCODING": "utf-8"}
        self.out = (workdir / "stdout.txt").open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "vaani", "--config", str(config)],
            cwd=ROOT,
            env=env,
            stdout=self.out,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + startup_timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = (workdir / "stdout.txt").read_text(encoding="utf-8", errors="replace")[-800:]
                if "already running" in tail:
                    raise SystemExit(
                        "Vaani is already running. Quit it from the tray, then rerun the e2e test."
                    )
                raise SystemExit(f"Vaani exited during startup (code {self.proc.returncode}):\n{tail}")
            m = READY.search(self._text())
            if m and m.group(2):
                self.stop()
                raise SystemExit(f"Vaani startup failed: {m.group(2)}")
            if m:
                self.device = m.group(1).lower()
                return
            time.sleep(0.25)
        self.stop()
        raise SystemExit("Vaani did not become ready in time; see " + str(self.log))

    def _text(self) -> str:
        try:
            return self.log.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def log_size(self) -> int:
        try:
            return self.log.stat().st_size
        except FileNotFoundError:
            return 0

    def events_since(self, offset: int) -> list[str]:
        with self.log.open("rb") as f:
            f.seek(offset)
            new = f.read().decode("utf-8", errors="replace")
        return [m.group("kind") for m in EVENT.finditer(new)]

    def alive(self) -> bool:
        return self.proc.poll() is None

    def memory(self) -> dict[str, float]:
        import psutil

        p = psutil.Process(self.proc.pid)
        mi = p.memory_info()
        return {"rss_mb": mi.rss / 2**20, "private_mb": getattr(mi, "private", 0) / 2**20,
                "handles": p.num_handles(), "threads": p.num_threads()}  # fmt: skip

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.out.close()


# ---------- audio out ----------


def resolve_output_device(name: str) -> int:
    """Partial name -> one output device index, preferring the default host API (as Vaani does for mics)."""
    import sounddevice as sd

    matches = [
        i for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0 and name.lower() in d["name"].lower()
    ]  # fmt: skip
    if not matches:
        raise SystemExit(
            f"no playback device matches {name!r}. Install VB-Audio Virtual Cable (see evals/e2e/README.md) "
            "or pass --play with another output device."
        )
    api = sd.default.hostapi
    return next((i for i in matches if sd.query_devices(i)["hostapi"] == api), matches[0])


# ---------- one dictation ----------


def dictate(app: VaaniProcess, target, audio: np.ndarray, play_dev: int, a: argparse.Namespace) -> dict:
    """One dictation; if another window stole focus, redo it once (the first attempt is recorded)."""
    d = _dictate(app, target, audio, play_dev, a)
    if not d["focus_ok"]:
        time.sleep(1.0)
        d = _dictate(app, target, audio, play_dev, a) | {"focus_retried": True}
    return d


def _dictate(app: VaaniProcess, target, audio: np.ndarray, play_dev: int, a: argparse.Namespace) -> dict:
    import keyboard
    import sounddevice as sd

    target.clear()
    focus_at_start = target.focus()
    before = target.read()
    offset = app.log_size()
    keyboard.press(a.hotkey)
    time.sleep(a.lead_ms / 1000)  # Vaani opens the mic on press; give it time before speech starts
    sd.play(audio, SAMPLE_RATE, device=play_dev)
    sd.wait()
    time.sleep(a.hold_tail_ms / 1000)
    keyboard.release(a.hotkey)
    t_rel = time.perf_counter()

    text, last_change, reads = before, None, []
    events: list[str] = []
    deadline = t_rel + a.timeout_s
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        now = target.read()
        reads.append((time.perf_counter() - t0) * 1000)
        if now != text:
            text, last_change = now, time.perf_counter()
        events = app.events_since(offset)
        final_seen = any(not e.startswith("phrase") for e in events)
        settled = last_change is None or time.perf_counter() - last_change >= a.settle_ms / 1000
        if final_seen and settled and (last_change is not None or time.perf_counter() - t_rel > 1.0):
            break  # Vaani finished, and the text has been stable for settle_ms
        if not app.alive():
            break
        time.sleep(0.01)
    focus_at_end = target.has_focus()  # False: another window took focus, the paste went elsewhere
    handled = bool(events)  # a final event, or streamed phrases (a silent tail after them logs nothing)
    return {
        "pasted": text.replace("\r\n", "\n").replace("\r", "\n").strip(),
        "release_to_paste_ms": (last_change - t_rel) * 1000 if last_change else (0.0 if text else None),
        "events": events,
        "handled": handled,
        "focus_ok": focus_at_start and focus_at_end,
        "read_ms_p50": float(np.median(reads)) if reads else None,
    }


# ---------- runs ----------


def _clips(a: argparse.Namespace) -> tuple[list[Clip], list[dict]]:
    paths = [Path(m) for m in a.manifest] or [manifest_path(n.strip()) for n in a.datasets.split(",")]
    clips, manifests = [], []
    for p in paths:
        meta, cs = read_manifest(p)
        if not cs:
            raise SystemExit(f"manifest {p} has no clips (empty file?); restore it from git or rerun prepare")
        cs = cs[: a.limit] if a.limit else cs
        clips += cs
        manifests.append({"path": sysinfo.rel_path(p), "n_used": len(cs)})
    return clips, manifests


def score_rows(rows: list[dict]) -> dict[str, Any]:
    speech = [r for r in rows if r["wer_raw"][1] > 0]
    silent = [r for r in rows if r["wer_raw"][1] == 0]
    lat = [r["release_to_paste_ms"] for r in speech if r["release_to_paste_ms"] is not None]
    strict = [r for r in rows if "strict_final" in r]
    out = {
        "n": len(rows),
        "wer": metrics.rate_with_ci([metrics.Edits(*r["wer_final"]) for r in speech]),
        "strict_wer": metrics.rate_with_ci([metrics.Edits(*r["strict_final"]) for r in strict]),
        "clean_wer": metrics.rate_with_ci([metrics.Edits(*r["clean_final"]) for r in strict]),
        "release_to_paste_ms": metrics.latency_stats(lat),
        "speech_with_nothing_pasted": sum(1 for r in speech if not r["pasted"]),
        "silence_pasted": {"k": sum(1 for r in silent if r["pasted"]), "n": len(silent)},
        "dropped_hotkeys": sum(1 for r in rows if not r["handled"]),
        "focus_lost_retried": sum(1 for r in rows if r.get("focus_retried")),
        "focus_lost_final": sum(1 for r in rows if not r.get("focus_ok", True)),
        "ui_read_ms_p50": float(np.median([r["read_ms_p50"] for r in rows if r["read_ms_p50"]] or [0])),
    }
    by_bucket: dict[str, list[float]] = {}
    for r in speech:
        if r["release_to_paste_ms"] is not None:
            b = "<5 s" if r["duration_s"] < 5 else "5-15 s" if r["duration_s"] < 15 else ">=15 s"
            by_bucket.setdefault(b, []).append(r["release_to_paste_ms"])
    out["latency_by_length"] = {k: metrics.latency_stats(v) for k, v in sorted(by_bucket.items())}
    return out


def run_accuracy(app, targets, clips, play_dev, a) -> dict[str, Any]:
    results = {}
    for tname in targets:
        from .targets import TARGETS

        target = TARGETS[tname]()
        rows = []
        try:
            for i, clip in enumerate(clips, 1):
                d = dictate(app, target, load_audio(clip.path), play_dev, a)
                rows.append(
                    {"id": clip.id, "dataset": clip.dataset, "category": clip.category, "ref": clip.ref,
                     "duration_s": clip.duration_s} | d | _score(clip, d["pasted"], d["pasted"])
                )  # fmt: skip
                print(
                    f"  [{tname}] {i}/{len(clips)} {clip.id}: {d['release_to_paste_ms'] or 0:.0f} ms",
                    file=sys.stderr,
                )
                if not app.alive():
                    print("  Vaani exited!", file=sys.stderr)
                    break
        finally:
            target.close()
        results[tname] = {"summary": score_rows(rows), "clips": rows}
    return results


def run_soak(app, clips, play_dev, a) -> dict[str, Any]:
    from .targets import TARGETS

    target = TARGETS[a.target if a.target != "both" else "notepad"]()
    audio = {c.id: load_audio(c.path) for c in clips}
    samples, failures = [], []
    stats = {"done": 0, "dropped": 0, "speech_no_text": 0, "silence_pasted": 0}
    t0 = time.time()
    try:
        for i in range(a.soak):
            clip = clips[i % len(clips)]
            d = dictate(app, target, audio[clip.id], play_dev, a)
            stats["done"] += 1
            stats["dropped"] += not d["handled"]
            if clip.ref and not d["pasted"]:
                stats["speech_no_text"] += 1
            if not clip.ref and d["pasted"]:
                stats["silence_pasted"] += 1
            if not d["handled"] or (clip.ref and not d["pasted"]):
                failures.append({"i": i, "id": clip.id, "events": d["events"]})
            if i % a.mem_every == 0 or i == a.soak - 1:
                if not app.alive():
                    break
                samples.append({"i": i, "t_s": round(time.time() - t0, 1)} | app.memory())
                print(f"  soak {i + 1}/{a.soak} rss {samples[-1]['rss_mb']:.0f} MB", file=sys.stderr)
            if not app.alive():
                break
    finally:
        target.close()
    crashed = not app.alive()
    growth = None
    warm = [s for s in samples if s["i"] >= a.soak // 10]  # skip warm-up (caches, CUDA arenas)
    if len(warm) >= 3:
        x = np.array([s["i"] for s in warm], dtype=float)
        for key in ("rss_mb", "private_mb", "handles"):
            slope = float(np.polyfit(x, [s[key] for s in warm], 1)[0])
            growth = (growth or {}) | {f"{key}_per_100": round(slope * 100, 3)}
    return {"stats": stats | {"crashed": crashed, "exit_code": app.proc.returncode, "wall_s": round(time.time() - t0)},
            "memory_samples": samples, "growth_after_warmup": growth, "failures": failures[:50]}  # fmt: skip


# ---------- report ----------


def render(r: dict[str, Any]) -> str:
    out = [
        f"# Vaani end-to-end: {r['label']}" + (" (DIRTY TREE)" if r["provenance"]["git"]["dirty"] else ""),
        "",
    ]
    s = r["settings"]
    out += [
        f"- Created {r['created_utc']} · commit `{(r['provenance']['git']['commit'] or '?')[:12]}` · "
        f"code `{(r['provenance']['code']['vaani_diff_sha256'] or '?')[:12]}` · ASR on {r['app_device']}",
        f"- Mic `{s['mic']}` ← playback `{s['play']}` · hotkey `{s['hotkey']}` · cleanup {s['cleanup']} · "
        f"streaming {s['streaming']} · lead {s['lead_ms']} ms",
        "",
    ]
    for tname, t in (r.get("accuracy") or {}).items():
        m = t["summary"]
        lat = m["release_to_paste_ms"]
        out += [
            f"## {tname}",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Clips | {m['n']} |",
            f"| Release-to-paste ms p50 / p95 / max (speech clips) | {lat['p50'] or 0:.0f} / {lat['p95'] or 0:.0f} / {lat['max'] or 0:.0f} |",
            f"| WER of pasted text % [95% CI] | {_pct(m['wer'])} |",
            f"| Content WER vs intended text % | {_pct(m['clean_wer'])} |",
            f"| Strict WER vs intended text % | {_pct(m['strict_wer'])} |",
            f"| Speech clips with nothing pasted | {m['speech_with_nothing_pasted']} |",
            f"| Silence/noise clips that pasted text | {m['silence_pasted']['k']}/{m['silence_pasted']['n']} |",
            f"| Dropped hotkeys | {m['dropped_hotkeys']} |",
            f"| Clips redone because another window took focus / still unfocused after redo | "
            f"{m['focus_lost_retried']} / {m['focus_lost_final']} |",
            f"| UI Automation read time p50 (latency resolution) | {m['ui_read_ms_p50']:.0f} ms |",
            "",
            "| Clip length | n | p50 ms | p95 ms |",
            "|---|---:|---:|---:|",
            *[
                f"| {k} | {v['n']} | {v['p50']:.0f} | {v['p95']:.0f} |"
                for k, v in m["latency_by_length"].items()
            ],
            "",
        ]
        bad = [
            c for c in t["clips"] if c["wer_final"][0] or (not c["ref"] and c["pasted"]) or not c["handled"]
        ][:15]
        if bad:
            out += ["| Clip | Reference | Pasted | ms |", "|---|---|---|---:|"]
            out += [
                f"| `{c['id']}` | {_cell(c['ref'])} | {_cell(c['pasted'])} | {c['release_to_paste_ms'] or 0:.0f} |"
                for c in bad
            ]
            out.append("")
    soak = r.get("soak")
    if soak:
        st = soak["stats"]
        out += [
            "## Soak",
            "",
            f"- Dictations: {st['done']} in {st['wall_s']} s · crashed: **{st['crashed']}** · dropped hotkeys: "
            f"**{st['dropped']}** · speech with no text: {st['speech_no_text']} · silence that pasted: {st['silence_pasted']}",
            f"- Growth after warm-up (linear fit, per 100 dictations): {soak['growth_after_warmup']}",
            "",
            "| Dictation | RSS MB | Private MB | Handles | Threads |",
            "|---:|---:|---:|---:|---:|",
            *[
                f"| {x['i']} | {x['rss_mb']:.0f} | {x['private_mb']:.0f} | {x['handles']} | {x['threads']} |"
                for x in soak["memory_samples"]
            ],
            "",
        ]
    return "\n".join(out)


def _pct(m: dict) -> str:
    if not m or m.get("value") is None:
        return "–"
    lo, hi = m["ci95"]
    return f"{m['value'] * 100:.2f} [{lo * 100:.2f}, {hi * 100:.2f}]"


def _cell(s: str, n: int = 90) -> str:
    s = s.replace("|", "\\|").replace("\n", " ⏎ ")
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m evals.e2e.run_e2e", description=__doc__.splitlines()[0])
    p.add_argument("--datasets", default="dictation,safety,negatives")
    p.add_argument("--manifest", action="append", default=[])
    p.add_argument("--limit", type=int, help="first N clips per manifest")
    p.add_argument("--target", choices=["notepad", "browser", "both"], default="notepad")
    p.add_argument("--mic", default="CABLE Output", help="Vaani's input device (partial name)")
    p.add_argument("--play", default="CABLE Input", help="playback device the clips are played into")
    p.add_argument(
        "--hotkey", default="f9", help="hold-mode key used for the test (default f9: no side effects)"
    )
    p.add_argument("--config", default=str(ROOT / "config.toml"))
    p.add_argument("--cleanup", choices=["config", "on", "off"], default="config")
    p.add_argument("--streaming", choices=["config", "on", "off"], default="config")
    p.add_argument("--lead-ms", type=int, default=300, help="press -> start of playback")
    p.add_argument("--hold-tail-ms", type=int, default=100, help="end of playback -> release")
    p.add_argument("--settle-ms", type=int, default=400, help="text unchanged this long = paste finished")
    p.add_argument("--timeout-s", type=float, default=15.0)
    p.add_argument("--soak", type=int, default=0, help="run N dictations for stability instead of accuracy")
    p.add_argument("--mem-every", type=int, default=25)
    p.add_argument("--label", default="e2e")
    p.add_argument("--out", default=str(ROOT / "evals" / "results"))
    a = p.parse_args(argv)
    if sys.platform != "win32":
        raise SystemExit("the e2e test drives Windows apps; run it on Windows")

    play_dev = resolve_output_device(a.play)
    clips, manifests = _clips(a)
    if a.soak:
        clips = [c for c in clips if c.duration_s <= 12] or clips  # short synthetic clips cycle faster
    work = Path(tempfile.mkdtemp(prefix="vaani-e2e-app-"))
    overrides = {
        "hotkey": {"key": a.hotkey, "mode": "hold"},
        "audio": {"input_device": a.mic, "sounds": False},  # beeps would be recorded by an acoustic mic
        "output": {"save_history": False},
        "cleanup": {"llm_enabled": None if a.cleanup == "config" else a.cleanup == "on"},
        "streaming": {"streaming": None if a.streaming == "config" else a.streaming == "on"},
    }
    data = write_config(Path(a.config), work / "config.toml", overrides)
    print(f"starting Vaani (log: {work / 'Vaani' / 'vaani.log'})", file=sys.stderr)
    code_at_start = sysinfo.code_state()
    app = VaaniProcess(work / "config.toml", work)
    try:
        targets = ["notepad", "browser"] if a.target == "both" else [a.target]
        result: dict[str, Any] = {
            "schema": 1,
            "label": a.label,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "app_device": app.device,
            "settings": {
                "mic": a.mic,
                "play": a.play,
                "hotkey": a.hotkey,
                "lead_ms": a.lead_ms,
                "cleanup": data.get("cleanup", {}).get("llm_enabled"),
                "streaming": data.get("streaming", {}).get("streaming"),
                "release_tail_ms": data.get("audio", {}).get("release_tail_ms"),
                "settle_ms": a.settle_ms,
                "targets": targets,
            },  # fmt: skip
            "args": vars(a) | {"out": sysinfo.rel_path(a.out), "config": sysinfo.rel_path(a.config)},
            "manifests": manifests,
            "provenance": {
                "git": sysinfo.git_info(),
                "code": code_at_start,
                "gpu": sysinfo.gpu_info(),
                "os": platform.platform(),
            },  # fmt: skip
        }
        if a.soak:
            result["soak"] = run_soak(app, clips, play_dev, a)
        else:
            result["accuracy"] = run_accuracy(app, targets, clips, play_dev, a)
        result["app_log_tail"] = [ln for ln in app._text().splitlines() if "WARNING" in ln or "ERROR" in ln][
            -20:
        ]
    finally:
        app.stop()

    sha = (result["provenance"]["git"]["commit"] or "nogit")[:7]
    stem = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{sha}_{a.label}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = render(result)
    (out / f"{stem}.md").write_text(md, encoding="utf-8", newline="\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(md)
    print(f"wrote {out / stem}.json / .md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
