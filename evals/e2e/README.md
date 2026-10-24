# End-to-end Windows test

Drives the **real app**: it starts Vaani, plays eval clips into a virtual microphone, holds and
releases the hotkey, and reads what was pasted into Notepad (and optionally a browser `<textarea>`).
It measures what the unit evals can't: true **release-to-paste latency** (VAD + ASR + cleanup +
clipboard paste + the target app showing the text), dropped hotkeys, and long-run stability.

## One-time setup

1. Install **VB-Audio Virtual Cable** (free, https://vb-audio.com/Cable/): run
   `VBCABLE_Setup_x64.exe` as administrator, then reboot. It adds a playback device **"CABLE Input"**
   and a recording device **"CABLE Output"**; whatever is played into the first comes out of the second.
2. In Windows Sound settings, leave your normal speakers and mic as the defaults. The test points only
   its own Vaani instance at the cable, so your everyday setup is unchanged.
3. Install the eval dependencies (includes `pywinauto`):
   `.\.venv\Scripts\python -m pip install -r requirements-eval.txt`
4. **Quit Vaani** from the tray (only one instance can hold the hotkey) and close any Notepad
   window with unsaved work you care about. The test opens its own Notepad file and never closes yours.

## Run

```powershell
.\.venv\Scripts\python -m evals.datasets.prepare negatives safety     # synthetic clips, once
.\.venv\Scripts\python -m evals.e2e.run_e2e --label e2e-notepad        # dictation + safety + negatives
.\.venv\Scripts\python -m evals.e2e.run_e2e --target both --limit 10   # Notepad and an Edge textarea
.\.venv\Scripts\python -m evals.e2e.run_e2e --cleanup off --streaming off --label e2e-raw
.\.venv\Scripts\python -m evals.e2e.run_e2e --soak 1000 --datasets safety,negatives --label soak
```

Don't touch the keyboard or mouse while it runs: the test owns focus and the hotkey. A 1000-dictation
soak takes about an hour.

- The test hotkey is **F9** by default (`--hotkey`), not your configured key, so caps lock or a
  modifier never gets stuck. The hook code path is the same for any key.
- Vaani runs with your `config.toml` plus: hotkey = F9 (hold), mic = `CABLE Output`, beeps off,
  history off. `--cleanup` and `--streaming` override those settings. Its log goes to a temp folder
  (printed at start), not `%LOCALAPPDATA%\Vaani`.
- Without VB-Cable you can smoke-test the harness acoustically (laptop speakers into the laptop mic):
  `--play "Speakers" --mic "Microphone Array" --limit 3`. Accuracy is then meaningless; it only proves
  the harness works.

## What the report shows

`evals/results/<time>_<commit>_<label>.json` and `.md`:

- **Release-to-paste** p50 / p95 / max for speech clips, also by clip length. Clock starts at key
  release and stops at the last change of the target's text. It includes Vaani's `release_tail_ms`.
  With streaming on, phrases can land while the key is held; if nothing changes after release the
  latency is 0. The UI Automation read time is reported as the measurement resolution.
- **Accuracy of the pasted text**: normalized WER, and strict / content WER against the intended
  text for dictation and safety clips.
- **Silence/noise clips that pasted anything** (target 0) and **speech clips with nothing pasted**.
- **Dropped hotkeys**: press/release cycles with no sign in Vaani's log that the dictation was handled.
- **Soak**: crashes, dropped hotkeys, and Vaani's RSS / private memory / handle count every 25
  dictations, plus a linear growth rate per 100 dictations after the first 10% (warm-up).
