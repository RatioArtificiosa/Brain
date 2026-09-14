"""Outside watchdog: sends a message INTO the live grok session via headless mode.

Why this exists: monitors, schedulers and subagents all die with the session or
the model gateway. An outside OS process does not — it watches the heartbeat
file and, on silence, runs ``grok -p <msg> -r <session>``, which lands in the
session as a new turn. That is the wake path. Everything else is decoration.

Protocol files in WATCH_DIR (default G:\\BRAIN\\VNR\\.watch):
  heartbeat    mtime = last agent work (touched by the agent).
  status       ACTIVE or PAUSED (missing/garbage = PAUSED).
  session.id   grok session UUID to message (written at deploy; or env
               VNR_GROK_SESSION_ID wins).
  sends.log    one line per attempted send (timestamp, reason, returncode).
  watchdog.log diagnostics. watchdog.health  liveness timestamp.

Firing rules (checked every INTERVAL_SEC, default 15):
  PAUSED + heartbeat older than PAUSE_AFTER_SEC (default 30)  -> send WAKEUP.
  ACTIVE + heartbeat older than STALLED_AFTER_SEC (default 900) -> send STALLED.
  Otherwise silent. One send per episode: after firing, no resend until the
  heartbeat goes fresh again. Hard cap MAX_SENDS_PER_HOUR (default 10): past
  the cap it logs and backs off instead of burning money on a broken setup.

Surviving reboots/sessions: run detached (Start-Process, no console), PID in
watchdog.pid. Stop by killing that PID. The loop itself never exits on errors:
per-iteration faults are logged and skipped.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from collections import deque
from pathlib import Path

DEFAULT_WATCH_DIR = r"G:\BRAIN\VNR\.watch"
DEFAULT_INTERVAL_SEC = 15.0
DEFAULT_PAUSE_AFTER_SEC = 30.0
DEFAULT_STALLED_AFTER_SEC = 900.0
DEFAULT_MAX_SENDS_PER_HOUR = 10


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _log(watch_dir: Path, message: str) -> None:
    try:
        with (watch_dir / "watchdog.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
    except OSError:
        pass


def _read_status(watch_dir: Path) -> str:
    try:
        lines = (watch_dir / "status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return "PAUSED"
    return "ACTIVE" if lines and lines[0].strip().upper() == "ACTIVE" else "PAUSED"


def _heartbeat_age_sec(watch_dir: Path) -> float:
    try:
        return time.time() - (watch_dir / "heartbeat").stat().st_mtime
    except OSError:
        return float("inf")


def _session_id(watch_dir: Path) -> str:
    env = os.environ.get("VNR_GROK_SESSION_ID", "").strip()
    if env:
        return env
    try:
        return (watch_dir / "session.id").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def needs_fire(
    status: str, age_sec: float, pause_after: float, stalled_after: float
) -> str | None:
    """Pure decision: 'WAKEUP' / 'STALLED' / None. Unit-tested."""
    if status == "ACTIVE":
        return "STALLED" if age_sec > stalled_after else None
    return "WAKEUP" if age_sec > pause_after else None


def build_message(reason: str, age_sec: float, status: str) -> str:
    age_txt = "missing" if age_sec == float("inf") else f"{age_sec:.0f}s old"
    return (
        f"VNR watchdog {reason}: no agent heartbeat for {age_txt} "
        f"(status={status}). Touch G:\\BRAIN\\VNR\\.watch\\heartbeat, then read "
        f"status and the first unchecked item in G:\\BRAIN\\VNR\\02-checklist.md. "
        f"If the user is talking to you, answer them first. Otherwise, if genuinely "
        f"blocked on the user, say so briefly and stop; else resume the next item "
        f"quietly without asking anything."
    )


def send_message(session_id: str, message: str, timeout_sec: float = 300.0) -> int:
    """Deliver one message into the session via headless grok. Returns exit code."""
    try:
        proc = subprocess.run(
            ["grok", "-p", message, "-r", session_id],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return proc.returncode
    except (OSError, subprocess.TimeoutExpired):
        return 99


def check_once(
    watch_dir: Path,
    pause_after: float,
    stalled_after: float,
    send_history: deque,
    max_per_hour: int,
    dry_run: bool = False,
) -> str | None:
    """Evaluate once. Sends at most one message. Returns the alarm line or None."""
    status = _read_status(watch_dir)
    age = _heartbeat_age_sec(watch_dir)
    reason = needs_fire(status, age, pause_after, stalled_after)
    if reason is None:
        return None
    now = time.time()
    while send_history and now - send_history[0] > 3600:
        send_history.popleft()
    if len(send_history) >= max_per_hour:
        _log(
            watch_dir, f"send cap hit ({max_per_hour}/h): backing off, reason={reason}"
        )
        return None
    session_id = _session_id(watch_dir)
    if not session_id:
        _log(
            watch_dir, "cannot fire: no session id (VNR_GROK_SESSION_ID or session.id)"
        )
        return None
    message = build_message(reason, age, status)
    line = f"{reason} heartbeat {age:.0f}s stale (status={status})"
    if dry_run:
        _log(watch_dir, f"DRY-RUN would send: {line}")
        return line
    code = send_message(session_id, message)
    send_history.append(now)
    try:
        with (watch_dir / "sends.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {reason} rc={code}\n")
    except OSError:
        pass
    _log(watch_dir, f"SENT {reason} rc={code}: {line}")
    return line


def run_forever(
    watch_dir: Path,
    interval_sec: float,
    pause_after: float,
    stalled_after: float,
    max_per_hour: int,
    max_iterations: int = 0,
) -> int:
    watch_dir.mkdir(parents=True, exist_ok=True)
    try:
        (watch_dir / "watchdog.pid").write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass
    _log(
        watch_dir,
        f"session-watchdog start pid={os.getpid()} interval={interval_sec}s "
        f"pause_after={pause_after}s stalled_after={stalled_after}s",
    )
    history: deque = deque()
    alarmed = False
    iteration = 0
    while True:
        try:
            alarm = check_once(
                watch_dir, pause_after, stalled_after, history, max_per_hour
            )
            if alarm is not None and not alarmed:
                print(alarm, flush=True)
                alarmed = True
            elif alarm is None and alarmed:
                _log(watch_dir, "activity resumed: alarm re-armed")
                alarmed = False
            try:
                (watch_dir / "watchdog.health").write_text(
                    f"{time.time():.0f}\n", encoding="utf-8"
                )
            except OSError:
                pass
        except Exception:  # noqa: BLE001 — the loop must survive anything unexpected
            _log(watch_dir, "iteration fault (survived):\n" + traceback.format_exc())
        iteration += 1
        if max_iterations and iteration >= max_iterations:
            return 0
        time.sleep(interval_sec)
    return 0


def main(argv: list[str]) -> int:
    watch_dir = Path(os.environ.get("VNR_WATCH_DIR", DEFAULT_WATCH_DIR))
    interval = _env_float("VNR_WATCH_INTERVAL_SEC", DEFAULT_INTERVAL_SEC)
    pause_after = _env_float("VNR_WATCH_PAUSE_AFTER_SEC", DEFAULT_PAUSE_AFTER_SEC)
    stalled_after = _env_float("VNR_WATCH_STALLED_AFTER_SEC", DEFAULT_STALLED_AFTER_SEC)
    max_per_hour = _env_int("VNR_WATCH_MAX_SENDS_PER_HOUR", DEFAULT_MAX_SENDS_PER_HOUR)
    if min(interval, pause_after, stalled_after) <= 0 or max_per_hour <= 0:
        print("watchdog tuning env vars must be positive", file=sys.stderr)
        return 2
    max_iter = 0
    if "--max-iterations" in argv:
        try:
            max_iter = int(argv[argv.index("--max-iterations") + 1])
        except (IndexError, ValueError):
            print("--max-iterations needs an integer", file=sys.stderr)
            return 2
    if "--once" in argv:
        watch_dir.mkdir(parents=True, exist_ok=True)
        alarm = check_once(watch_dir, pause_after, stalled_after, deque(), max_per_hour)
        if alarm is not None:
            print(alarm, flush=True)
        return 0
    return run_forever(
        watch_dir, interval, pause_after, stalled_after, max_per_hour, max_iter
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
