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

import json
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


def _live_peer(watch_dir: Path) -> int | None:
    """PID of another live watchdog, else None. Prevents duplicate daemons
    (seen 3 concurrent instances on 2026-09-14 from mystery relaunches)."""
    try:
        pid = int((watch_dir / "watchdog.pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if pid == os.getpid():
        return None
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return pid if str(pid) in out.stdout else None


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


def send_message(
    session_id: str,
    message: str,
    timeout_sec: float = 300.0,
    err_log: Path | None = None,
) -> int:
    """Deliver one message into the session via headless grok. Returns exit code.

    stderr tail is appended to err_log: a silent rc is a mystery, a logged rc
    is a diagnosis (lesson of the 11:14 rc=1 pair).
    """
    try:
        proc = subprocess.run(
            ["grok", "-p", message, "-r", session_id],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
            cwd="E:\\",
        )
        if err_log is not None and proc.stderr:
            try:
                with err_log.open("a", encoding="utf-8") as fh:
                    tail = proc.stderr.strip().splitlines()[-5:]
                    fh.write(
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} rc={proc.returncode}\n"
                    )
                    for line in tail:
                        fh.write(f"  stderr: {line[:300]}\n")
            except OSError:
                pass
        return proc.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        if err_log is not None:
            try:
                with err_log.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"{time.strftime('%Y-%m-%dT%H:%M:%S')} EXC {type(exc).__name__}\n"
                    )
            except OSError:
                pass
        return 99


POKE_PS1 = Path(__file__).with_name("tui_poke.ps1")


def _proxy_state(timeout_sec: float = 5.0) -> tuple[float, int] | None:
    """(idle_sec, in_flight) from muse-proxy, else None. Silent fallback to files."""
    url = os.environ.get("VNR_PROXY_WATCHDOG_URL", "http://127.0.0.1:8120/__watchdog")
    if not url:
        return None
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout_sec) as r:
            j = json.loads(r.read().decode("utf-8"))
        return (float(j.get("idle_sec", 0.0)), int(j.get("in_flight", 0)))
    except (OSError, ValueError, TypeError, KeyError):
        # URLError/OSError (network), JSONDecodeError/ValueError (bad body),
        # TypeError/KeyError (bad shape): any of these means "no signal",
        # never a crash. Narrow by construction; BLE001 stays clean.
        return None


def _poke_enabled() -> bool:
    return os.environ.get("VNR_WATCH_POKE", "1") not in ("0", "", "false", "no")


def _poke_tui(message: str, timeout_sec: float = 60.0) -> int | None:
    """Paste message+Enter into the worker PowerShell window.

    Returns ps1 rc (0 = poked, 2 = no window), or None when disabled.
    """
    if not _poke_enabled():
        return None
    last = 3
    for _ in range(3):
        try:
            proc = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(POKE_PS1),
                    "-TitlePattern",
                    os.environ.get("VNR_WATCH_TITLE", "grok"),
                    "-Message",
                    message,
                ],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return 3
        if proc.returncode == 0:
            return 0
        last = proc.returncode
        time.sleep(2)
    return last


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
    # Proxy is authoritative: prefer real traffic age, and never fire while a
    # response stream is open (it's transmitting — leave it alone).
    proxy = _proxy_state()
    if proxy is not None:
        p_idle, p_flying = proxy
        if p_flying > 0:
            return None
        if p_idle < age:
            age = p_idle
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
        return line
    # Primary wake path: paste into the live worker TUI (no competing session).
    # Falls through to the headless resume below only when poking is disabled
    # or the worker window isn't found (rc 2).
    poke_rc = _poke_tui(message)
    if poke_rc == 0:
        send_history.append(now)
        try:
            with (watch_dir / "sends.log").open("a", encoding="utf-8") as fh:
                fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {reason} POKE rc=0\n")
        except OSError:
            pass
        _log(watch_dir, f"SENT {reason} POKE rc=0: {line}")
        return line
    if poke_rc not in (None, 0, 2):
        _log(
            watch_dir,
            f"POKE-DEFER rc={poke_rc}: worker window exists, retry next episode (no headless run)",
        )
        return line
    if poke_rc == 2:
        _log(watch_dir, "POKE-SKIP rc=2 (no worker window): falling back to headless")
    code = send_message(session_id, message, err_log=watch_dir / "sends.log")
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
    older = _live_peer(watch_dir)
    if older is not None:
        _log(watch_dir, f"another watchdog (pid={older}) is alive: this one exits")
        print(f"watchdog already running as pid={older}; exiting", file=sys.stderr)
        return 0
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
    pause_logged = False
    iteration = 0
    while True:
        try:
            if (watch_dir / "watchdog.pause").exists():
                if not pause_logged:
                    _log(watch_dir, "paused via watchdog.pause flag: holding fire")
                    pause_logged = True
                alarmed = False
                alarm = None
            else:
                if pause_logged:
                    _log(watch_dir, "pause flag removed: resuming")
                    pause_logged = False
                alarm = check_once(
                    watch_dir,
                    pause_after,
                    stalled_after,
                    history,
                    max_per_hour,
                    dry_run=alarmed,  # one send per episode: already fired, just watch
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
