"""VNR agent watchdog: alarm after a pause, stay silent while work happens.

Protocol (all files live in WATCH_DIR, default G:\\BRAIN\\VNR\\.watch):
  heartbeat   empty file; its mtime = last observed work activity (touched by
              the agent/tooling after every work step).
  status      first line ACTIVE or PAUSED (missing/unreadable/garbage = PAUSED).
  watchdog.log        diagnostics (never stdout — stdout is the alarm channel).
  watchdog.health     watchdog's own last-check timestamp (self-health proof).

Loop (every INTERVAL_SEC, default 15):
  ACTIVE + fresh beat    -> silent (working: skip). Clears a prior episode.
  ACTIVE + stale beat    -> STALLED alarm once per episode. The agent is supposed
                           to be working but produced no heartbeat for longer than
                           STALLED_AFTER_SEC (default 900): it may be stuck, or a
                           turn ended without yielding to PAUSED.
  PAUSED + fresh beat   -> silent. Clears a prior episode (re-arms the alarm).
  PAUSED + stale beat   -> print exactly one WAKEUP line, then stay silent until
                           activity resumes (no alarm spam for one pause episode).
  heartbeat missing     -> treated as infinitely stale (fault-tolerant: still alarms).

Stdout discipline: ONLY alarm lines ("WAKEUP ..."/"STALLED ...") go to stdout,
because a monitor forwards every stdout line to the agent. Everything else goes
to the log file. The loop never exits on its own: per-iteration errors are logged
and skipped; only KeyboardInterrupt/SystemExit stop it.

Tuning via environment (used by the test harness):
  VNR_WATCH_DIR, VNR_WATCH_INTERVAL_SEC, VNR_WATCH_PAUSE_AFTER_SEC,
  VNR_WATCH_STALLED_AFTER_SEC
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

DEFAULT_WATCH_DIR = r"G:\BRAIN\VNR\.watch"
DEFAULT_INTERVAL_SEC = 15.0
DEFAULT_PAUSE_AFTER_SEC = 30.0
DEFAULT_STALLED_AFTER_SEC = 900.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _log(log_path: Path, message: str) -> None:
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
    except OSError:
        pass  # logging must never break the loop


def _read_status(watch_dir: Path) -> str:
    """Return ACTIVE or PAUSED. Anything unreadable or unexpected means PAUSED."""
    try:
        text = (watch_dir / "status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return "PAUSED"
    return "ACTIVE" if text and text[0].strip().upper() == "ACTIVE" else "PAUSED"


def _heartbeat_age_sec(watch_dir: Path) -> float:
    """Seconds since last heartbeat; +inf when the file is missing/unreadable."""
    try:
        return time.time() - (watch_dir / "heartbeat").stat().st_mtime
    except OSError:
        return float("inf")


def check_once(
    watch_dir: Path,
    pause_after_sec: float,
    log_path: Path,
    stalled_after_sec: float = DEFAULT_STALLED_AFTER_SEC,
) -> str | None:
    """One evaluation. Returns an alarm line when one fires, else None.

    Pure decision logic (no sleeping, no printing) so tests can drive it directly.
    Caller tracks episode state via the returned value: a non-None return means
    "episode already alarmed, stay silent until activity resumes".
    """
    status = _read_status(watch_dir)
    age = _heartbeat_age_sec(watch_dir)
    if status == "ACTIVE":
        if age <= stalled_after_sec:
            return None
        age_txt = "missing" if age == float("inf") else f"{age:.0f}s stale"
        return (
            f"STALLED status=ACTIVE but heartbeat {age_txt} "
            f"(threshold={stalled_after_sec:.0f}s): worker may be stuck or a turn "
            f"ended without yielding — resume next unchecked item in "
            f"G:\\BRAIN\\VNR\\02-checklist.md"
        )
    if age <= pause_after_sec:
        return None
    age_txt = "missing" if age == float("inf") else f"{age:.0f}s stale"
    return (
        f"WAKEUP paused {age_txt} with no heartbeat (status=PAUSED, "
        f"threshold={pause_after_sec:.0f}s): resume next unchecked item in "
        f"G:\\BRAIN\\VNR\\02-checklist.md"
    )


def run_forever(
    watch_dir: Path,
    interval_sec: float,
    pause_after_sec: float,
    max_iterations: int = 0,
    stalled_after_sec: float = DEFAULT_STALLED_AFTER_SEC,
) -> int:
    """Main loop. max_iterations>0 bounds it (used by tests; production = 0)."""
    watch_dir.mkdir(parents=True, exist_ok=True)
    log_path = watch_dir / "watchdog.log"
    health_path = watch_dir / "watchdog.health"
    _log(
        log_path,
        f"watchdog start dir={watch_dir} interval={interval_sec}s pause_after={pause_after_sec}s stalled_after={stalled_after_sec}s",
    )
    alarmed = False
    iteration = 0
    while True:
        try:
            alarm = check_once(watch_dir, pause_after_sec, log_path, stalled_after_sec)
            if alarm is not None and not alarmed:
                print(alarm, flush=True)  # THE alarm channel: monitor forwards this
                _log(log_path, f"ALARM: {alarm}")
                alarmed = True
            elif alarm is None and alarmed:
                _log(log_path, "activity resumed: alarm re-armed")
                alarmed = False
            try:
                health_path.write_text(f"{time.time():.0f}\n", encoding="utf-8")
            except OSError:
                pass
        except Exception:  # noqa: BLE001 — the loop must survive anything unexpected
            _log(log_path, "iteration fault (survived):\n" + traceback.format_exc())
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
    if interval <= 0 or pause_after <= 0 or stalled_after <= 0:
        print(
            "VNR_WATCH_INTERVAL_SEC, VNR_WATCH_PAUSE_AFTER_SEC and "
            "VNR_WATCH_STALLED_AFTER_SEC must be positive",
            file=sys.stderr,
        )
        return 2
    max_iter = 0
    if "--max-iterations" in argv:
        try:
            max_iter = int(argv[argv.index("--max-iterations") + 1])
        except (IndexError, ValueError):
            print("--max-iterations needs an integer", file=sys.stderr)
            return 2
    return run_forever(watch_dir, interval, pause_after, max_iter, stalled_after)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
