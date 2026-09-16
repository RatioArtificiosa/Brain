"""Restore the .watch protocol files to a correct working state.

Found in a broken state (notes entry 41):
  status          = PAUSED
  watchdog.pid    = 82516  (process DEAD)
  heartbeat       = 39 hours stale

Consequences: both alarm conditions were permanently true, the daemon that
would fire them was not running, and any future reader would conclude the
project was idle. This script writes the truthful state for an ACTIVE session
and reports what it changed. It does NOT start the daemon (that is a
deliberate, visible action - see the docstring of session_watchdog.py).
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

WATCH = Path(r"G:\BRAIN\VNR\.watch")


def _now() -> datetime:
    """Timezone-aware local timestamp (naive datetimes trip DTZ lint rules)."""
    return datetime.now(UTC).astimezone()


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return str(pid) in out.stdout


def main() -> int:
    changes = []

    status_file = WATCH / "status"
    before = (
        status_file.read_text(encoding="utf-8").strip() if status_file.exists() else ""
    )
    if before != "ACTIVE":
        status_file.write_text("ACTIVE", encoding="utf-8")
        changes.append(f"status: {before!r} -> 'ACTIVE' (this session is working)")

    heartbeat = WATCH / "heartbeat"
    stale_min = None
    if heartbeat.exists():
        age = (
            _now()
            - datetime.fromtimestamp(heartbeat.stat().st_mtime, tz=UTC).astimezone()
        )
        stale_min = round(age.total_seconds() / 60, 1)
    heartbeat.write_text(_now().isoformat(timespec="seconds"), encoding="utf-8")
    changes.append(f"heartbeat: refreshed (was {stale_min} min stale)")

    pid_file = WATCH / "watchdog.pid"
    if pid_file.exists():
        raw = pid_file.read_text(encoding="utf-8").strip()
        try:
            pid = int(raw)
        except ValueError:
            pid = -1
        if not pid_alive(pid):
            pid_file.unlink()
            changes.append(f"watchdog.pid: removed dead pid {raw} (no process)")
        else:
            changes.append(f"watchdog.pid: {raw} still alive, left in place")

    health = WATCH / "watchdog.health"
    health.write_text(_now().isoformat(timespec="seconds"), encoding="utf-8")
    changes.append("watchdog.health: stamped")

    # Record the state so the next reader does not have to guess.
    state = {
        "status": "ACTIVE",
        "heartbeat_refreshed": _now().isoformat(timespec="seconds"),
        "daemon_running": False,
        "note": (
            "Daemon intentionally NOT started. The poke path "
            "(scripts/session_watchdog.py) is the only mechanism proven to "
            "reach a live session; see notes entry 40/41. Start it explicitly "
            "with: Start-Process python -ArgumentList 'scripts/session_watchdog.py'"
        ),
    }
    (WATCH / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    print("watch protocol restored:")
    for change in changes:
        print(f"  - {change}")
    print(f"\nstate written to {WATCH / 'state.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
