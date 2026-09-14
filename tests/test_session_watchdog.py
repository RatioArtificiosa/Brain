"""Tests for scripts/session_watchdog.py. Never sends a real message."""

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from session_watchdog import build_message, check_once, needs_fire


def _watch(tmp_path: Path, status: str | None, beat_age: float | None) -> Path:
    watch = tmp_path / "watch"
    watch.mkdir(parents=True, exist_ok=True)
    if status is not None:
        (watch / "status").write_text(status, encoding="utf-8")
    if beat_age is not None:
        beat = watch / "heartbeat"
        beat.touch()
        import os

        stale = time.time() - beat_age
        os.utime(beat, (stale, stale))
    (watch / "session.id").write_text("test-session-id", encoding="utf-8")
    return watch


def test_needs_fire_matrix():
    assert needs_fire("ACTIVE", 10, 30.0, 900.0) is None
    assert needs_fire("PAUSED", 10, 30.0, 900.0) is None
    assert needs_fire("PAUSED", 31, 30.0, 900.0) == "WAKEUP"
    assert needs_fire("ACTIVE", 901, 30.0, 900.0) == "STALLED"
    assert needs_fire("ACTIVE", 899, 30.0, 900.0) is None


def test_message_points_at_checklist():
    msg = build_message("WAKEUP", 45.0, "PAUSED")
    assert "02-checklist.md" in msg and "heartbeat" in msg


def test_dry_run_fires_without_sending(tmp_path):
    watch = _watch(tmp_path, "PAUSED", 120)
    line = check_once(watch, 30.0, 900.0, deque(), 10, dry_run=True)
    assert line is not None and line.startswith("WAKEUP")
    assert not (watch / "sends.log").exists(), "dry run must not record a send"


def test_missing_session_id_never_sends(tmp_path):
    watch = _watch(tmp_path, "PAUSED", 120)
    (watch / "session.id").unlink()
    assert check_once(watch, 30.0, 900.0, deque(), 10, dry_run=True) is None


def test_hourly_cap_backs_off(tmp_path):
    watch = _watch(tmp_path, "PAUSED", 120)
    history = deque([time.time()] * 10)  # cap already exhausted
    assert check_once(watch, 30.0, 900.0, history, 10, dry_run=True) is None
