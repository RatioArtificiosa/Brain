"""Tests for scripts/watchdog.py: skip-when-active, alarm-once-on-pause, fault paths."""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from watchdog import check_once


def _setup_watch(tmp_path: Path, status: str | None, beat_age_sec: float | None) -> Path:
    watch = tmp_path / "watch"
    watch.mkdir(parents=True, exist_ok=True)
    if status is not None:
        (watch / "status").write_text(status, encoding="utf-8")
    if beat_age_sec is not None:
        beat = watch / "heartbeat"
        beat.touch()
        stale = time.time() - beat_age_sec
        os.utime(beat, (stale, stale))
    return watch


def _log(watch: Path) -> Path:
    return watch / "watchdog.log"


def test_active_with_fresh_heartbeat_stays_silent(tmp_path):
    watch = _setup_watch(tmp_path, "ACTIVE", 60)
    assert check_once(watch, 180.0, _log(watch)) is None


def test_active_but_stalled_alarms_once_per_episode(tmp_path):
    watch = _setup_watch(tmp_path, "ACTIVE", 10_000)
    alarm = check_once(watch, 180.0, _log(watch), stalled_after_sec=900.0)
    assert alarm is not None and alarm.startswith("STALLED")
    # still stale but inside the stalled threshold: silent
    assert check_once(watch, 180.0, _log(watch), stalled_after_sec=20_000.0) is None


def test_paused_with_fresh_heartbeat_stays_silent(tmp_path):
    watch = _setup_watch(tmp_path, "PAUSED", 5)
    assert check_once(watch, 180.0, _log(watch)) is None


def test_paused_with_stale_heartbeat_alarms(tmp_path):
    watch = _setup_watch(tmp_path, "PAUSED", 500)
    alarm = check_once(watch, 180.0, _log(watch))
    assert alarm is not None and alarm.startswith("WAKEUP")


def test_missing_files_alarm_fault_tolerant(tmp_path):
    watch = tmp_path / "watch"
    watch.mkdir(parents=True, exist_ok=True)
    alarm = check_once(watch, 180.0, _log(watch))
    assert alarm is not None and alarm.startswith("WAKEUP")


def test_garbage_status_means_paused(tmp_path):
    watch = _setup_watch(tmp_path, "???", 500)
    assert check_once(watch, 180.0, _log(watch)) is not None
    watch2 = _setup_watch(tmp_path / "w2p", None, None)
    assert check_once(watch2, 180.0, _log(watch2)) is not None


def _launch(watch: Path, out_path: Path) -> subprocess.Popen:
    env = dict(os.environ, VNR_WATCH_DIR=str(watch),
               VNR_WATCH_INTERVAL_SEC="0.5", VNR_WATCH_PAUSE_AFTER_SEC="2")
    out = out_path.open("w", encoding="utf-8")
    return subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve().parent.parent
                             / "scripts" / "watchdog.py")],
                            env=env, stdout=out, stderr=subprocess.DEVNULL)


def _wakeups(out_path: Path) -> list[str]:
    try:
        lines = out_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.startswith("WAKEUP")]


def test_full_episode_lifecycle_integration(tmp_path):
    watch = _setup_watch(tmp_path, "ACTIVE", 0)
    out_path = tmp_path / "stdout.txt"
    proc = _launch(watch, out_path)
    try:
        time.sleep(3)
        assert _wakeups(out_path) == [], "ACTIVE must stay silent"
        (watch / "status").write_text("PAUSED", encoding="utf-8")
        for _ in range(3):  # ongoing work: keep the beat fresh inside the window
            (watch / "heartbeat").touch()
            time.sleep(1)
        assert _wakeups(out_path) == [], "fresh beat must stay silent"
        stale = time.time() - 60  # backdate: simulate a 60 s pause
        os.utime(watch / "heartbeat", (stale, stale))
        time.sleep(3)
        assert len(_wakeups(out_path)) == 1, "exactly one alarm per pause episode"
        time.sleep(2.5)
        assert len(_wakeups(out_path)) == 1, "no alarm spam while paused"
        (watch / "heartbeat").touch()  # work resumes
        time.sleep(1)
        stale = time.time() - 60  # pause again: must re-arm
        os.utime(watch / "heartbeat", (stale, stale))
        time.sleep(3)
        assert len(_wakeups(out_path)) == 2, "alarm must re-arm after activity"
        (watch / "heartbeat").touch()  # activity resumes: clears the episode
        time.sleep(1)
        (watch / "heartbeat").unlink()  # fault path: files vanish mid-new-episode
        (watch / "status").unlink()
        time.sleep(3)
        assert len(_wakeups(out_path)) == 3, "missing files must still alarm"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    assert (watch / "watchdog.log").exists()
