"""Clean-room reachability test: clone the PUBLIC repo and follow the README.

This is the check that found entry-40's defects. It answers one question
honestly: can a stranger who has never seen this machine get a working
product? Every step below is a command the README tells them to run.

Run:  python scripts/reachability_check.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "https://github.com/RatioArtificiosa/Brain.git"


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,  # returncode is inspected by the caller
    )
    return proc.returncode, (proc.stdout + proc.stderr)


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="vnr-reach-"))
    clone = root / "Brain"
    venv = root / "venv"
    print(f"clean room: {root}")

    try:
        # 1. Clone the public repo, exactly as the README implies.
        code, out = run(["git", "clone", "--quiet", "--depth", "1", REPO, str(clone)])
        if code != 0:
            print("FAIL clone:", out[-400:])
            return 1
        print("  [ok] cloned public repo")

        # 2. Create an isolated environment.
        code, out = run([sys.executable, "-m", "venv", str(venv)])
        if code != 0:
            print("FAIL venv:", out[-400:])
            return 1
        py = venv / "Scripts" / "python.exe"
        vnr = venv / "Scripts" / "vnr.exe"
        if not py.exists():
            py = venv / "bin" / "python"
            vnr = venv / "bin" / "vnr"
        print("  [ok] isolated venv")

        # 3. README step: pip install -e .
        code, out = run([str(py), "-m", "pip", "install", "--quiet", "-e", "."], clone)
        if code != 0:
            print("FAIL install:", out[-600:])
            return 1
        print("  [ok] pip install -e .")

        # 4. numpy must arrive automatically (entry-40 regression).
        code, out = run([str(py), "-c", "import numpy; print(numpy.__version__)"])
        status = "ok" if code == 0 else "FAIL"
        print(f"  [{status}] numpy present: {out.strip()[:40]}")
        if code != 0:
            return 1

        # 5. The quickstart commands must actually run.
        checks = [
            ("vnr doctor", [str(vnr), "doctor"], 0),
            (
                "vnr simulate",
                [str(vnr), "simulate", "--neurons", "96", "--ticks", "250"],
                0,
            ),
            (
                "vnr benchmark",
                [
                    str(vnr),
                    "benchmark",
                    "--neurons",
                    "64",
                    "--ticks",
                    "150",
                    "--repeat",
                    "2",
                ],
                0,
            ),
            ("vnr generate", [str(vnr), "generate", "--list"], 0),
            (
                "vnr experiment (no-write)",
                [
                    str(vnr),
                    "experiment",
                    "four-way",
                    "--neurons",
                    "64",
                    "--ticks",
                    "150",
                    "--no-write",
                ],
                0,
            ),
            # Without the data extra this must be GUIDANCE, not a traceback.
            ("vnr connectome stats (no extra)", [str(vnr), "connectome", "stats"], 1),
        ]
        failed = 0
        for label, cmd, want in checks:
            code, out = run(cmd)
            ok = code == want and "Traceback" not in out
            if label.startswith("vnr benchmark"):
                ok = ok and "numpy (vectorized)" in out
            if label.startswith("vnr connectome"):
                ok = ok and ".[data]" in out
            print(f"  [{'ok' if ok else 'FAIL'}] {label}")
            if not ok:
                failed += 1
                print(
                    "       ", out.strip().splitlines()[-1][:120] if out.strip() else ""
                )

        print()
        print("REACHABLE" if failed == 0 else f"NOT REACHABLE ({failed} failed)")
        return 0 if failed == 0 else 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
