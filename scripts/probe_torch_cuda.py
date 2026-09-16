"""Determine empirically which torch CUDA builds can serve sm_52 (Maxwell).

Recorded verdict (notes entry 27, R1): the only CUDA line driver 537.99 can
load is cu121, and "cu121 is retired from publication". This script tests
that claim against the actual package indexes instead of assuming it, and
finds the newest torch that still ships Maxwell kernels.

Method: query each candidate (version, cuda) pair against the official
PyTorch wheel index and record whether a distribution exists. No guessing.

    python scripts/probe_torch_cuda.py
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

# torch versions plausibly spanning the Maxwell-support boundary, crossed with
# the CUDA lines this driver (537.99 -> CUDA 12.2 max) can load.
TORCH_VERSIONS = [
    "1.13.1",
    "2.0.1",
    "2.1.2",
    "2.2.2",
    "2.3.1",
    "2.4.1",
    "2.5.1",
    "2.6.0",
]
CUDA_LINES = ["cu118", "cu121", "cu124"]

INDEX = "https://download.pytorch.org/whl"


def wheel_exists(version: str, cuda: str) -> tuple[bool, str]:
    """Is there a wheel for this (torch, cuda) pair on the official index?"""
    url = f"{INDEX}/{cuda}/torch/"
    try:
        with urllib.request.urlopen(url, timeout=45) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - network is the variable under test
        return False, f"index unreachable: {type(exc).__name__}"
    # Wheel filenames look like torch-2.4.1+cu121-cp313-cp313-win_amd64.whl
    pattern = re.compile(
        rf"torch-{re.escape(version)}\+{cuda}-cp3\d+-cp3\d+-win_amd64\.whl"
    )
    found = pattern.findall(html)
    if found:
        return True, found[0]
    # Fall back to any cp3xx win wheel for that version
    loose = re.compile(rf"torch-{re.escape(version)}\+{cuda}-cp3\d+-[^\"']+\.whl")
    loose_found = loose.findall(html)
    return (bool(loose_found), loose_found[0] if loose_found else "none")


def local_python_tag() -> str:
    major, minor = sys.version_info[:2]
    return f"cp{major}{minor}"


def main() -> int:
    print(f"local interpreter tag: {local_python_tag()}")
    print("driver ceiling: CUDA 12.2 (so cu118/cu121 are the loadable lines)\n")
    results = []
    for version in TORCH_VERSIONS:
        row = {"torch": version, "cuda": {}}
        for cuda in CUDA_LINES:
            ok, detail = wheel_exists(version, cuda)
            row["cuda"][cuda] = {"available": ok, "wheel": detail}
            print(
                f"  torch {version:>7} + {cuda}: {'YES' if ok else 'no '}  {detail[:60]}"
            )
        results.append(row)

    out = Path("artifacts/torch_cuda_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    if sys.platform == "win32":
        print(
            "NOTE: wheels listed above are Windows x86_64 builds on the official index."
        )
    print(f"full report: {out}")

    # Report the newest loadable option, whatever it is.
    for row in reversed(results):
        for cuda in CUDA_LINES:
            if row["cuda"][cuda]["available"]:
                print(
                    f"\nNEWEST loadable candidate: torch {row['torch']} + {cuda}\n"
                    f"  (sm_52 support must still be verified by installing and "
                    f"running torch.cuda.is_available() plus a kernel launch)"
                )
                return 0
    print("\nNo Windows wheel found for any tested (torch, cuda) pair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
