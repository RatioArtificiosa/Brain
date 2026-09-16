"""Which torch CUDA builds actually have Python 3.13 (cp313) Windows wheels?

The previous probe's regex matched whatever cp-tag appeared first (cp310),
which is not what this machine runs. Python 3.13 is the local interpreter, so
only cp313 wheels are installable here. This narrows the search to the pairs
that could actually be installed.
"""

import re
import urllib.request

INDEX = "https://download.pytorch.org/whl"
VERSIONS = ["2.4.1", "2.5.1", "2.6.0", "2.7.1", "2.8.0", "2.9.1"]
CUDA_LINES = ["cu118", "cu121", "cu124", "cu126", "cu128"]

print("Looking for cp313 win_amd64 wheels (this interpreter is Python 3.13)\n")
found = []
for cuda in CUDA_LINES:
    try:
        with urllib.request.urlopen(f"{INDEX}/{cuda}/torch/", timeout=60) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        print(f"  {cuda}: index unreachable ({type(exc).__name__})")
        continue
    for ver in VERSIONS:
        hits = re.findall(rf"torch-{re.escape(ver)}\+{cuda}-cp313-[^\"'>]+\.whl", html)
        for hit in hits:
            found.append((ver, cuda, hit))
            print(f"  torch {ver:>6} + {cuda}: {hit}")

print()
windows = [f for f in found if "win_amd64" in f[2]]
linux = [f for f in found if "win_amd64" not in f[2]]

if windows:
    print("WINDOWS cp313 CUDA wheels (installable on this machine):")
    for ver, cuda, wheel in windows:
        print(f"  torch {ver} + {cuda}  ->  {wheel}")
    print()
    print("Driver 537.99 supports CUDA <= 12.2, so cu118 and cu121 are the")
    print("only loadable lines. Next: install the newest and test a kernel.")
else:
    print("NO Windows cp313 CUDA wheels exist for the tested versions.")
    print("The cp313 builds that DO exist are Linux-only:")
    for ver, cuda, wheel in linux:
        print(f"  torch {ver} + {cuda}: {wheel}")
    print()
    print("CONCLUSION: on Windows + Python 3.13 the CUDA path is closed by")
    print("wheel availability, not by the GPU. Options that remain: (a) a")
    print("second interpreter at 3.11/3.12 where Windows CUDA wheels exist,")
    print("(b) WSL2 Linux (cp313 wheels DO exist there) - which also matches")
    print("the GeNN path already built, or (c) accept CPU and say so.")
