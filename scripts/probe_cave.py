"""Probe CAVE/FlyWire access across datastack names (ops script, not shipped).

Reads the token from the vnr credential store (or VNR_FLYWIRE_TOKEN), sets
CAVE_TOKEN for the libraries that expect it, writes the cloudvolume secret
file, then tries datastacks in order and reports which ones open. Prints NO
secrets — only names, versions, and HTTP outcomes.

Usage: python scripts/probe_cave.py [--write-secret] [--datastack NAME ...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from vnr.connectome.credentials import load_token

DEFAULT_DATASTACKS = [
    "flywire_fafb_production",
    "flywire_fafb_public",
    "flywire_v141",
]


def ensure_cloudvolume_secret(token: str) -> Path:
    path = Path.home() / ".cloudvolume" / "secrets" / "cave-secret.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"token": token}), encoding="utf-8")
    return path


def probe(datastack: str, token: str) -> str:
    from caveclient import CAVEclient

    try:
        client = CAVEclient(datastack, auth_token=token)
    except Exception as exc:  # noqa: BLE001 — outcome reporting, not handling
        text = str(exc).splitlines()
        tail = " | ".join(text[-2:])[:220]
        return f"{datastack}: INIT-FAIL {type(exc).__name__} :: {tail}"
    try:
        version = client.materialize.get_versions()[:3]
    except Exception as exc:  # noqa: BLE001 — outcome reporting
        return f"{datastack}: OPEN-OK but version-list failed ({type(exc).__name__})"
    return f"{datastack}: OPEN-OK versions={version}"


def main(argv: list[str]) -> int:
    try:
        token = load_token()
    except KeyError as exc:
        print(f"no token: {exc}")
        return 2
    os.environ["CAVE_TOKEN"] = token
    names = DEFAULT_DATASTACKS
    if "--datastack" in argv:
        idx = argv.index("--datastack")
        names = argv[idx + 1 :]
    if "--write-secret" in argv:
        path = ensure_cloudvolume_secret(token)
        print(f"cloudvolume secret written: {path}")
    for name in names:
        print(probe(name, token), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
