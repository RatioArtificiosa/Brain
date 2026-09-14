"""Credential store (plan PH4-WI02b). One rule: secrets NEVER touch the repo,
the lab log, or stdout. Token lives in the user profile (``~/.vnr/``) or the
``VNR_FLYWIRE_TOKEN`` env var, which takes precedence at read time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = [
    "base_dir",
    "credentials_path",
    "has_token",
    "load_token",
    "save_token",
    "token_source",
]


def base_dir() -> Path:
    """User profile dir; ``VNR_HOME`` overrides for tests."""
    override = os.environ.get("VNR_HOME", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".vnr"


def credentials_path() -> Path:
    return base_dir() / "credentials.json"


def save_token(token: str) -> Path:
    """Persist the FlyWire token. Returns the path written (never the token)."""
    if not isinstance(token, str) or not token.strip():
        raise ValueError("token must be a non-empty string")
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"flywire_token": token.strip()}), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # Windows ACLs: best effort, documented
    return path


def token_source() -> str:
    """Where a token would come from: 'env', 'file', or 'none' (no secrets)."""
    if os.environ.get("VNR_FLYWIRE_TOKEN", "").strip():
        return "env"
    try:
        data = json.loads(credentials_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "none"
    return "file" if isinstance(data, dict) and data.get("flywire_token") else "none"


def has_token() -> bool:
    return token_source() in ("env", "file")


def load_token() -> str:
    """Return the token, or raise KeyError naming the fix (never the value)."""
    env = os.environ.get("VNR_FLYWIRE_TOKEN", "").strip()
    if env:
        return env
    try:
        data = json.loads(credentials_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    token = data.get("flywire_token", "") if isinstance(data, dict) else ""
    if not token:
        raise KeyError(
            "no FlyWire token: run `vnr dataset auth` or set VNR_FLYWIRE_TOKEN"
        )
    return token
