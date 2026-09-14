"""PH4-WI02b tests: credential store + dataset CLI (secrets never surface)."""

import json

import pytest
from click.testing import CliRunner

from vnr.cli import cli
from vnr.connectome.credentials import (
    has_token,
    load_token,
    save_token,
    token_source,
)


def test_save_load_roundtrip_isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VNR_HOME", str(tmp_path))
    monkeypatch.delenv("VNR_FLYWIRE_TOKEN", raising=False)
    assert token_source() == "none" and not has_token()
    path = save_token("  secret-123  ")
    assert path.parent == tmp_path
    assert token_source() == "file" and has_token()
    assert load_token() == "secret-123"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == {"flywire_token": "secret-123"}


def test_env_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("VNR_HOME", str(tmp_path))
    monkeypatch.setenv("VNR_FLYWIRE_TOKEN", "env-token")
    save_token("file-token")
    assert token_source() == "env"
    assert load_token() == "env-token"


def test_missing_token_names_fix(tmp_path, monkeypatch):
    monkeypatch.setenv("VNR_HOME", str(tmp_path))
    monkeypatch.delenv("VNR_FLYWIRE_TOKEN", raising=False)
    try:
        load_token()
        raise AssertionError("must raise")
    except KeyError as exc:
        assert "vnr dataset auth" in str(exc)
    with pytest.raises(ValueError, match="non-empty"):
        save_token("   ")


def test_cli_auth_and_status_never_print_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("VNR_HOME", str(tmp_path))
    monkeypatch.delenv("VNR_FLYWIRE_TOKEN", raising=False)
    runner = CliRunner()
    result = runner.invoke(cli, ["dataset", "auth", "--token", "super-secret-xyz"])
    assert result.exit_code == 0
    assert "super-secret-xyz" not in result.output
    assert "stored" in result.output
    result = runner.invoke(cli, ["dataset", "status"])
    assert result.exit_code == 0
    assert "super-secret-xyz" not in result.output
    assert "configured" in result.output
