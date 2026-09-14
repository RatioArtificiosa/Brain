"""PH0 tests: config roundtrip/hash/validation (plan PH0-WI03)."""

import pytest

from vnr.config import config_hash, default_config, load_config, validate_config


def test_default_config_is_valid():
    validate_config(default_config())


def test_config_hash_stable_and_seed_sensitive():
    a, b = default_config(), default_config()
    assert config_hash(a) == config_hash(b)
    b["connectivity"]["seed"] = 999
    assert config_hash(a) != config_hash(b)


def test_missing_section_fails_loud_with_all_errors(tmp_path):
    bad = {"experiment": {"name": "x"}}  # missing id + 3 sections
    with pytest.raises(ValueError, match="missing section"):
        validate_config(bad)


def test_bad_dt_and_budgets_rejected():
    cfg = default_config()
    cfg["neuron"]["dt_ms"] = -1.0
    cfg["runtime"]["active_neuron_budget"] = 0
    with pytest.raises(ValueError) as exc:
        validate_config(cfg)
    assert "dt_ms" in str(exc.value) and "active_neuron_budget" in str(exc.value)


def test_load_config_roundtrip(tmp_path):
    import yaml

    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(default_config("E001")), encoding="utf-8")
    assert load_config(p)["experiment"]["id"] == "E001"
