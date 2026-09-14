"""YAML experiment configuration with validation and stable content hashing.

Every run record embeds ``config_hash`` (sha256 over canonical YAML) so results
are traceable to the exact configuration that produced them (plan §55).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1

_REQUIRED_TOP_LEVEL = ("experiment", "network", "neuron", "runtime")


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a VNR experiment YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(data)
    return data


def validate_config(data: dict[str, Any]) -> None:
    """Raise ValueError listing every schema violation found (fail loud, all at once)."""
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    errors = [f"missing section: {key}" for key in _REQUIRED_TOP_LEVEL if key not in data]
    exp = data.get("experiment", {})
    if not isinstance(exp, dict) or "id" not in exp:
        errors.append("experiment.id is required")
    neuron = data.get("neuron", {})
    if isinstance(neuron, dict) and neuron.get("dt_ms", 0.1) is not None:
        try:
            if float(neuron.get("dt_ms", 0.1)) <= 0:
                errors.append("neuron.dt_ms must be positive")
        except (TypeError, ValueError):
            errors.append("neuron.dt_ms must be a number")
    runtime = data.get("runtime", {})
    if isinstance(runtime, dict):
        for key in ("active_neuron_budget", "gpu_memory_budget_mb"):
            if key in runtime and (not isinstance(runtime[key], int) or runtime[key] <= 0):
                errors.append(f"runtime.{key} must be a positive integer")
    if errors:
        raise ValueError("invalid VNR config:\n- " + "\n- ".join(errors))


def config_hash(data: dict[str, Any]) -> str:
    """Stable sha256 over canonical (sorted-key) YAML serialization."""
    canonical = yaml.safe_dump(data, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def default_config(experiment_id: str = "E000") -> dict[str, Any]:
    """Minimal valid config used by tests and `vnr experiment run --dry-run`."""
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": {"id": experiment_id, "name": "smoke"},
        "network": {"generator": "toy", "target_neurons": 1000, "target_synapses": 10000},
        "neuron": {"model": "lif", "dt_ms": 0.1},
        "connectivity": {"mode": "procedural", "seed": 12345},
        "runtime": {
            "active_neuron_budget": 500,
            "gpu_memory_budget_mb": 6000,
            "eviction_policy": "activity_decay",
        },
        "plasticity": {"enabled": False},
        "logging": {"spike_sampling": 0.001, "traces": True},
    }
