"""PH3-WI03 tests: adapter contract + probe (live run deferred, documented)."""

import pytest

from vnr.backend.genn_backend import (
    GennNotAvailable,
    available,
    build_spec,
    require_available,
)
from vnr.backend.reference import StaticNetSpec


def test_probe_reports_boolean():
    # Pinned False on THIS machine (no PyGeNN possible — see module docstring).
    # If this ever flips True, phase 2 (live compile/run exactness) activates
    # and this test must be updated to demand it. A breaking test is the tripwire.
    assert available() is False


def test_require_available_fails_loud_with_unblock():
    with pytest.raises(GennNotAvailable, match="passworded sudo"):
        require_available()


def test_build_spec_carries_oracle_constants():
    spec = StaticNetSpec(n_neurons=64, ticks=600, out_degree=4, seed=5)
    built = build_spec(spec)
    assert built["backend"] == "genn" and built["device"] == "cpu"
    assert built["neuron_params"] == {
        "tau_ms": 20.0,
        "v_threshold": 1.0,
        "v_reset": 0.0,
        "refractory_ms": 2.0,
    }
    assert built["weight"] == spec.weight
    assert built["delay_ticks"] == spec.delay_ticks
    assert built["network"]["n_neurons"] == 64
    assert built["ticks"] == 600
