"""PH1-WI02 tests: LIF dynamics vs closed form on integer ticks."""

import math

import pytest

from vnr.core.neuron import LIFNeuron, LIFParams, LIFState, closed_form_voltage


def test_decay_matches_closed_form_no_input():
    p = LIFParams(tau_ms=20.0, dt_ms=0.1, v_threshold=100.0)
    n = LIFNeuron(params=p)
    n.state.v = 5.0
    steps = 50
    for tick in range(steps):
        assert not n.step(tick, 0.0)
    assert n.state.v == pytest.approx(closed_form_voltage(5.0, 0.0, steps, p.decay))


def test_constant_current_converges_to_steady_state():
    p = LIFParams(tau_ms=10.0, dt_ms=0.1, v_threshold=100.0)
    n = LIFNeuron(params=p)
    i_syn = 3.0
    steps = 2000
    for tick in range(steps):
        n.step(tick, i_syn)
    assert n.state.v == pytest.approx(i_syn, rel=1e-6)
    assert n.state.v == pytest.approx(closed_form_voltage(0.0, i_syn, steps, p.decay))


def test_single_step_matches_analytic_update():
    p = LIFParams(tau_ms=20.0, dt_ms=0.1, v_threshold=100.0)
    n = LIFNeuron(params=p)
    n.state.v = 1.5
    n.step(0, 0.5)
    assert n.state.v == pytest.approx(0.5 + (1.5 - 0.5) * math.exp(-0.1 / 20.0))


def test_spike_resets_and_arms_refractory():
    p = LIFParams(tau_ms=20.0, dt_ms=0.1, v_threshold=1.0, refractory_ms=2.0)
    n = LIFNeuron(params=p)
    assert p.refractory_ticks == 20
    # One tick from rest rises only I*(1-decay) ≈ 0.005*I, so drive hard.
    assert n.step(0, 1000.0) is True
    assert n.state.v == pytest.approx(p.v_reset)
    assert n.state.refractory_until_tick == 0 + 1 + 20
    # Held far above threshold through refractoriness: no spike, clamped.
    for tick in range(1, 21):
        assert not n.step(tick, 1000.0)
        assert n.state.v == pytest.approx(p.v_reset)


def test_subthreshold_never_spikes():
    p = LIFParams()
    n = LIFNeuron(params=p)
    assert n.run(0, [0.5] * 5000) == []


def test_spike_tick_predictable_for_step_current():
    p = LIFParams(tau_ms=20.0, dt_ms=0.1, v_threshold=1.0, refractory_ms=0.0)
    i_syn = 2.0
    # Closed form: smallest n with I*(1-decay^n) >= threshold.
    expected = next(
        n for n in range(1, 10000) if i_syn * (1.0 - p.decay**n) >= p.v_threshold
    )
    n = LIFNeuron(params=p)
    spikes = n.run(0, [i_syn] * (expected + 5))
    assert spikes[0] == expected - 1  # zero-based tick of the nth update


def test_deterministic_replay_identical():
    p = LIFParams()
    # Sustained blocks: single isolated ticks can only lift V by I*(1-decay),
    # so replay uses supra-threshold steady-state drive (I=3 > threshold 1).
    drive = ([3.0] * 60 + [0.0] * 60) * 10
    first = LIFNeuron(params=p).run(0, drive)
    second = LIFNeuron(params=p).run(0, drive)
    assert first == second
    assert len(first) > 0


def test_refractory_holds_at_reset_despite_input():
    p = LIFParams(refractory_ms=1.0)  # 10 ticks
    n = LIFNeuron(params=p, state=LIFState(v=0.9, refractory_until_tick=10))
    for tick in range(10):
        assert not n.step(tick, 50.0)
    assert n.state.v == pytest.approx(p.v_reset)


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        LIFParams(tau_ms=0.0)
    with pytest.raises(ValueError):
        LIFParams(dt_ms=-0.1)
    with pytest.raises(ValueError):
        LIFParams(v_threshold=0.0, v_reset=0.0)
    with pytest.raises(ValueError):
        LIFParams(refractory_ms=-1.0)
