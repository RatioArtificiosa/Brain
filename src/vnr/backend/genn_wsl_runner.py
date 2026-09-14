"""GeNN WSL runner: StaticNetSpec in, spike JSON out (PH3-WI03 phase 2).

Runs INSIDE WSL (``wsl -d Ubuntu python3 <this file> spec.json out.json``).
Builds the custom-LIF network with explicit sparse connectivity + per-tick DC
drive, records all spikes, and dumps ``{spikes, final_v}``. The Windows-side
test compares this output against ``run_explicit`` for bit-exactness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# vnr lives on the Windows side, mounted in WSL.
sys.path.insert(0, "/mnt/g/BRAIN/VNR/vnr/src")

import numpy as np

from vnr.backend.reference import StaticNetSpec, build_drive
from vnr.core.procedural import ConnectivityParams, ProceduralConnectivity


def build_lif(model_class, create_fn):
    return create_fn(
        "vnr_lif",
        params=["Decay", "Vthr", "Vreset", "RefrTicks"],
        vars=[("V", "scalar"), ("Refr", "scalar")],
        sim_code="""
        if (Refr > 0.5) {
            Refr -= 1.0;
            V = Vreset;
        }
        else {
            V = Isyn + (V - Isyn) * Decay;
        }
        """,
        threshold_condition_code="V >= Vthr",
        reset_code="V = Vreset; Refr = RefrTicks;",
    )


def main(spec_path: str, out_path: str) -> None:
    # This module executes ONLY inside WSL (see test_genn_live.py): on Windows
    # the import below is unresolvable by construction, and the exactness test
    # — not the type checker — verifies the GeNN API usage.
    from pygenn import (  # pyright: ignore
        GeNNModel,
        create_current_source_model,
        create_neuron_model,
        init_postsynaptic,
        init_var,
        init_weight_update,
    )

    spec_dict = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    spec = StaticNetSpec(**spec_dict)
    decay = 2.718281828459045 ** (-0.1 / 20.0)

    rule = ProceduralConnectivity(
        params=ConnectivityParams(
            global_seed=spec.seed,
            target_population_id=0,
            connectivity_version=1,
            out_degree=spec.out_degree,
            weight=spec.weight,
            delay_ticks=spec.delay_ticks,
        )
    )
    pre, post = [], []
    for nid in range(spec.n_neurons):
        for tgt in rule.sample_targets(nid):
            pre.append(nid)
            post.append(tgt % spec.n_neurons)
    drive = build_drive(spec)

    # "double": float32 integration drifts ~1e-6 over hundreds of ticks
    # (measured), so the oracle gate demands double precision. The precision
    # is part of the model NAME: GeNN caches compiled code by name, and a
    # stale float build once masqueraded as a double failure here.
    model = GeNNModel("double", f"vnr_static_{spec.seed}_double")
    model.dt = 0.1
    lif = build_lif(model, create_neuron_model)
    pop = model.add_neuron_population(
        "pop",
        spec.n_neurons,
        lif,
        {"Decay": decay, "Vthr": 1.0, "Vreset": 0.0, "RefrTicks": 20.0},
        {
            "V": init_var("Constant", {"constant": 0.0}),
            "Refr": init_var("Constant", {"constant": 0.0}),
        },
    )
    pop.spike_recording_enabled = True
    dc_model = create_current_source_model(
        "vnr_dc",
        vars=[("magnitude", "scalar")],
        injection_code="injectCurrent(magnitude);",
    )
    dc = model.add_current_source("drive", dc_model, pop, {}, {"magnitude": 0.0})
    syn = model.add_synapse_population(
        "syn",
        "SPARSE",
        pop,
        pop,
        init_weight_update("StaticPulseConstantWeight", {"g": spec.weight}),
        init_postsynaptic("DeltaCurr"),
    )
    syn.set_sparse_connections(
        np.asarray(pre, dtype=np.int32), np.asarray(post, dtype=np.int32)
    )
    # Measured (delay probe): GeNN delivers at spike_tick + 1 + axonal steps,
    # so oracle delay D needs axonal D-1. Same-tick delivery (D=0) is not
    # expressible in GeNN's event pipeline — fail loud, never silently shift.
    if spec.delay_ticks < 1:
        raise ValueError("GeNN runner requires delay_ticks >= 1")
    syn.max_dendritic_delay_timesteps = spec.delay_ticks
    syn.axonal_delay_steps = spec.delay_ticks - 1
    model.build()
    model.load(num_recording_timesteps=spec.ticks)

    mag = np.zeros(spec.n_neurons)
    for t in range(spec.ticks):
        mag.fill(0.0)
        for nid in drive[t]:
            mag[nid] = spec.drive_amplitude
        dc.vars["magnitude"].values = mag
        dc.vars["magnitude"].push_to_device()
        model.step_time()

    model.pull_recording_buffers_from_device()
    rec = pop.spike_recording_data
    times_ms, ids = rec[0]
    spikes: dict[int, list[int]] = {nid: [] for nid in range(spec.n_neurons)}
    # GeNN records spike times in MILLISECONDS: convert with the model dt.
    # (A truncation here once masqueraded as an 81-vs-8 timing anomaly.)
    for t_ms, nid in zip(times_ms.tolist(), ids.tolist()):
        tick = round(float(t_ms) / 0.1)
        if 0 <= tick < spec.ticks:
            spikes[int(nid)].append(tick)
    pop.vars["V"].pull_from_device()
    final_v = {nid: float(v) for nid, v in enumerate(pop.vars["V"].values.tolist())}
    Path(out_path).write_text(
        json.dumps({"spikes": spikes, "final_v": final_v}), encoding="utf-8"
    )
    print(f"DONE spikes={sum(len(v) for v in spikes.values())}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
