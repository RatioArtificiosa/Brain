"""PH3-WI03 phase 2 test: GeNN WSL run bit-exact vs oracle (slow, real)."""

import json
import subprocess

from vnr.backend.reference import StaticNetSpec, run_explicit

WSL_RUNNER = "/mnt/g/BRAIN/VNR/vnr/src/vnr/backend/genn_wsl_runner.py"


def _run_genn_wsl(spec: StaticNetSpec, tmp_path):
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / "out.json"
    spec_path.write_text(
        json.dumps(
            {
                "n_neurons": spec.n_neurons,
                "ticks": spec.ticks,
                "out_degree": spec.out_degree,
                "weight": spec.weight,
                "delay_ticks": spec.delay_ticks,
                "seed": spec.seed,
                "drive_density": spec.drive_density,
                "drive_amplitude": spec.drive_amplitude,
            }
        ),
        encoding="utf-8",
    )

    def wslpath(p) -> str:
        text = str(p)
        drive, rest = text[0].lower(), text[2:].replace("\\", "/")
        assert text[1] == ":", f"expected absolute Windows path, got {text!r}"
        return f"/mnt/{drive}{rest}"

    win_spec = wslpath(spec_path)
    win_out = wslpath(out_path)
    proc = subprocess.run(
        ["wsl", "-d", "Ubuntu", "python3", WSL_RUNNER, win_spec, win_out],
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert proc.returncode == 0, f"WSL runner failed:\n{proc.stderr[-2000:]}"
    return json.loads(out_path.read_text(encoding="utf-8"))


def test_genn_wsl_matches_oracle_exactly(tmp_path):
    spec = StaticNetSpec(n_neurons=32, ticks=300, out_degree=4, seed=3)
    expected = run_explicit(spec)
    assert expected.total_spikes > 0, "spec must ignite or the test is vacuous"
    got = _run_genn_wsl(spec, tmp_path)
    got_spikes = {int(k): v for k, v in got["spikes"].items()}
    assert got_spikes == expected.spikes
    for nid in range(spec.n_neurons):
        assert got["final_v"][str(nid)] == expected.final_v[nid]


def test_genn_available_probe_documents_wsl_home():
    # The live backend lives in WSL; this documents where. It passes iff the
    # runner file exists (the heavy proof is the exactness test above).
    from pathlib import Path

    assert Path(WSL_RUNNER.replace("/mnt/g/", "G:/").replace("/", "\\")).exists()
