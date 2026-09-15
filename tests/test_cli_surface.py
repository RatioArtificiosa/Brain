"""PH5-WI00 tests: the CLI is the product surface, and it must tell the truth.

These tests replace the PH0 "stub exits 2" contract. Every command that has
real machinery behind it must RUN, print MEASURED numbers, and exit 0. The
guiding rule from the checklist ("no stubs that lie") applies in both
directions: a command must not claim to be unimplemented when it works, and
must not claim success when it cannot do the work.

Commands are exercised through click's CliRunner (in-process, fast, and it
captures output the way a user sees it). Anything touching the real FlyWire
pilot is skipped when the corpus is absent, so a fresh clone stays green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

pytest.importorskip("pyarrow", reason="dataset commands need the [data] extra")

from vnr.cli import cli

PILOT = Path(r"G:\BRAIN\VNR\data\flywire_v783")
has_pilot = pytest.mark.skipif(
    not sorted(PILOT.glob("chunk-*.parquet")), reason="FlyWire pilot not present"
)


def run(*args: str):
    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


# ------------------------------------------------------------------ doctor


def test_doctor_reports_real_dataset_state_not_a_hardcoded_missing():
    """doctor must MEASURE the dataset, not print a placeholder.

    On this machine 992,991 real rows are on disk; the old output claimed
    "missing (PH4)". If the pilot is absent the honest answer is still a real
    check, not a constant.
    """
    out = run("doctor").output
    assert "VNR SYSTEM DIAGNOSTICS" in out
    assert "STATUS: READY" in out
    assert "missing (PH4)" not in out  # the lie this test exists to kill
    if sorted(PILOT.glob("chunk-*.parquet")):
        assert "rows" in out.lower()


def test_doctor_output_has_no_mojibake():
    """Console output must be valid text: no U+FFFD replacement characters.

    The old doctor printed em-dashes through a cp1252 console as '?', which
    reads as a broken product.
    """
    out = run("doctor").output
    assert "\ufffd" not in out, "replacement char in doctor output"


# ----------------------------------------------------------------- simulate


def test_simulate_runs_and_reports_measured_numbers():
    """simulate is the demo a newcomer runs first: it must actually simulate."""
    result = run("simulate", "--neurons", "128", "--ticks", "300")
    assert result.exit_code == 0, result.output
    out = result.output
    # Real measurements, not placeholders.
    assert "spikes" in out.lower()
    assert "resident" in out.lower()
    assert "eviction" in out.lower()
    # The virtualized path must stay bounded: that is the whole thesis.
    assert "peak resident" in out.lower() or "max resident" in out.lower()


def test_simulate_is_deterministic_for_a_fixed_seed():
    """Same seed -> same brain (D2.5). Two runs must agree exactly."""

    def spikes(seed: str) -> str:
        out = run(
            "simulate", "--neurons", "96", "--ticks", "200", "--seed", seed
        ).output
        return next(ln for ln in out.splitlines() if "spikes" in ln.lower())

    assert spikes("11") == spikes("11")
    assert spikes("11") != spikes("12")


def test_simulate_json_is_machine_readable(tmp_path):
    """--json makes the command scriptable (the report chain consumes it)."""
    payload = tmp_path / "sim.json"
    result = run(
        "simulate", "--neurons", "64", "--ticks", "150", "--json", str(payload)
    )
    assert result.exit_code == 0, result.output
    data = json.loads(payload.read_text(encoding="utf-8"))
    assert data["total_spikes"] >= 0
    assert data["n_neurons"] == 64
    assert "max_resident" in data
    assert data["max_resident"] <= 64  # cannot be more resident than exists


# --------------------------------------------------------------- benchmark


def test_benchmark_runs_the_real_backends():
    result = run("benchmark", "--neurons", "64", "--ticks", "200")
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "explicit" in out and "numpy" in out
    assert "x" in out or "speedup" in out


# --------------------------------------------------------------- generate


def test_generate_lists_all_eight_generators():
    out = run("generate", "--list").output
    for name in (
        "replication",
        "duplication_divergence",
        "modular",
        "spatial",
        "motif_fill",
        "population",
        "hybrid",
        "random_control",
    ):
        assert name in out, name


def test_generate_writes_a_graph_and_its_lineage(tmp_path):
    """Generators must emit metadata (§16) so any graph is traceable."""
    dest = tmp_path / "grown.json"
    result = run("generate", "replication", "--copies", "2", "--out", str(dest))
    assert result.exit_code == 0, result.output
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data["metadata"]["generator"] == "replication"
    assert data["metadata"]["n_out"] > 0
    assert data["metadata"]["seed"] == 0
    assert "edges" in data


def test_generate_rejects_an_unknown_generator():
    result = run("generate", "not_a_generator")
    assert result.exit_code != 0
    assert "unknown generator" in result.output.lower()


# --------------------------------------------------------------- connectome


@has_pilot
def test_connectome_stats_reads_the_real_corpus():
    result = run("connectome", "stats")
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "rows" in out and "neurons" in out
    # Measured by reading the files, so it must exceed the committed row count.
    assert "992,991" in result.output or "992991" in result.output


@has_pilot
def test_connectome_census_reports_triples_and_is_exact():
    result = run("connectome", "census", "--limit-chunks", "1")
    assert result.exit_code == 0, result.output
    assert "exact" in result.output.lower()
    assert "triple" in result.output.lower()


@has_pilot
def test_connectome_census_ndjson_is_structured(tmp_path):
    dest = tmp_path / "census.json"
    result = run("connectome", "census", "--limit-chunks", "1", "--json", str(dest))
    assert result.exit_code == 0, result.output
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert data["exact"] is True
    assert data["triples_counted"] > 0
    assert isinstance(data["motifs"], list)


# --------------------------------------------------------------- experiment


def test_experiment_four_way_runs_and_writes_a_record(tmp_path):
    """PH5-WI01: the four-way comparison, producing a run record (§55)."""
    run_dir = tmp_path / "E004-test"
    result = run(
        "experiment",
        "four-way",
        "--neurons",
        "96",
        "--ticks",
        "200",
        "--out",
        str(run_dir),
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "run.json").exists(), "run record missing (plan §55)"
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    for key in ("run_id", "config_hash", "seed", "commit", "created"):
        assert key in record, key
    assert record["experiment_id"] == "E004-test"
    # The four conditions of §37, all present.
    conds = {c["condition"] for c in record["conditions"]}
    assert {"explicit_dense", "sparse", "procedural", "procedural_virtualized"} <= conds


def test_experiment_four_way_reports_compression_and_fidelity():
    result = run(
        "experiment", "four-way", "--neurons", "64", "--ticks", "150", "--no-write"
    )
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "fidelity" in out
    assert "resident" in out or "memory" in out or "compression" in out


# ------------------------------------------------------------------- report


def test_report_renders_a_run_directory_as_html(tmp_path):
    run_dir = tmp_path / "E004-test"
    run(
        "experiment",
        "four-way",
        "--neurons",
        "64",
        "--ticks",
        "150",
        "--out",
        str(run_dir),
    )
    html = tmp_path / "r.html"
    result = run("report", str(run_dir), "--out", str(html))
    assert result.exit_code == 0, result.output
    text = html.read_text(encoding="utf-8")
    assert "<html" in text.lower()
    assert "E004-test" in text


def test_report_separates_hypothesis_observation_interpretation(tmp_path):
    """Plan §5 / §83: reports must never blur claim and measurement."""
    run_dir = tmp_path / "E004-test"
    run(
        "experiment",
        "four-way",
        "--neurons",
        "64",
        "--ticks",
        "150",
        "--out",
        str(run_dir),
    )
    json_out = tmp_path / "r.json"
    run("report", str(run_dir), "--out", str(json_out))
    data = json.loads(json_out.read_text(encoding="utf-8"))
    for key in ("hypothesis", "observation", "interpretation"):
        assert key in data, key


# --------------------------------------------------------------------- rules


def test_no_command_claims_a_false_phase():
    """Regression guard for the actual bug: stale phase pointers."""
    for cmd in (
        "simulate",
        "benchmark",
        "experiment",
        "report",
        "connectome",
        "generate",
    ):
        out = run(cmd, "--help").output
        assert "not implemented" not in out.lower(), f"{cmd} still a stub"


def test_help_lists_every_documented_command():
    out = run("--help").output
    for cmd in (
        "doctor",
        "dataset",
        "connectome",
        "generate",
        "simulate",
        "benchmark",
        "experiment",
        "report",
    ):
        assert cmd in out
