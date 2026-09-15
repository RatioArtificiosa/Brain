"""PH0 tests: CLI contract — doctor works on this machine (PH0-WI02).

The companion file ``test_cli_surface.py`` covers the full command set. This
file keeps the original PH0 contract that started the project (the doctor
status block) and the PH0-era *spirit* of "no stubs that lie" — updated to
the post-PH5 reality, where the stub itself is gone.
"""

from click.testing import CliRunner

from vnr.cli import cli


def test_doctor_reports_ready_and_status_block():
    result = CliRunner().invoke(cli, ["doctor"])
    assert result.exit_code == 0
    for token in ("VNR SYSTEM DIAGNOSTICS", "STATUS: READY"):
        assert token in result.output, result.output
    # Hardware tokens are machine-dependent; assert on the SHAPE instead when
    # this is not the reference machine, so the suite travels.
    if "M5000" not in result.output:
        for label in ("CPU ", "RAM ", "GPU ", "CUDA "):
            assert label in result.output, result.output


def test_simulate_no_longer_claims_to_be_unimplemented():
    """Regression guard for the real defect: stale phase pointers.

    PH0 shipped honest exit-2 stubs. PH1-PH3 then COMPLETED while the stubs
    kept claiming "not implemented until PH1", so the CLI was lying about the
    project's own state. The command must now do the work instead.
    """
    result = CliRunner().invoke(cli, ["simulate", "--neurons", "48", "--ticks", "120"])
    assert result.exit_code == 0, result.output
    assert "not implemented" not in result.output.lower()
    assert "PH1" not in result.output
