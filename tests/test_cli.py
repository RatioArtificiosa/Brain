"""PH0 tests: CLI contract — doctor works, future commands fail honest (PH0-WI02)."""

from click.testing import CliRunner

from vnr.cli import cli


def test_doctor_reports_ready_and_status_block():
    result = CliRunner().invoke(cli, ["doctor"])
    assert result.exit_code == 0
    for token in ("VNR SYSTEM DIAGNOSTICS", "STATUS: READY", "M5000", "8.0 GB"):
        assert token in result.output, result.output


def test_unimplemented_commands_exit_2_with_phase_pointer():
    result = CliRunner().invoke(cli, ["simulate"])
    assert result.exit_code == 2
    assert "PH1" in result.output
