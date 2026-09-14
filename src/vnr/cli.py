"""VNR command line. `vnr doctor` is the first command that must work (plan PH0-WI02).

Subcommands not yet implemented exit 2 with an honest pointer to their phase —
never a fake success.
"""

from __future__ import annotations

import click

from vnr import __version__
from vnr.hardware import derive_budget, discover_hardware, format_bytes

_NOT_YET = {
    "connectome": "PH4",
    "generate": "PH4",
    "simulate": "PH1",
    "benchmark": "PH3",
    "experiment": "PH5",
    "report": "PH5",
}


@click.group()
@click.version_option(__version__, prog_name="vnr")
def cli() -> None:
    """Virtual Neural Runtime — procedurally generated neural simulation."""


@cli.command()
def doctor() -> None:
    """Probe hardware and report readiness (plan §10 STATUS block)."""
    prof = discover_hardware()
    budget = derive_budget(prof)
    gpu = prof.gpu_name or "not detected"
    vram = format_bytes(prof.gpu_vram_bytes) if prof.gpu_vram_bytes else "n/a"
    cuda = prof.cuda_version or (
        "unavailable" if prof.torch_version else "torch not installed"
    )
    click.echo("VNR SYSTEM DIAGNOSTICS")
    click.echo(
        f"CPU ..................... {prof.cpu_name} ({prof.cpu_threads} threads)"
    )
    click.echo(f"RAM ..................... {format_bytes(prof.ram_bytes)}")
    click.echo(f"NVIDIA GPU .............. {gpu}")
    click.echo(f"VRAM .................... {vram}")
    click.echo(f"CUDA .................... {cuda}")
    click.echo(
        f"PyTorch ................. {prof.torch_version or 'missing (optional)'}"
    )
    click.echo("FlyWire dataset ......... missing (PH4)")
    for note in budget.notes:
        click.echo(f"NOTE: {note}")
    ready = prof.ram_bytes > 0 and prof.cpu_threads > 0
    click.echo(f"STATUS: {'READY' if ready else 'NOT READY'}")


@cli.group()
def dataset() -> None:
    """Connectome dataset operations (PH4-WI02b)."""


@dataset.command()
@click.option("--token", default=None, help="FlyWire token (else secure prompt).")
def auth(token: str | None) -> None:
    """Store the FlyWire token in the user profile (never in the repo)."""
    from vnr.connectome.credentials import save_token

    value = token or click.prompt("FlyWire token", hide_input=True)
    path = save_token(value)
    click.echo(f"token stored: {path} (source: file)")


@dataset.command()
def status() -> None:
    """Show dataset credential state (never prints secrets)."""
    from vnr.connectome.credentials import has_token, token_source

    if has_token():
        click.echo(f"FlyWire token: configured (source: {token_source()})")
    else:
        click.echo("FlyWire token: missing — run `vnr dataset auth`")


def _stub(name: str) -> None:
    phase = _NOT_YET[name]
    click.echo(
        f"'vnr {name}' is not implemented until {phase} (see 01-master-plan.md)",
        err=True,
    )
    raise SystemExit(2)


for _name in _NOT_YET:
    cli.command(name=_name)(lambda _n=_name: _stub(_n))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
