"""Console output helpers: correct text, honest formatting (PH5-WI00).

Two problems this module solves, both of which made the CLI read as a
prototype rather than a product:

1. **Encoding.** On Windows the console defaults to a legacy code page, so an
   em-dash or a micro sign printed through :func:`click.echo` decomposes into
   U+FFFD. Every user-visible string therefore goes through :func:`echo`,
   which reconfigures the streams to UTF-8 once and falls back to an ASCII
   transliteration if that is not possible.
2. **Consistent numeric voice.** Measurements are the product's currency, so
   they are formatted identically everywhere: thousands separators, sensible
   precision, and units that never lie about what was measured.
"""

from __future__ import annotations

import sys

import click

__all__ = ["echo", "fmt_bytes", "fmt_count", "fmt_seconds", "rule", "section"]

_ASCII_FALLBACK = {
    "\u2014": "--",  # em dash
    "\u2013": "-",  # en dash
    "\u2192": "->",  # right arrow
    "\u00b5": "u",  # micro sign
    "\u00b7": "-",  # middle dot
    "\u2248": "~",  # approx
    "\u2713": "OK",  # check mark
    "\u2717": "X",  # ballot X
    "\u2588": "#",  # full block (sparklines)
    "\u2584": "#",
    "\u2580": "#",
    "\u2591": ".",
    "\u2592": "+",
    "\u2593": "*",
}


def _configure_utf8() -> bool:
    """Best-effort UTF-8 on stdout/stderr. Returns True when safe to emit
    non-ASCII."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            return False
    encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    return "utf" in encoding


_UTF8_OK = _configure_utf8()


def _safe(text: str) -> str:
    """Transliterate to ASCII when the console cannot carry the characters."""
    if _UTF8_OK:
        return text
    for fancy, plain in _ASCII_FALLBACK.items():
        text = text.replace(fancy, plain)
    return text.encode("ascii", "replace").decode("ascii")


def echo(text: str = "", err: bool = False) -> None:
    """Print user-visible text, safe on any console encoding."""
    click.echo(_safe(text), err=err)


def section(title: str) -> None:
    """A titled block: the CLI's main structural device."""
    echo()
    echo(title)
    echo("-" * max(8, len(title)))


def rule() -> None:
    """A full-width thin separator."""
    echo("-" * 72)


def fmt_count(value: int) -> str:
    """Thousands-separated integer (measurements must be readable)."""
    return f"{value:,}"


def fmt_bytes(value: int) -> str:
    """Binary units, honest about precision at each magnitude."""
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def fmt_seconds(value: float) -> str:
    """Human-readable duration with enough precision to compare runs."""
    if value < 1e-3:
        return f"{value * 1e6:.0f} us"
    if value < 1.0:
        return f"{value * 1e3:.1f} ms"
    if value < 60.0:
        return f"{value:.2f} s"
    minutes, seconds = divmod(value, 60.0)
    return f"{int(minutes)}m {seconds:04.1f}s"
