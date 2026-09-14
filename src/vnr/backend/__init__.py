"""Reference backend: exact oracles all later backends are measured against."""

from vnr.backend.reference import (
    RunResult,
    StaticNetSpec,
    VirtualStats,
    build_drive,
    run_explicit,
    run_virtualized,
)

__all__ = [
    "RunResult",
    "StaticNetSpec",
    "VirtualStats",
    "build_drive",
    "run_explicit",
    "run_virtualized",
]
