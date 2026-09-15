"""Pilot corpus access: read the real FlyWire data that is on disk.

The pilot is NOT committed (see plan §3), so every consumer must degrade
honestly when it is absent rather than pretend. This module is the single
place that knows the on-disk layout, so the CLI, the census script, and the
tests cannot drift apart.

Layout: ``<root>/chunk-<offset:012d>.parquet`` with columns
``pre, post, nt, conf`` (uint64 root ids, uint8 NT code, float32 confidence).
``cursor.json`` holds the next fetch offset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PilotCorpus",
    "PilotNotFound",
    "default_pilot_path",
    "load_parquet_rows",
]


class PilotNotFound(FileNotFoundError):
    """Raised when the corpus directory has no chunks (message names the fix)."""


def default_pilot_path() -> Path:
    """The documented pilot location (data/ next to the docs repo)."""
    return Path(r"G:\BRAIN\VNR\data\flywire_v783")


@dataclass(frozen=True)
class PilotCorpus:
    """A located pilot corpus: chunk paths plus the resume cursor."""

    root: Path
    chunks: tuple[Path, ...]
    cursor_offset: int = 0

    @classmethod
    def discover(cls, root: Path | None = None) -> PilotCorpus:
        """Locate the corpus, or raise :class:`PilotNotFound` with the fix."""
        path = Path(root) if root is not None else default_pilot_path()
        chunks = tuple(sorted(path.glob("chunk-*.parquet"))) if path.is_dir() else ()
        if not chunks:
            raise PilotNotFound(
                f"no FlyWire pilot chunks under {path} — run "
                f"`python scripts/bulk_synapses.py` once credentials allow"
            )
        offset = 0
        cursor = path / "cursor.json"
        if cursor.is_file():
            try:
                offset = int(json.loads(cursor.read_text(encoding="utf-8"))["offset"])
            except (OSError, ValueError, KeyError, TypeError):
                offset = 0
        return cls(root=path, chunks=chunks, cursor_offset=offset)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    def total_bytes(self) -> int:
        return sum(c.stat().st_size for c in self.chunks)

    def row_count(self) -> int:
        """Rows across all chunks, read from parquet metadata (no decode)."""
        import pyarrow.parquet as pq

        return sum(pq.read_metadata(c).num_rows for c in self.chunks)

    def load_edges(
        self,
        limit_chunks: int | None = None,
        columns: tuple[str, ...] = ("pre", "post"),
    ) -> list[tuple[int, int]]:
        """Every (pre, post) row, optionally limited to the first N chunks."""
        chunks = self.chunks[:limit_chunks] if limit_chunks else self.chunks
        return load_parquet_rows(chunks, columns=columns)

    def successors(self, limit_chunks: int | None = None) -> dict[int, set[int]]:
        """Adjacency as ``{pre_root_id: {post_root_id, ...}}``."""
        graph: dict[int, set[int]] = {}
        for pre, post in self.load_edges(limit_chunks=limit_chunks):
            graph.setdefault(pre, set()).add(post)
        return graph

    def describe(self) -> dict[str, Any]:
        """Measured summary: never a figure we did not read off the files."""
        return {
            "root": str(self.root),
            "chunks": self.chunk_count,
            "rows": self.row_count(),
            "bytes": self.total_bytes(),
            "cursor_offset": self.cursor_offset,
        }


def load_parquet_rows(
    chunks: tuple[Path, ...] | list[Path], columns: tuple[str, ...] = ("pre", "post")
) -> list[tuple[int, int]]:
    """Decode ``columns`` from parquet chunks into tuples of ints."""
    import pyarrow.parquet as pq

    rows: list[tuple[int, int]] = []
    for chunk in chunks:
        table = pq.read_table(chunk, columns=list(columns))
        decoded = [table.column(name).to_pylist() for name in columns]
        rows.extend(zip(*decoded))  # type: ignore[arg-type]
    return rows
