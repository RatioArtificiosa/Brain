"""FlyWire ingestion machinery (plan PH4-WI02, spec §12–13).

Covers everything up to the credential boundary: chunked resumable fetching
(file:// and http(s):// chunk sources with persisted cursors), the normalized
edge schema + validation, and version manifests that record dataset version,
filters, and the synapse-count figure in use (resolving the 50M-vs-54.5M
ambiguity per-manifest, never by assumption).

NOT covered here (needs the owner's FlyWire account): the live download
itself. Re-entry: owner runs ``vnr dataset auth`` (PH4-WI02b CLI, next),
then the same cursor machinery drives the real fetch. Nothing below changes.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "ChunkCursor",
    "ChunkSource",
    "EdgeRow",
    "FileChunkSource",
    "FlywireRelease",
    "HttpChunkSource",
    "Manifest",
    "normalize_edge",
    "read_edges",
    "write_manifest",
]

EDGE_COLUMNS = ("pre_id", "post_id", "synapses", "neurotransmitter")

KNOWN_NTS = frozenset(
    {"GABA", "ACH", "GLUT", "SER", "OCT", "TYR", "DOP", "HIST", "UNKNOWN"}
)


@dataclass(frozen=True)
class FlywireRelease:
    """Pinned dataset version. Counts recorded per-manifest (see §13)."""

    name: str = "v783"
    neurons: int = 139_255
    synapses: int = 50_000_000
    synapse_note: str = "rounded figure; exact count recorded in manifest at ingest"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("release name required")
        if self.neurons <= 0 or self.synapses <= 0:
            raise ValueError("counts must be positive")


@dataclass(frozen=True)
class EdgeRow:
    """One normalized directed edge: pre -> post with synapse count + NT."""

    pre_id: int
    post_id: int
    synapses: int
    neurotransmitter: str = "UNKNOWN"


def normalize_edge(raw: dict[str, str]) -> EdgeRow:
    """Validate + coerce one raw row. Raises ValueError listing the problem."""
    try:
        pre = int(raw["pre_id"])
        post = int(raw["post_id"])
        syn = int(raw["synapses"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"bad ids/counts in row: {raw!r}")
    if pre < 0 or post < 0:
        raise ValueError(f"ids must be non-negative: {raw!r}")
    if syn <= 0:
        raise ValueError(f"synapse count must be positive: {raw!r}")
    nt = (raw.get("neurotransmitter") or "UNKNOWN").strip().upper()
    if nt not in KNOWN_NTS:
        raise ValueError(f"unknown neurotransmitter {nt!r} in row: {raw!r}")
    return EdgeRow(pre, post, syn, nt)


@dataclass
class ChunkCursor:
    """Resumable position: path + byte offset, persisted as JSON."""

    path: Path
    offset: int = 0

    def save(self) -> None:
        self.path.write_text(json.dumps({"offset": self.offset}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ChunkCursor:
        try:
            offset = int(json.loads(path.read_text(encoding="utf-8"))["offset"])
        except (OSError, ValueError, KeyError, TypeError):
            offset = 0
        return cls(path=path, offset=max(0, offset))


class ChunkSource:
    """Byte-chunk iterator with resume. Subclasses define transport."""

    chunk_bytes: int = 1 << 20

    def read_chunks(self, cursor: ChunkCursor) -> Iterator[bytes]:
        raise NotImplementedError


class FileChunkSource(ChunkSource):
    """Local file sliced into chunks (tests + file:// datasets)."""

    def __init__(self, data_path: Path, chunk_bytes: int = 1 << 20) -> None:
        self.data_path = Path(data_path)
        self.chunk_bytes = chunk_bytes

    def read_chunks(self, cursor: ChunkCursor) -> Iterator[bytes]:
        with self.data_path.open("rb") as fh:
            fh.seek(cursor.offset)
            while True:
                chunk = fh.read(self.chunk_bytes)
                if not chunk:
                    return
                cursor.offset += len(chunk)
                yield chunk


class HttpChunkSource(ChunkSource):
    """HTTP(S) with Range resume. Tested against local servers (no internet)."""

    def __init__(
        self, url: str, chunk_bytes: int = 1 << 20, timeout_sec: float = 60.0
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("url must be http(s)")
        self.url = url
        self.chunk_bytes = chunk_bytes
        self.timeout_sec = timeout_sec

    def read_chunks(self, cursor: ChunkCursor) -> Iterator[bytes]:
        offset = cursor.offset
        while True:
            req = urllib.request.Request(
                self.url, headers={"Range": f"bytes={offset}-"}
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    if resp.status not in (200, 206):
                        raise OSError(f"unexpected HTTP {resp.status}")
                    while True:
                        chunk = resp.read(self.chunk_bytes)
                        if not chunk:
                            return
                        offset += len(chunk)
                        cursor.offset = offset
                        yield chunk
                    return
            except OSError:
                raise
            except Exception as exc:
                raise OSError(f"fetch failed at offset {offset}: {exc}") from exc


def read_edges(chunks: Iterator[bytes]) -> Iterator[EdgeRow]:
    """Decode CSV chunks (header + rows) into validated EdgeRows.

    Chunk boundaries can split rows: the decoder buffers the tail.
    """
    text = ""
    header: list[str] | None = None
    for chunk in chunks:
        text += chunk.decode("utf-8", errors="strict")
        *lines, text = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = [h.strip() for h in line.split(",")]
                missing = [
                    c for c in ("pre_id", "post_id", "synapses") if c not in header
                ]
                if missing:
                    raise ValueError(f"missing columns: {missing}")
                continue
            values = [v.strip() for v in line.split(",")]
            yield normalize_edge(dict(zip(header, values)))
    if text.strip():
        if header is None:
            raise ValueError("no header row found")
        values = [v.strip() for v in text.strip().split(",")]
        yield normalize_edge(dict(zip(header, values)))


@dataclass
class Manifest:
    """Ingest record: what, from where, filtered how, hashed (spec §13)."""

    release: str
    source: str
    neurons: int
    synapses: int
    edge_rows: int
    filters: list[str] = field(default_factory=list)
    sha256: str = ""
    config_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "release": self.release,
            "source": self.source,
            "neurons": self.neurons,
            "synapses": self.synapses,
            "edge_rows": self.edge_rows,
            "filters": self.filters,
            "sha256": self.sha256,
            "config_hash": self.config_hash,
        }


def write_manifest(path: Path, manifest: Manifest) -> None:
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
