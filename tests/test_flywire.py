"""PH4-WI02a tests: chunked resume, schema validation, manifests (no network)."""

import functools
import http.server
import json
import threading

import pytest

from vnr.connectome.flywire import (
    ChunkCursor,
    EdgeRow,
    FileChunkSource,
    FlywireRelease,
    HttpChunkSource,
    Manifest,
    normalize_edge,
    read_edges,
    sha256_file,
    write_manifest,
)

CSV = (
    "pre_id,post_id,synapses,neurotransmitter\n"
    "0,1,5,GABA\n1,2,3,ach\n2,0,7,GLUT\n3,4,1,\n"
)


def _write_csv(path, content: str = CSV):
    path.write_text(content, encoding="utf-8")
    return path


def test_normalize_edge_coerces_and_validates():
    row = normalize_edge(
        {"pre_id": "0", "post_id": "1", "synapses": "5", "neurotransmitter": "ach"}
    )
    assert row == EdgeRow(0, 1, 5, "ACH")
    assert (
        normalize_edge(
            {"pre_id": "3", "post_id": "4", "synapses": "1"}
        ).neurotransmitter
        == "UNKNOWN"
    )
    for bad in (
        {"pre_id": "-1", "post_id": "0", "synapses": "2"},
        {"pre_id": "0", "post_id": "0", "synapses": "0"},
        {"pre_id": "0", "post_id": "1", "synapses": "2", "neurotransmitter": "XYZ"},
        {"pre_id": "x", "post_id": "1", "synapses": "2"},
        {"pre_id": "0", "post_id": "1"},
    ):
        with pytest.raises(ValueError):
            normalize_edge(bad)


def test_release_and_manifest_roundtrip(tmp_path):
    rel = FlywireRelease()
    assert rel.neurons == 139_255
    man = Manifest(
        release=rel.name,
        source="test",
        neurons=100,
        synapses=200,
        edge_rows=150,
        filters=["nt != UNKNOWN"],
        sha256="abc",
        config_hash="def",
    )
    write_manifest(tmp_path / "m.json", man)
    loaded = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert loaded["edge_rows"] == 150 and loaded["filters"] == ["nt != UNKNOWN"]
    with pytest.raises(ValueError, match="release name required"):
        FlywireRelease(name="")


def test_file_chunks_resume_mid_stream(tmp_path):
    data = _write_csv(tmp_path / "edges.csv")
    cursor = ChunkCursor(path=tmp_path / "cursor.json")
    src = FileChunkSource(data, chunk_bytes=37)  # awkward size splits rows
    first = next(src.read_chunks(cursor))
    assert cursor.offset == len(first)
    cursor.save()
    resumed = ChunkCursor.load(tmp_path / "cursor.json")
    rest = b"".join(src.read_chunks(resumed))
    assert first + rest == data.read_bytes()
    loaded = ChunkCursor.load(tmp_path / "missing.json")
    assert loaded.offset == 0


def test_read_edges_across_split_rows(tmp_path):
    data = _write_csv(tmp_path / "edges.csv")
    src = FileChunkSource(data, chunk_bytes=29)
    rows = list(read_edges(src.read_chunks(ChunkCursor(path=tmp_path / "c.json"))))
    assert [(r.pre_id, r.post_id, r.synapses, r.neurotransmitter) for r in rows] == [
        (0, 1, 5, "GABA"),
        (1, 2, 3, "ACH"),
        (2, 0, 7, "GLUT"),
        (3, 4, 1, "UNKNOWN"),
    ]


def test_read_edges_rejects_bad_header(tmp_path):
    data = _write_csv(tmp_path / "bad.csv", "a,b,c\n1,2,3\n")
    src = FileChunkSource(data)
    with pytest.raises(ValueError, match="missing columns"):
        list(read_edges(src.read_chunks(ChunkCursor(path=tmp_path / "c.json"))))


def _serve(directory, port_holder):
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port_holder.append(server.server_address[1])
    server.serve_forever()


def test_http_source_against_local_server(tmp_path):
    _write_csv(tmp_path / "edges.csv")
    ports: list[int] = []
    thread = threading.Thread(target=_serve, args=(tmp_path, ports), daemon=True)
    thread.start()
    for _ in range(100):
        if ports:
            break
        import time

        time.sleep(0.05)
    assert ports, "local test server did not start"
    src = HttpChunkSource(f"http://127.0.0.1:{ports[0]}/edges.csv", chunk_bytes=41)
    rows = list(read_edges(src.read_chunks(ChunkCursor(path=tmp_path / "c.json"))))
    assert len(rows) == 4 and rows[0].pre_id == 0
    with pytest.raises(ValueError, match="http"):
        HttpChunkSource("file:///x")


def test_sha256_file(tmp_path):
    data = _write_csv(tmp_path / "edges.csv")
    assert len(sha256_file(data)) == 64
