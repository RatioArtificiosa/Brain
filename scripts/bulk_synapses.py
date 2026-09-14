"""Bulk FlyWire synapse fetch (ops script, resumable, committable).

Pages ``synapses_nt_v1`` (244M contacts) via offset/limit, normalizes each
page to (pre_root, post_root, nt_code, confidence), and writes snappy parquet
chunks. Cursor (next offset) persists after every page: kill and rerun any
time, nothing re-downloads. Manifest finalized with --manifest-only.

NT codes: 0 GABA, 1 ACH, 2 GLUT, 3 OCT, 4 SER, 5 DOP, 255 UNKNOWN (valid_nt
false). Confidence = winning probability (float32).

Usage:
  python scripts/bulk_synapses.py --max-pages 2 --page-rows 500000   # pilot 1M
  python scripts/bulk_synapses.py                                    # full run
  python scripts/bulk_synapses.py --manifest-only                    # finalize
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

NT_ORDER = ("gaba", "ach", "glut", "oct", "ser", "da")
NT_NAMES = ("GABA", "ACH", "GLUT", "OCT", "SER", "DOP")
UNKNOWN = 255

DATA_DIR = Path(r"G:\BRAIN\VNR\data\flywire_v783")
TABLE = "synapses_nt_v1"


def _client():
    from vnr.connectome.credentials import load_token

    os.environ.setdefault("CAVE_TOKEN", load_token())
    from caveclient import CAVEclient

    return CAVEclient("flywire_fafb_public", auth_token=os.environ["CAVE_TOKEN"])


def _cursor_path() -> Path:
    return DATA_DIR / "cursor.json"


def _load_offset() -> int:
    try:
        return int(json.loads(_cursor_path().read_text(encoding="utf-8"))["offset"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def _save_offset(offset: int) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _cursor_path().write_text(json.dumps({"offset": offset}), encoding="utf-8")


def _normalize(df):
    import pandas as pd

    probs = df[list(NT_ORDER)].to_numpy(dtype="float64", na_value=0.0)
    best = probs.argmax(axis=1).astype("uint8")
    conf = probs.max(axis=1).astype("float32")
    valid = df["valid_nt"].to_numpy(dtype=bool)
    best[~valid] = UNKNOWN
    pre = df["pre_pt_root_id"].to_numpy(dtype="int64")
    post = df["post_pt_root_id"].to_numpy(dtype="int64")
    keep = (pre != 0) & (post != 0)  # root 0 = unmapped segment, not a neuron
    dropped = int((~keep).sum())
    frame = pd.DataFrame(
        {"pre": pre[keep], "post": post[keep], "nt": best[keep], "conf": conf[keep]}
    )
    return frame, dropped


def fetch_page(client, offset: int, page_rows: int, retries: int = 6):
    cols = ["pre_pt_root_id", "post_pt_root_id", "valid_nt", *NT_ORDER]
    last: Exception | None = None
    for attempt in range(retries):
        try:
            # split_positions=False: simpler server-side plan (the 500
            # 'ipc_compress' blowup rode on the default split path).
            return client.materialize.query_table(
                TABLE,
                limit=page_rows,
                offset=offset,
                select_columns=cols,
                split_positions=False,
            )
        except Exception as exc:  # noqa: BLE001 — transient 503/500s; backoff + retry
            last = exc
            time.sleep(5.0 * (attempt + 1))
    raise OSError(f"page at offset {offset} failed {retries}x: {last}") from last


def run(max_pages: int, page_rows: int) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    client = _client()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    offset = _load_offset()
    pages = 0
    total_rows = 0
    total_dropped = 0
    start = time.perf_counter()
    while True:
        if max_pages and pages >= max_pages:
            break
        df = fetch_page(client, offset, page_rows)
        if len(df) == 0:
            break
        out, dropped = _normalize(df)
        total_dropped += dropped
        chunk = DATA_DIR / f"chunk-{offset:012d}.parquet"
        pq.write_table(pa.Table.from_pandas(out, preserve_index=False), chunk)
        offset += len(df)
        total_rows += len(out)
        pages += 1
        _save_offset(offset)
        rate = total_rows / max(time.perf_counter() - start, 1e-9)
        print(
            f"page {pages}: offset={offset:,} rows={total_rows:,} "
            f"dropped={total_dropped:,} {rate:,.0f}/s",
            flush=True,
        )
    print(
        f"DONE pages={pages} rows={total_rows} dropped={total_dropped} next_offset={offset}"
    )
    return offset


def finalize() -> None:
    import pyarrow.parquet as pq

    from vnr.connectome.flywire import Manifest, sha256_file, write_manifest

    chunks = sorted(DATA_DIR.glob("chunk-*.parquet"))
    rows = sum(pq.read_metadata(c).num_rows for c in chunks)
    manifest = Manifest(
        release="v783-public",
        source=f"CAVE flywire_fafb_public/{TABLE}",
        neurons=139_255,
        synapses=rows,
        edge_rows=rows,
        filters=[
            "valid_nt false -> nt UNKNOWN(255), preserved",
            "pre==0 or post==0 (unmapped segments) dropped",
        ],
        sha256=",".join(f"{c.name}:{sha256_file(c)[:16]}" for c in chunks)[:4000],
    )
    write_manifest(DATA_DIR / "manifest.json", manifest)
    print(f"manifest: {len(chunks)} chunks, {rows:,} rows")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--page-rows", type=int, default=100_000)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)
    if args.manifest_only:
        finalize()
        return 0
    run(args.max_pages, args.page_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
