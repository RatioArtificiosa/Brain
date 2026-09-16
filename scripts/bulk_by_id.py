"""ID-paged bulk fetch of the real FlyWire corpus (PH4-WI02, the unblock).

WHY THIS EXISTS
---------------
The recorded plan said the full corpus needed "id-range paging" but that it
was "not yet implemented", and the standing assumption was that the live
download was blocked on account approval. Measurement changed that picture:

- `flywire_fafb_public` IS open and `synapses_nt_v1` IS readable.
- OFFSET paging is a dead end: latency grows 117x from offset 0 to 5M, which
  extrapolates to ~1774 hours (~74 days) for the full corpus.
- `filter_in_dict` on `pre_pt_root_id` returns a neuron's synapses in ~0.22s
  and is INDEPENDENT of any offset (probe_fetch_ceiling / probe_id_paging).

So the tractable path is: enumerate the neurons (from `proofread_neurons`,
which is readable), then fetch each neuron's outgoing synapses by ID. Measured
extrapolation for the full fly: ~8.6 hours, ~102M rows -- versus 74 days.

Design:
  * resumable: a cursor of completed neuron IDs, written atomically per batch
  * parallel-safe: each neuron is an independent request
  * zero silent loss: every failing neuron is retried, then recorded
  * parquet chunks + a manifest that records MEASURED counts

Usage:
  python scripts/bulk_by_id.py --list-neurons          # build the neuron list
  python scripts/bulk_by_id.py --max-neurons 500       # pilot
  python scripts/bulk_by_id.py                         # full run
  python scripts/bulk_by_id.py --manifest-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DATA_DIR = Path(r"G:\BRAIN\VNR\data\flywire_v783")
TABLE = "synapses_nt_v1"
NEURON_TABLE = "proofread_neurons"
NT_ORDER = ("gaba", "ach", "glut", "oct", "ser", "da")
UNKNOWN = 255
COLS = ["pre_pt_root_id", "post_pt_root_id", "valid_nt", *NT_ORDER]

NEURON_LIST = DATA_DIR / "neurons.json"
DONE_CURSOR = DATA_DIR / "id_cursor.json"
FAILED_LOG = DATA_DIR / "id_failures.json"


def _client():
    from vnr.connectome.credentials import load_token

    os.environ.setdefault("CAVE_TOKEN", load_token())
    from caveclient import CAVEclient

    return CAVEclient("flywire_fafb_public", auth_token=os.environ["CAVE_TOKEN"])


# --------------------------------------------------------------- neuron list


def list_neurons(force: bool = False) -> list[int]:
    """Enumerate every neuron root id reachable in the public datastack."""
    if NEURON_LIST.exists() and not force:
        ids = json.loads(NEURON_LIST.read_text(encoding="utf-8"))
        print(f"neuron list cached: {len(ids):,} ids")
        return [int(i) for i in ids]

    client = _client()
    ids: list[int] = []
    offset = 0
    page = 100_000
    start = time.perf_counter()
    while True:
        try:
            df = client.materialize.query_table(
                NEURON_TABLE,
                limit=page,
                offset=offset,
                select_columns=["pt_root_id"],
                split_positions=False,
            )
        except Exception as exc:  # noqa: BLE001 - report, then stop cleanly
            print(
                f"  neuron page at {offset} failed: {type(exc).__name__}: {str(exc)[:90]}"
            )
            break
        if len(df) == 0:
            break
        found = df["pt_root_id"].dropna().astype("int64").tolist()
        ids.extend(int(i) for i in found if int(i) != 0)
        offset += len(df)
        print(
            f"  neurons: {len(ids):,} ({time.perf_counter() - start:.0f}s)",
            flush=True,
        )
        # Range state is tiny; save as we go so a crash does not lose the list.
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        NEURON_LIST.write_text(json.dumps(ids), encoding="utf-8")
        if len(df) < page:
            break

    uniq = sorted(set(ids))
    NEURON_LIST.write_text(json.dumps(uniq), encoding="utf-8")
    print(f"neuron list: {len(uniq):,} unique root ids -> {NEURON_LIST}")
    return uniq


# ------------------------------------------------------------------- fetching


def _normalize(df):
    """(pre, post, nt, conf) with unmapped roots dropped, counts reported."""
    import pandas as pd

    if len(df) == 0:
        return None, 0
    probs = df[list(NT_ORDER)].to_numpy(dtype="float64", na_value=0.0)
    best = probs.argmax(axis=1).astype("uint8")
    conf = probs.max(axis=1).astype("float32")
    valid = df["valid_nt"].to_numpy(dtype=bool)
    best[~valid] = UNKNOWN
    pre = df["pre_pt_root_id"].to_numpy(dtype="int64")
    post = df["post_pt_root_id"].to_numpy(dtype="int64")
    keep = (pre != 0) & (post != 0)
    dropped = int((~keep).sum())
    frame = pd.DataFrame(
        {"pre": pre[keep], "post": post[keep], "nt": best[keep], "conf": conf[keep]}
    )
    return frame, dropped


def fetch_neuron(client, neuron_id: int, retries: int = 6):
    """Every outgoing synapse for one neuron. Returns (frame, dropped).

    Rate limits are handled explicitly: the CAVE server answers 429 with a
    Retry-After hint when a client is too eager (observed at 12 workers).
    We honour that hint with exponential fallback rather than hammering, and
    give up only after `retries` attempts so one bad neuron cannot stall the
    run.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            df = client.materialize.query_table(
                TABLE,
                filter_in_dict={"pre_pt_root_id": [int(neuron_id)]},
                limit=100_000,
                select_columns=COLS,
                split_positions=False,
            )
            return _normalize(df)
        except Exception as exc:  # noqa: BLE001 - transient server errors
            last = exc
            text = str(exc)
            if "429" in text or "Too Many Requests" in text:
                # Back off hard: this is the server asking us to slow down.
                wait = _retry_after(text) or (5.0 * (attempt + 1))
            else:
                wait = 2.0 * (attempt + 1)
            time.sleep(wait)
    raise OSError(f"neuron {neuron_id} failed {retries}x: {last}") from last


def _retry_after(text: str) -> float | None:
    """Parse a Retry-After hint out of an error message, if present."""
    import re

    match = re.search(r"Retry-After['\"]?\s*[:=]\s*['\"]?(\d+)", text, re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r"Too Many Requests:\s*(\d+)", text)
    if match:
        return float(match.group(1))
    return None


def run(max_neurons: int | None, workers: int, chunk_size: int) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    ids = list_neurons()
    if max_neurons:
        ids = ids[:max_neurons]

    done: set[int] = set()
    if DONE_CURSOR.exists():
        try:
            done = {int(i) for i in json.loads(DONE_CURSOR.read_text(encoding="utf-8"))}
        except (OSError, ValueError):
            done = set()
    todo = [i for i in ids if i not in done]
    print(
        f"neurons: {len(ids):,} total, {len(done):,} already fetched, {len(todo):,} to go"
    )

    if not todo:
        print("nothing to fetch")
        return finalize()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = _client()
    buffer: list = []
    total_rows = 0
    total_dropped = 0
    failures: list[int] = []
    completed = list(done)
    start = time.perf_counter()
    chunk_index = len(sorted(DATA_DIR.glob("byid-*.parquet")))

    def flush() -> None:
        nonlocal buffer, chunk_index, total_rows
        if not buffer:
            return
        import pandas as pd

        merged = pd.concat(buffer, ignore_index=True)
        path = DATA_DIR / f"byid-{chunk_index:05d}.parquet"
        pq.write_table(pa.Table.from_pandas(merged, preserve_index=False), path)
        total_rows += len(merged)
        chunk_index += 1
        buffer = []
        DONE_CURSOR.write_text(json.dumps(sorted(set(completed))), encoding="utf-8")
        rate = total_rows / max(time.perf_counter() - start, 1e-9)
        print(
            f"  chunk {chunk_index:>4} rows={total_rows:,} "
            f"neurons={len(completed):,} {rate:,.0f} rows/s",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_neuron, client, nid): nid for nid in todo}
        for i, future in enumerate(as_completed(futures), 1):
            nid = futures[future]
            try:
                frame, dropped = future.result()
                total_dropped += dropped
                if frame is not None:
                    buffer.append(frame)
                completed.append(nid)
            except Exception as exc:  # noqa: BLE001 - record and continue
                failures.append(nid)
                print(f"  FAILED neuron {nid}: {str(exc)[:80]}", flush=True)
            if len(buffer) >= chunk_size:
                flush()
            if i % 200 == 0:
                DONE_CURSOR.write_text(
                    json.dumps(sorted(set(completed))), encoding="utf-8"
                )

    flush()
    DONE_CURSOR.write_text(json.dumps(sorted(set(completed))), encoding="utf-8")
    if failures:
        FAILED_LOG.write_text(json.dumps(failures), encoding="utf-8")
    elapsed = time.perf_counter() - start
    print(
        f"\nfetched {total_rows:,} rows from {len(completed):,} neurons in "
        f"{elapsed / 3600:.2f} h ({elapsed / 60:.1f} min)"
    )
    print(f"zero-root rows dropped: {total_dropped:,}")
    if failures:
        print(f"neurons needing retry: {len(failures):,} -> {FAILED_LOG}")
    return finalize()


def finalize() -> int:
    """Write the manifest with MEASURED counts (spec 13)."""
    import pyarrow.parquet as pq

    from vnr.connectome.flywire import Manifest, sha256_file, write_manifest

    chunks = sorted(DATA_DIR.glob("byid-*.parquet"))
    if not chunks:
        print("no byid chunks to manifest")
        return 1
    rows = sum(pq.read_metadata(c).num_rows for c in chunks)
    neurons = 0
    if NEURON_LIST.exists():
        neurons = len(json.loads(NEURON_LIST.read_text(encoding="utf-8")))
    manifest = Manifest(
        release="v783-public-byid",
        source=f"CAVE flywire_fafb_public/{TABLE} via id-paging",
        neurons=neurons,
        synapses=rows,
        edge_rows=rows,
        filters=[
            "valid_nt false -> nt UNKNOWN(255), preserved",
            "pre==0 or post==0 (unmapped segments) dropped",
            "fetched by pre_pt_root_id (OFFSET-free), resumable",
        ],
        sha256=",".join(f"{c.name}:{sha256_file(c)[:16]}" for c in chunks)[:4000],
    )
    write_manifest(DATA_DIR / "manifest_byid.json", manifest)
    print(f"manifest: {len(chunks)} chunks, {rows:,} rows, {neurons:,} neurons listed")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-neurons", action="store_true")
    parser.add_argument("--force-list", action="store_true")
    parser.add_argument("--max-neurons", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent requests. 12 provoked HTTP 429 rate limiting; 4 is "
        "the measured-safe default.",
    )
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args(argv)

    if args.list_neurons:
        list_neurons(force=args.force_list)
        return 0
    if args.manifest_only:
        return finalize()
    return run(args.max_neurons, args.workers, args.chunk_size)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
