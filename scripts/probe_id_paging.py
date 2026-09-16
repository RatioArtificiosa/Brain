"""Design an OFFSET-free fetch strategy by paging over the ID space.

Measured facts that motivate this (scripts/probe_fetch_ceiling.py):
  - OFFSET paging degrades 117x from offset 0 to 5M; a full corpus via OFFSET
    would take ~1774 hours (~74 days). That path is dead.
  - filter_in_dict on pre_pt_root_id returned 346 rows in 0.37s, i.e. fast
    and independent of any offset.

So: if we can enumerate the neuron root IDs that exist, we can fetch each
neuron's synapses directly and never touch OFFSET. This script measures:
  1. how many neurons exist (from proofread_neurons, which IS readable)
  2. the distribution of their root IDs (sparse 64-bit space)
  3. real per-neuron fetch latency, to extrapolate the full-corpus cost

    python scripts/probe_id_paging.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TABLE = "synapses_nt_v1"
NEURONS = "proofread_neurons"
COLS = ["pre_pt_root_id", "post_pt_root_id"]


def main() -> int:
    from vnr.connectome.credentials import load_token

    token = load_token()
    os.environ.setdefault("CAVE_TOKEN", token)
    from caveclient import CAVEclient

    client = CAVEclient("flywire_fafb_public", auth_token=token)

    # --- 1. Neuron inventory ----------------------------------------------
    print("=== neuron inventory (proofread_neurons) ===")
    start = time.perf_counter()
    try:
        nmeta = client.materialize.get_table_metadata(NEURONS)
        print("metadata keys:", sorted(nmeta.keys())[:12])
        for key in ("num_rows", "row_count", "count", "table_name"):
            if key in nmeta:
                print(f"  {key} = {nmeta[key]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  metadata failed: {type(exc).__name__}: {str(exc)[:100]}")

    # Pull a page of the neuron table to learn the id column name + values.
    ids: list[int] = []
    try:
        df = client.materialize.query_table(
            NEURONS, limit=100_000, select_columns=None, split_positions=False
        )
        print(
            f"  fetched {len(df):,} neuron rows in {time.perf_counter() - start:.1f}s"
        )
        print(f"  columns: {list(df.columns)}")
        for candidate in ("pt_root_id", "root_id", "id", "pre_pt_root_id"):
            if candidate in df.columns:
                vals = df[candidate].dropna().astype("int64").tolist()
                ids = vals
                print(f"  using id column: {candidate}")
                break
    except Exception as exc:  # noqa: BLE001
        print(f"  neuron fetch failed: {type(exc).__name__}: {str(exc)[:120]}")

    if not ids:
        print("\ncannot enumerate neuron ids; id-paging design blocked")
        return 1

    ids.sort()
    print(f"\n=== id space ({len(ids):,} sampled neurons) ===")
    print(f"  min  = {ids[0]:,}")
    print(f"  max  = {ids[-1]:,}")
    print(f"  span = {ids[-1] - ids[0]:,}")
    if len(ids) > 1:
        import itertools

        gaps = [b - a for a, b in itertools.pairwise(ids)]
        print(f"  median gap = {statistics.median(gaps):,.0f}")
        print(f"  mean gap   = {statistics.mean(gaps):,.0f}")
        print(f"  density    = {len(ids) / (ids[-1] - ids[0]):.2e} ids per unit span")
        print("  -> the id space is SPARSE: a dense range scan would waste most calls")

    # --- 2. Real per-neuron fetch latency ---------------------------------
    print("\n=== per-neuron synapse fetch latency ===")
    sample = ids[:20] if len(ids) >= 20 else ids
    times: list[float] = []
    total_rows = 0
    for nid in sample:
        start = time.perf_counter()
        try:
            df = client.materialize.query_table(
                TABLE,
                filter_in_dict={"pre_pt_root_id": [int(nid)]},
                limit=10_000,
                select_columns=COLS,
                split_positions=False,
            )
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            total_rows += len(df)
        except Exception as exc:  # noqa: BLE001
            print(f"  neuron {nid}: {type(exc).__name__}")
    if times:
        med = statistics.median(times)
        avg_rows = total_rows / len(times)
        print(f"  neurons probed     : {len(times)}")
        print(f"  median latency     : {med:.3f}s")
        print(f"  mean rows/neuron   : {avg_rows:.1f}")
        print(f"  implied throughput : {avg_rows / med:,.0f} rows/s")
        print()
        print("  Extrapolation to the full population:")
        for n_neurons, label in [(139_255, "full fly (139,255 neurons)")]:
            hours = n_neurons * med / 3600
            rows = n_neurons * avg_rows
            print(
                f"    {label}: {n_neurons:,} calls x {med:.3f}s = {hours:.1f} h, "
                f"~{rows / 1e6:.1f}M rows"
            )
        print()
        print("  Compare: OFFSET path measured ~1774 h. ID-paging is the way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
