"""Measure the REAL fetch ceiling on flywire_fafb_public.

The recorded lesson (notes entry 33) is that OFFSET paging decays with depth:
50K-row pages were "stable at ~5-12K rows/s DECAYING with OFFSET", and 500K
pages returned server 500/503. That makes a naive full bulk = 100h+.

This script answers three questions with measurements, not assumptions:

1. How many rows does synapses_nt_v1 ACTUALLY contain (count, not estimate)?
2. How does page latency scale with OFFSET - is it really linear decay?
3. Is there an id-range / filter path that avoids OFFSET entirely
   (which would make the full corpus tractable)?

    python scripts/probe_fetch_ceiling.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

TABLE = "synapses_nt_v1"
COLS = ["pre_pt_root_id", "post_pt_root_id"]


def main() -> int:
    from vnr.connectome.credentials import load_token

    token = load_token()
    os.environ.setdefault("CAVE_TOKEN", token)
    from caveclient import CAVEclient

    client = CAVEclient("flywire_fafb_public", auth_token=token)

    # --- 1. Real row count -------------------------------------------------
    print("=== row count ===")
    try:
        n = client.materialize.get_table_metadata(TABLE)
        print("metadata:", {k: str(v)[:60] for k, v in list(n.items())[:6]})
        for key in ("num_rows", "row_count", "count"):
            if key in n:
                print(f"  {key} = {n[key]:,}")
    except Exception as exc:  # noqa: BLE001 - report, do not crash
        print(f"  metadata unavailable: {type(exc).__name__}: {str(exc)[:120]}")

    # --- 2. OFFSET latency curve ------------------------------------------
    print("\n=== OFFSET latency curve (5000-row pages) ===")
    print(f"{'offset':>12}{'rows':>8}{'seconds':>10}{'rows/s':>10}")
    offsets = [0, 100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
    latencies: list[tuple[int, float]] = []
    for offset in offsets:
        start = time.perf_counter()
        try:
            df = client.materialize.query_table(
                TABLE,
                limit=5000,
                offset=offset,
                select_columns=COLS,
                split_positions=False,
            )
            elapsed = time.perf_counter() - start
            rows = len(df)
            rate = rows / elapsed if elapsed else 0
            latencies.append((offset, elapsed))
            print(f"{offset:>12,}{rows:>8,}{elapsed:>10.2f}{rate:>10,.0f}")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            print(f"{offset:>12,}{'ERR':>8}{elapsed:>10.2f}  {type(exc).__name__}")
            break

    # --- 3. Filter / id-range path (avoids OFFSET entirely) ---------------
    print("\n=== id-range filter path ===")
    # A bounded pre_pt_root_id range: if this works, the full corpus becomes
    # a sequence of range queries instead of a deepening OFFSET scan.
    try:
        lo = 720575940600000000
        start = time.perf_counter()
        df = client.materialize.query_table(
            TABLE,
            filter_equal_dict={"pre_pt_root_id": lo},
            limit=1000,
            select_columns=COLS,
            split_positions=False,
        )
        elapsed = time.perf_counter() - start
        print(f"  filter_equal_dict(pre=...): {len(df)} rows in {elapsed:.2f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"  filter_equal_dict failed: {type(exc).__name__}: {str(exc)[:130]}")

    try:
        start = time.perf_counter()
        df = client.materialize.query_table(
            TABLE,
            filter_in_dict={"pre_pt_root_id": [720575940630479697]},
            limit=2000,
            select_columns=COLS,
            split_positions=False,
        )
        elapsed = time.perf_counter() - start
        print(f"  filter_in_dict(pre=[one id]): {len(df)} rows in {elapsed:.2f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"  filter_in_dict failed: {type(exc).__name__}: {str(exc)[:130]}")

    # --- 4. Extrapolation --------------------------------------------------
    if len(latencies) >= 2:
        print("\n=== extrapolation ===")
        first = latencies[0][1]
        last = latencies[-1][1]
        print(f"  offset {latencies[0][0]:,}: {first:.2f}s/page")
        print(f"  offset {latencies[-1][0]:,}: {last:.2f}s/page")
        if last > first:
            growth = last / first
            print(f"  latency grew {growth:.2f}x across that span")
        med = statistics.median(t for _, t in latencies)
        for target, label in [
            (244_358_226, "full 244M corpus"),
            (10_000_000, "10M rows"),
        ]:
            pages = target / 5000
            hours = pages * med / 3600
            print(
                f"  {label}: {pages:,.0f} pages x {med:.2f}s = {hours:.1f} h (at median rate)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
