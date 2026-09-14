"""Throughput probe: 100K rows timed (ops script, read-only)."""

import os
import time

from vnr.connectome.credentials import load_token

os.environ["CAVE_TOKEN"] = load_token()

from caveclient import CAVEclient

client = CAVEclient("flywire_fafb_public", auth_token=os.environ["CAVE_TOKEN"])
start = time.perf_counter()
df = client.materialize.query_table(
    "synapses_nt_v1",
    limit=100_000,
    select_columns=[
        "pre_pt_root_id",
        "post_pt_root_id",
        "gaba",
        "ach",
        "glut",
        "oct",
        "ser",
        "da",
    ],
)
elapsed = time.perf_counter() - start
print(f"rows={len(df)} secs={elapsed:.1f} rows_per_sec={len(df) / elapsed:,.0f}")
full = 244_358_226
print(f"full-table estimate: {full / (len(df) / elapsed) / 3600:.1f} hours")
print(
    f"approx bytes: {df.memory_usage(deep=True).sum() * (full / len(df)) / 1e9:.1f} GB"
)
