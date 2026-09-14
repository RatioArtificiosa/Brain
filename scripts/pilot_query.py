"""Pilot: metadata + first rows of synapses_nt_v1 (ops script, read-only)."""

import json
import os

from vnr.connectome.credentials import load_token

os.environ["CAVE_TOKEN"] = load_token()

from caveclient import CAVEclient

client = CAVEclient("flywire_fafb_public", auth_token=os.environ["CAVE_TOKEN"])
meta = client.materialize.get_table_metadata("synapses_nt_v1", version=783)
print("meta keys:", sorted(meta.keys())[:15])
print("meta:", json.dumps(meta, default=str)[:800])
df = client.materialize.query_table(
    "synapses_nt_v1", limit=5, version=783, select_columns=None
)
print("rows:", len(df))
print("dtypes:", dict(df.dtypes.astype(str)))
print(df.head(3).to_string())
