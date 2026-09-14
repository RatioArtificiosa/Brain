"""Signature + sample rows (ops script, read-only)."""

import inspect
import os

from vnr.connectome.credentials import load_token

os.environ["CAVE_TOKEN"] = load_token()

from caveclient import CAVEclient

client = CAVEclient("flywire_fafb_public", auth_token=os.environ["CAVE_TOKEN"])
print(inspect.signature(client.materialize.query_table))
df = client.materialize.query_table("synapses_nt_v1", limit=3)
print("rows:", len(df))
print("columns:", list(df.columns)[:20])
print(df.head(2).to_string())
