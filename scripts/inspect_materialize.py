"""Inspect materialize surface on the public datastack (ops script)."""

import os

from vnr.connectome.credentials import load_token

os.environ["CAVE_TOKEN"] = load_token()

from caveclient import CAVEclient

client = CAVEclient("flywire_fafb_public", auth_token=os.environ["CAVE_TOKEN"])
names = [m for m in dir(client.materialize) if not m.startswith("_")]
print("materialize surface:")
for name in sorted(names)[:40]:
    print("  ", name)
