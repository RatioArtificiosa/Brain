"""Validate the 1M pilot corpus (ops script, local-only)."""

from pathlib import Path

import pyarrow.parquet as pq

NT_NAMES = ("GABA", "ACH", "GLUT", "OCT", "SER", "DOP")

data = Path(r"G:\BRAIN\VNR\data\flywire_v783")
chunks = sorted(data.glob("chunk-*.parquet"))
print("chunks:", len(chunks))
tables = [pq.read_table(c) for c in chunks]
rows = sum(t.num_rows for t in tables)
print("rows:", f"{rows:,}")
import pyarrow as pa

full = pa.concat_tables(tables)


pre = full.column("pre").to_pylist()
post = full.column("post").to_pylist()
nt = full.column("nt").to_pylist()
conf = full.column("conf").to_pylist()
print("unique pre:", f"{len(set(pre)):,}", " unique post:", f"{len(set(post)):,}")
print("unique neurons:", f"{len(set(pre) | set(post)):,}")
from collections import Counter

dist = Counter(int(v) for v in nt)
for code, count in sorted(dist.items()):
    name = NT_NAMES[code] if code < 6 else "UNKNOWN"
    print(f"  nt {name}: {count:,} ({count / rows:.1%})")
print(f"conf mean={sum(conf) / len(conf):.3f} min={min(conf):.3f}")
print("id range pre:", min(pre), "-", max(pre))
