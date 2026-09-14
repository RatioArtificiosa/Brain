"""Triad census of the pilot corpus (ops script; exact, timed)."""

import time
from pathlib import Path

import pyarrow.parquet as pq

from vnr.connectome.motifs import census

DATA = Path(r"G:\BRAIN\VNR\data\flywire_v783")


def main() -> None:
    succ: dict[int, set[int]] = {}
    rows = 0
    for chunk in sorted(DATA.glob("chunk-*.parquet")):
        table = pq.read_table(chunk, columns=["pre", "post"])
        pre = table.column("pre").to_pylist()
        post = table.column("post").to_pylist()
        for s, t in zip(pre, post):
            succ.setdefault(s, set()).add(t)
        rows += len(pre)
    print(f"graph: {len(succ):,} sources, {rows:,} edges", flush=True)
    start = time.perf_counter()
    cat = census(succ)
    elapsed = time.perf_counter() - start
    print(
        f"census: {cat.triples_counted:,} triples in {elapsed:.1f}s (exact={cat.exact})"
    )
    print("top motifs by count:")
    for m in cat.top(15):
        print(
            f"  M{m.code:02d} edges={m.n_edges} mutual={m.n_mutual} "
            f"cyclic={int(m.cyclic)} count={m.count:,} conc={m.concentration:.4f} "
            f"roles={m.roles}"
        )


if __name__ == "__main__":
    main()
