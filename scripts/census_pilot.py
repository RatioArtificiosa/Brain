"""Triad census of the pilot corpus (ops script; exact, timed).

Uses census_fast (structural dedupe, table-lookup codes). Measured on the
full 1M-edge pilot, 2026-09-14: 68.7 s vs 1888.1 s for the reference census
(27.5x), with identical per-code counts. Pass --reference to run the slow
oracle instead (useful for a one-off equivalence check).
"""

import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq

from vnr.connectome.motifs import census, census_fast

DATA = Path(r"G:\BRAIN\VNR\data\flywire_v783")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        action="store_true",
        help="run the slow reference oracle instead of the fast path",
    )
    args = parser.parse_args()

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
    cat = (census if args.reference else census_fast)(succ)
    elapsed = time.perf_counter() - start
    which = "reference" if args.reference else "fast"
    print(
        f"census[{which}]: {cat.triples_counted:,} triples in {elapsed:.1f}s "
        f"(exact={cat.exact})"
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
