"""Deterministic uint64 virtual neuron IDs (plan PH1-WI01, spec §20).

Identity = blake2b-8 over (root_seed, module_id, lineage_id, generation,
local_index), each masked to 64 bits. Same config + seed reconstructs the same
neuron after eviction — the invariant the runtime rests on. ID generation runs
at graph-build/materialization time, never in the event hot loop, so pure-Python
throughput (~10⁶/s) is sufficient; the hot loop only *compares* IDs.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable

_MASK64 = (1 << 64) - 1
_PACK = struct.Struct("<5Q").pack


def virtual_id(
    root_seed: int,
    module_id: int,
    lineage_id: int,
    generation: int,
    local_index: int,
) -> int:
    """Deterministic uint64 ID. Inputs wrap mod 2⁶⁴ (negative/large safe)."""
    digest = hashlib.blake2b(
        _PACK(
            root_seed & _MASK64,
            module_id & _MASK64,
            lineage_id & _MASK64,
            generation & _MASK64,
            local_index & _MASK64,
        ),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little")


def virtual_ids(
    root_seed: int,
    module_id: int,
    lineage_id: int,
    generation: int,
    local_indices: Iterable[int],
) -> list[int]:
    """Bulk path: shared lineage prefix, one ID per local index."""
    seed = root_seed & _MASK64
    mod = module_id & _MASK64
    lin = lineage_id & _MASK64
    gen = generation & _MASK64
    out: list[int] = []
    for idx in local_indices:
        digest = hashlib.blake2b(
            _PACK(seed, mod, lin, gen, idx & _MASK64), digest_size=8
        ).digest()
        out.append(int.from_bytes(digest, "little"))
    return out
