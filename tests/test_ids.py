"""PH1-WI01 tests: determinism, 64-bit range, avalanche, collisions, throughput."""

import time

from vnr.core.ids import virtual_id, virtual_ids


def test_deterministic_across_calls():
    args = (12345, 7, 42, 3, 999)
    assert virtual_id(*args) == virtual_id(*args)


def test_ids_are_uint64():
    for i in range(1000):
        value = virtual_id(1, 2, 3, 4, i)
        assert 0 <= value < 2**64


def test_negative_and_huge_inputs_wrap_safely():
    assert virtual_id(-1, -2, -3, -4, -5) == virtual_id(
        2**64 - 1, 2**64 - 2, 2**64 - 3, 2**64 - 4, 2**64 - 5
    )
    assert 0 <= virtual_id(2**200, 0, 0, 0, 0) < 2**64


def test_seed_change_avalanches():
    base = [virtual_id(12345, 1, 1, 0, i) for i in range(1000)]
    changed = [virtual_id(12346, 1, 1, 0, i) for i in range(1000)]
    assert all(a != b for a, b in zip(base, changed))


def test_every_field_matters():
    ref = virtual_id(11, 22, 33, 44, 55)
    variants = [
        virtual_id(12, 22, 33, 44, 55),
        virtual_id(11, 23, 33, 44, 55),
        virtual_id(11, 22, 34, 44, 55),
        virtual_id(11, 22, 33, 45, 55),
        virtual_id(11, 22, 33, 44, 56),
    ]
    assert all(v != ref for v in variants)


def test_zero_collisions_on_1M_and_bulk_matches_single():
    ids = virtual_ids(777, 3, 9, 1, range(1_000_000))
    assert len(set(ids)) == 1_000_000
    assert ids[:1000] == [virtual_id(777, 3, 9, 1, i) for i in range(1000)]


def test_throughput_floor_single_shot():
    start = time.perf_counter()
    n = 200_000
    for i in range(n):
        virtual_id(5, 5, 5, 0, i)
    rate = n / (time.perf_counter() - start)
    print(f"\nvirtual_id throughput: {rate:,.0f}/s")
    assert rate >= 300_000, f"ID generation too slow: {rate:,.0f}/s"
