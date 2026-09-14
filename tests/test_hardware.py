"""PH0 tests: hardware discovery degrades honestly; budgets respect reserves (PH0-WI04)."""

from vnr.hardware import HardwareProfile, derive_budget, discover_hardware


def test_discover_returns_sane_profile():
    prof = discover_hardware()
    assert prof.cpu_threads > 0
    assert prof.disk_free_bytes > 0


def test_gpu_detected_on_this_machine():
    prof = discover_hardware()
    assert prof.gpu_name is not None and "M5000" in prof.gpu_name
    assert prof.gpu_vram_bytes == 8 * 1024**3


def test_budget_reserves_are_exact():
    prof = HardwareProfile(ram_bytes=64 * 1024**3, gpu_vram_bytes=8 * 1024**3)
    b = derive_budget(prof)
    assert b.max_cpu_resident_bytes == int(64 * 1024**3 * 0.85)
    assert b.max_gpu_resident_bytes == int(8 * 1024**3 * 0.80)
    assert b.soft_gpu_limit < b.max_gpu_resident_bytes


def test_missing_gpu_zeroes_gpu_budget_with_note():
    b = derive_budget(HardwareProfile())
    assert b.max_gpu_resident_bytes == 0
    assert any("CPU-only" in n for n in b.notes)
