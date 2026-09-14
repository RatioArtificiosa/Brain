"""Hardware discovery → HardwareProfile + RuntimeBudget (plan PH0-WI04).

Never claims more than physically present minus safety reserves (plan D2.8):
20% VRAM reserve, 15% RAM reserve. All fields optional-tolerant: missing GPU or
torch degrades to informative "missing/optional" status instead of crashing.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class HardwareProfile:
    cpu_name: str = "unknown"
    cpu_cores: int = 0
    cpu_threads: int = 0
    ram_bytes: int = 0
    gpu_name: str | None = None
    gpu_vram_bytes: int = 0
    cuda_version: str | None = None
    compute_capability: str | None = None
    driver_version: str | None = None
    torch_version: str | None = None
    torch_cuda_available: bool = False
    disk_free_bytes: int = 0


@dataclass
class RuntimeBudget:
    max_gpu_resident_bytes: int = 0
    max_cpu_resident_bytes: int = 0
    soft_gpu_limit: int = 0
    soft_cpu_limit: int = 0
    active_neuron_budget: int = 100_000
    active_synapse_budget: int = 20_000_000
    event_queue_budget: int = 1_000_000
    notes: list[str] = field(default_factory=list)


def _windows_total_ram() -> int:
    """Total physical RAM via Win32 API; 0 on non-Windows or failure."""
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
    except (AttributeError, OSError):
        # Non-Windows (no ctypes.windll) or API failure: caller treats 0 as unknown.
        # Logging lands in PH5-WI05; until then silence here is a documented choice.
        return 0
    return 0


def _nvidia_smi(fields: str) -> str | None:
    try:
        out = subprocess.run(  # fixed argv, no shell; returncode checked below
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,  # returncode inspected below; missing GPU is normal
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def discover_hardware() -> HardwareProfile:
    """Probe the machine. Pure stdlib except an optional torch import."""
    import os

    prof = HardwareProfile()
    prof.cpu_name = platform.processor() or platform.machine()
    prof.cpu_cores = os.cpu_count() or 0
    prof.cpu_threads = os.cpu_count() or 0
    try:
        import psutil  # optional, not a hard dependency

        prof.ram_bytes = psutil.virtual_memory().total
    except ImportError:
        prof.ram_bytes = _windows_total_ram()
    if prof.cpu_threads == 0:
        prof.cpu_threads = (
            1  # pathological fallback keeps doctor honest, never zero-divides
        )
    smi = _nvidia_smi("name,memory.total,driver_version")
    if smi:
        parts = [p.strip() for p in smi.splitlines()[0].split(",")]
        if len(parts) >= 3:
            prof.gpu_name = parts[0]
            try:
                prof.gpu_vram_bytes = int(float(parts[1].split()[0]) * 1024**2)
            except ValueError:
                pass
            prof.driver_version = parts[2]
    try:
        import torch

        prof.torch_version = torch.__version__
        prof.torch_cuda_available = bool(torch.cuda.is_available())
        if prof.torch_cuda_available:
            cap = torch.cuda.get_device_capability(0)
            prof.compute_capability = f"{cap[0]}.{cap[1]}"
            prof.cuda_version = torch.version.cuda
    except ImportError:
        pass
    prof.disk_free_bytes = shutil.disk_usage(os.getcwd()).free
    return prof


def derive_budget(prof: HardwareProfile) -> RuntimeBudget:
    """Apply D2.8 safety reserves. Budgets count *usable* bytes, never totals."""
    budget = RuntimeBudget()
    budget.max_cpu_resident_bytes = int(prof.ram_bytes * 0.85)
    budget.soft_cpu_limit = int(prof.ram_bytes * 0.70)
    budget.max_gpu_resident_bytes = int(prof.gpu_vram_bytes * 0.80)
    budget.soft_gpu_limit = int(prof.gpu_vram_bytes * 0.60)
    if prof.gpu_vram_bytes == 0:
        budget.notes.append(
            "no NVIDIA GPU detected: GPU budgets are zero, CPU-only mode"
        )
    if prof.torch_version is None:
        budget.notes.append(
            "torch not installed: GPU backend unavailable (install vnr[gpu])"
        )
    elif not prof.torch_cuda_available:
        budget.notes.append(
            f"torch {prof.torch_version} present but CUDA unavailable: "
            "check driver/CUDA build (esp. sm_52 Maxwell support)"
        )
    return budget


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"
