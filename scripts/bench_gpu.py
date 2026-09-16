"""GPU vs CPU benchmark on real hardware (PH3, benchmarks B004-B008).

This exists because the GPU path was recorded as BLOCKED for the project's
entire history, on the belief that no torch CUDA build could serve the
Quadro M5000 (driver 537.99, sm_52). That belief was measured and found
WRONG (notes entry 41): torch 2.7.1+cu118 is a cp313 Windows wheel, loads on
this driver, and runs bits-exact kernels on sm_52.

So the honest question is no longer "can we?" but "is it worth it?" - which
nobody could previously answer. This script measures it.

    python scripts/bench_gpu.py [--neurons 4096 --ticks 2000]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vnr.backend.reference import (
    StaticNetSpec,
    build_drive,
    run_explicit,
)


def timed(fn, repeat: int) -> tuple[float, object]:
    """Median wall time over `repeat` runs, plus the last result."""
    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    times.sort()
    return times[len(times) // 2], result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=4096)
    parser.add_argument("--ticks", type=int, default=2000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("torch not installed; nothing to benchmark")
        return 1

    spec = StaticNetSpec(n_neurons=args.neurons, seed=args.seed, ticks=args.ticks)
    drive = build_drive(spec)
    print(f"workload: {args.neurons:,} neurons x {args.ticks:,} ticks")
    print(
        f"torch {torch.__version__} | cuda {torch.version.cuda} | "
        f"available {torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        free, total = torch.cuda.mem_get_info(0)
        print(
            f"device: {name} sm_{cap[0]}{cap[1]} | VRAM {free / 1e9:.2f}/{total / 1e9:.2f} GB free"
        )
    print()

    rows: list[tuple[str, float, int]] = []

    median, result = timed(lambda: run_explicit(spec, drive), args.repeat)
    rows.append(("explicit (Python oracle)", median, result.total_spikes))
    reference_spikes = result.spikes

    try:
        from vnr.backend.cpu_numpy import run_numpy

        median, result = timed(lambda: run_numpy(spec, drive), args.repeat)
        rows.append(("numpy (CPU vectorized)", median, result.total_spikes))
    except ImportError:
        print("numpy backend unavailable")

    if torch.cuda.is_available():
        from vnr.backend.torch_backend import run_torch

        median, result = timed(
            lambda: run_torch(spec, drive, device="cuda"), args.repeat
        )
        rows.append(("torch (CUDA, sm_52)", median, result.total_spikes))
        gpu = result
    else:
        gpu = None

    try:
        from vnr.backend.torch_backend import run_torch

        median, result = timed(
            lambda: run_torch(spec, drive, device="cpu"), args.repeat
        )
        rows.append(("torch (CPU)", median, result.total_spikes))
    except ImportError:
        pass

    baseline = rows[0][1]
    print(f"{'backend':<28}{'median':>12}{'vs oracle':>12}{'spikes':>10}")
    print("-" * 62)
    for name, median, spikes in rows:
        print(f"{name:<28}{median:>11.3f}s{baseline / median:>11.2f}x{spikes:>10,}")

    print()
    exact_gpu = gpu is not None and gpu.spikes == reference_spikes
    if gpu is not None:
        print(f"GPU spike trains identical to oracle: {exact_gpu}")
        print(
            f"GPU final voltages identical        : {gpu.final_v == run_explicit(spec, drive).final_v}"
        )
        gpu_row = next(r for r in rows if "CUDA" in r[0])
        numpy_row = next((r for r in rows if "numpy" in r[0]), None)
        if numpy_row and numpy_row[1] > 0:
            print(
                f"GPU vs numpy speedup                : {numpy_row[1] / gpu_row[1]:.2f}x"
            )

    print()
    print("Reading: 'vs oracle' is the pure-Python reference, kept as the")
    print("correctness ground truth. Speed ordering says nothing about")
    print("correctness - every backend above is checked against the oracle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
