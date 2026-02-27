#!/usr/bin/env python3
"""
Latency and power benchmarking for TensorRT engines on the Jetson Orin Nano.

Runs every .engine file found under --models.  The batch size baked into
each engine determines the scenario automatically:
  - batch=1  → single-cell scenario (one Q-V curve per inference)
  - batch=32 → 32-cell pack scenario (32 Q-V curves in one GPU call)

Run both scenarios by pointing at the single-cell and pack engine folders
in separate calls; results accumulate in hardware_benchmark_raw.csv.

Usage:
    python benchmark.py --models models/v2     --data data/tensor_qv --output results/
    python benchmark.py --models models/v2_b32 --data data/tensor_qv --output results/
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.benchmark.data       import load_test_cells, load_test_data, make_batches
from src.benchmark.inferencer import TRTInferencer
from src.benchmark.power      import PowerProfiler
from src.utils.notify         import send_notification


# Un-timed warm-up passes before each engine is benchmarked.
# Lets the GPU clock ramp to steady state and fills the TRT kernel cache.
WARMUP_RUNS = 50


def benchmark_engine(
    engine_path: Path,
    samples:     List[np.ndarray],
    output_dir:  Path,
) -> List[Dict]:
    """
    Benchmark one engine and return a list of raw per-batch result dicts.

    The engine's batch size is read directly from its binding shape:
      - batch=1  → single-cell scenario; one inference per Q-V curve
      - batch=32 → 32-cell pack scenario; 32 curves inferred in one GPU call

    Each dict contains: engine, model, precision, batch_size, batch_idx,
    latency_ms.  Power is in a separate tegrastats .log file.
    """
    print(f"  Loading engine ...")
    inferencer = TRTInferencer(engine_path)
    batch_size = inferencer.batch_size

    # Build warm-up input matching the engine's expected shape
    warmup_input = np.vstack(
        [samples[i % len(samples)] for i in range(batch_size)]
    )
    print(f"  Warming up ({WARMUP_RUNS} runs, batch={batch_size}) ...")
    for _ in range(WARMUP_RUNS):
        inferencer.infer(warmup_input)

    # Group samples into (batch_size, 120) arrays.
    # For batch=1 this is just the original list of (1, 120) arrays.
    # For batch=32 this tiles 13 test cells to the nearest multiple of 32
    # and stacks them — valid for latency benchmarking since timing depends
    # on arithmetic load, not which specific cells are in the batch.
    batches = make_batches(samples, batch_size)

    # Parse model name and precision from filename: CNN-LSTM_int8.engine
    parts     = engine_path.stem.rsplit("_", 1)
    model     = parts[0]
    precision = parts[1] if len(parts) == 2 else "unknown"

    # Start power logging immediately before the timed loop
    log_path = output_dir / f"{engine_path.stem}_power.log"
    profiler  = PowerProfiler()
    profiler.start(log_path)

    rows = []
    for idx, batch in enumerate(batches):
        _, latency_ms = inferencer.infer(batch)
        rows.append({
            "engine":     engine_path.name,
            "model":      model,
            "precision":  precision,
            "batch_size": batch_size,
            "batch_idx":  idx,
            "latency_ms": round(latency_ms, 4),
        })

    profiler.stop()

    print(f"  {len(rows)} batches × {batch_size} cells  |  power log → {log_path.name}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", type=Path, required=True,
                        help="Directory with model sub-folders containing .engine files")
    parser.add_argument("--data",   type=Path, required=True,
                        help="Directory of .pt test tensors (e.g. data/tensor_qv)")
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Directory for CSV and power logs (default: results/)")
    args = parser.parse_args()

    repo_root  = Path(__file__).resolve().parent
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load all test data into RAM before any engine or timer starts
    print("Loading test data into RAM ...")
    cell_ids = load_test_cells(repo_root)
    samples  = load_test_data(args.data, cell_ids)
    print()

    engine_files = sorted(args.models.glob("*/*/*.engine"))

    if not engine_files:
        print(f"ERROR: no .engine files found under {args.models}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(engine_files)} engine(s)\n")

    # Open CSV in append mode — repeated runs accumulate in one file.
    # batch_size in each row distinguishes single-cell (1) from pack (32) engines.
    csv_path     = output_dir / "hardware_benchmark_raw.csv"
    fieldnames   = ["engine", "model", "precision", "batch_size", "batch_idx", "latency_ms"]
    write_header = not csv_path.exists()
    csv_file     = open(csv_path, "a", newline="")
    writer       = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    failed = []
    for engine_path in engine_files:
        print(f"{'─' * 60}")
        print(f"  Engine: {engine_path.name}")
        try:
            rows = benchmark_engine(engine_path, samples, output_dir)
            writer.writerows(rows)
            csv_file.flush()
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failed.append(engine_path.name)
        print()

    csv_file.close()
    print(f"{'=' * 60}")
    print(f"  Done.  Results → {csv_path}")
    print(f"{'=' * 60}")

    if failed:
        send_notification(
            f"❌ Benchmark finished with errors — {args.models}\n"
            f"Failed: {', '.join(failed)}"
        )
    else:
        send_notification(
            f"✅ Benchmark complete — {args.models}\n"
            f"Results: {csv_path}"
        )


if __name__ == "__main__":
    main()
