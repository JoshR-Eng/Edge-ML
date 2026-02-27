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
from typing import Dict, List, Optional

import numpy as np

from src.benchmark.data       import load_test_cells, load_test_data, make_batches
from src.benchmark.inferencer import TRTInferencer
from src.benchmark.power      import PowerProfiler, parse_vdd_in
from src.utils.notify         import send_notification


# Un-timed warm-up passes before each engine is benchmarked.
# Lets the GPU clock ramp to steady state and fills the TRT kernel cache.
WARMUP_RUNS = 50


def benchmark_engine(
    engine_path: Path,
    samples:     List[np.ndarray],
    output_dir:  Path,
) -> Optional[Dict]:
    """
    Benchmark one engine and return a summary stats dict.

    Writes a single-row CSV to:
        output_dir/<model>/bs<batch_size>/<model>_<batch_size>_<precision>.csv

    Also appends the same stats to the master summary CSV at output_dir root.

    Returns the stats dict, or None on failure.
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

    batches = make_batches(samples, batch_size)

    # Parse model name and precision from filename: CNN-LSTM_int8.engine
    parts     = engine_path.stem.rsplit("_", 1)
    model     = parts[0]
    precision = parts[1] if len(parts) == 2 else "unknown"

    # Temp power log written to the engine's own directory
    log_path = engine_path.parent / f"{engine_path.stem}_power.log"
    profiler  = PowerProfiler()
    profiler.start(log_path)

    latencies = []
    for batch in batches:
        _, latency_ms = inferencer.infer(batch)
        latencies.append(latency_ms)

    profiler.stop()

    # Parse power log then remove the raw file
    mean_power, max_power = parse_vdd_in(log_path)
    log_path.unlink(missing_ok=True)

    # Latency summary statistics
    arr = np.array(latencies)
    stats = {
        "model":           model,
        "precision":       precision,
        "batch_size":      batch_size,
        "num_batches":     len(latencies),
        "mean_latency_ms": round(float(np.mean(arr)),           4),
        "p50_latency_ms":  round(float(np.percentile(arr, 50)), 4),
        "p95_latency_ms":  round(float(np.percentile(arr, 95)), 4),
        "std_latency_ms":  round(float(np.std(arr)),            4),
        "mean_power_mw":   round(mean_power, 1) if mean_power is not None else None,
        "max_power_mw":    round(max_power,  1) if max_power  is not None else None,
    }

    # Per-engine CSV: output_dir/<model>/bs<batch_size>/<model>_<batch_size>_<precision>.csv
    csv_dir = output_dir / model / f"bs{batch_size}"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{model}_{batch_size}_{precision}.csv"

    fieldnames = list(stats.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(stats)

    print(f"  {len(latencies)} batches * {batch_size} cells  |  CSV >> {csv_path.relative_to(output_dir)}")
    return stats


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
                        help="Directory for CSV outputs (default: results/)")
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

    # Master summary CSV at the output root — one row per engine
    fieldnames  = ["model", "precision", "batch_size", "num_batches",
                   "mean_latency_ms", "p50_latency_ms", "p95_latency_ms",
                   "std_latency_ms", "mean_power_mw", "max_power_mw"]
    master_path = output_dir / "master_benchmark_summary.csv"
    write_header = not master_path.exists()
    master_file  = open(master_path, "a", newline="")
    master_writer = csv.DictWriter(master_file, fieldnames=fieldnames)
    if write_header:
        master_writer.writeheader()

    failed = []
    for engine_path in engine_files:
        print(f"{'-' * 60}")
        print(f"  Engine: {engine_path.name}")
        try:
            stats = benchmark_engine(engine_path, samples, output_dir)
            if stats:
                master_writer.writerow(stats)
                master_file.flush()
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failed.append(engine_path.name)
        print()

    master_file.close()
    print(f"{'=' * 60}")
    print(f"  Done.  Master summary >> {master_path}")
    print(f"{'=' * 60}")

    if failed:
        send_notification(
            f"Benchmark finished with errors - {args.models}\n"
            f"Failed: {', '.join(failed)}"
        )
    else:
        send_notification(
            f"Benchmark complete - {args.models}\n"
            f"Master summary: {master_path}"
        )


if __name__ == "__main__":
    main()
