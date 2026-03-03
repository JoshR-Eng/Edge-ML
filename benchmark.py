"""
DESCRIPTION:
    Benchmarks all TensorRT .engine files found under MODELS_DIR.
    For each engine, runs one full pass over the test split data,
    scopes tegrastats power logging tightly around the inference loop,
    then computes accuracy and latency metrics post-inference.

    Output per engine:
        results/<folder>/<model>/bs<N>/<precision>/
            power.log      — raw tegrastats output (VDD_IN, VDD_SOC, VDD_CPU_CV)
            latency.csv    — mean/p95 latency, throughput, normalised latency
            accuracy.csv   — global + per-cell RMSE, MAE, MaxAE

    Final output:
        results/<folder>/summary.csv  — aggregated across all engines

USAGE:
    BENCHMARK_FOLDER=v4 python benchmark.py
    (set BENCHMARK_FOLDER via environment; defaults to 'v4')
"""

# =========================================================================
#                              IMPORTS
# =========================================================================

import csv
import os
from pathlib import Path
from time import sleep

import numpy as np

from src.benchmark.aggregate    import aggregate
from src.benchmark.dataloader   import load_data
from src.benchmark.discover_files import find_engines
from src.benchmark.inference    import TRTWrapper
from src.benchmark.power_log    import TegrastatsLogger
from src.utils.notify           import send_notification


# =========================================================================
#                              CONFIG
# =========================================================================

FOLDER           = os.environ.get("BENCHMARK_FOLDER", "v4")
RUN_NAME         = os.environ.get("BENCHMARK_RUN_NAME", FOLDER)
 # for notifications (optional)...
DISCORD_URL      = os.environ.get("DISCORD_WEBHOOK_URL", None)

MODELS_DIR       = Path("models") / FOLDER
DATA_DIR         = Path("data/tensor_qv")
CONFIGS_PATH     = Path("configs.yaml")
SPLIT            = "test"
NOMINAL_CAPACITY = 2.4        # Ah — used to normalise y labels

WARMUP_ITERS     = 50         # Iterations before timed benchmark (discarded)
RESULTS_DIR      = Path("results") / RUN_NAME


# =========================================================================
#                           HELPER FUNCTIONS
# =========================================================================

def _write_latency_csv(path: Path, mean_lat: float, p95_lat: float,
                        throughput: float, norm_lat: float, n_iters: int) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mean_latency_ms", "p95_latency_ms",
            "throughput_cells_per_sec", "norm_latency_ms_per_cell", "n_iters",
        ])
        writer.writeheader()
        writer.writerow({
            "mean_latency_ms":          round(mean_lat,  4),
            "p95_latency_ms":           round(p95_lat,   4),
            "throughput_cells_per_sec": round(throughput, 4),
            "norm_latency_ms_per_cell": round(norm_lat,  4),
            "n_iters":                  n_iters,
        })


def _write_accuracy_csv(path: Path, preds: np.ndarray, y: np.ndarray,
                         boundaries: list) -> None:
    rows = []

    # Global metrics across all 1300 samples
    diff = preds - y
    global_rmse = float(np.sqrt(np.mean(diff ** 2)))
    global_mae  = float(np.mean(np.abs(diff)))
    global_maxe = float(np.max(np.abs(diff)))
    rows.append({
        "cell": "global",
        "rmse": round(global_rmse, 6),
        "mae":  round(global_mae,  6),
        "max_abs_error": round(global_maxe, 6),
        "n_samples": len(preds),
    })

    # Per-cell metrics
    for i, (start, end) in enumerate(boundaries):
        d = preds[start:end] - y[start:end]
        rows.append({
            "cell": f"cell_{i+1:02d}",
            "rmse": round(float(np.sqrt(np.mean(d ** 2))), 6),
            "mae":  round(float(np.mean(np.abs(d))),        6),
            "max_abs_error": round(float(np.max(np.abs(d))), 6),
            "n_samples": end - start,
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cell", "rmse", "mae", "max_abs_error", "n_samples"])
        writer.writeheader()
        writer.writerows(rows)


# =========================================================================
#                               MAIN
# =========================================================================

def main() -> None:
    print(f"[benchmark] Folder  : {FOLDER}")
    print(f"[benchmark] Run name: {RUN_NAME}")
    print(f"[benchmark] Split   : {SPLIT}")


    # Add a small delay so that you could use something like tmux to 
    # detach and disconnect from ssh to remove some overhead
    delay = 10
    print(f"\n\tScript will run in {delay}s")
    sleep(delay)

    # ------------------------------------------------------------------
    # 1. Pre-load all test data into RAM
    # ------------------------------------------------------------------
    print(f"[benchmark] Loading '{SPLIT}' data from {DATA_DIR} ...")
    X, y, boundaries = load_data(DATA_DIR, CONFIGS_PATH, SPLIT, NOMINAL_CAPACITY)
    n_samples = len(X)
    print(f"[benchmark] {n_samples} samples across {len(boundaries)} cells  "
          f"(shape {X.shape})\n")

    # ------------------------------------------------------------------
    # 2. Discover engines
    # ------------------------------------------------------------------
    engines = find_engines(MODELS_DIR)
    print(f"[benchmark] Found {len(engines)} engines under {MODELS_DIR}\n")

    # ------------------------------------------------------------------
    # 3. Benchmark each engine
    # ------------------------------------------------------------------
    for engine in engines:
        bs    = engine.batch_size
        label = f"{engine.model} | bs{bs} | {engine.precision}"
        print(f"[benchmark] -- {label}")

        out_dir = RESULTS_DIR / engine.model / f"bs{bs}" / engine.precision
        out_dir.mkdir(parents=True, exist_ok=True)

        # -- Prepare batched data -------------------------------------- 
        if bs == 1:
            X_run   = X                              # (1300, 120)
            n_iters = n_samples                      # 1300 iters
            n_pad   = 0
        else:
            remainder = n_samples % bs
            n_pad     = (bs - remainder) % bs        # e.g. 12 for bs=32
            if n_pad > 0:
                dummy = np.zeros((n_pad, X.shape[1]), dtype=np.float32)
                X_run = np.concatenate([X, dummy], axis=0)
            else:
                X_run = X
            n_iters = len(X_run) // bs               # e.g. 41 for bs=32

        # -- Load TensorRT engine --------------------------------------
        trt = TRTWrapper(engine.path)

        # -- Warm up GPU (not timed) -----------------------------------
        warmup_batch = X_run[:bs]
        for _ in range(WARMUP_ITERS):
            trt.infer(warmup_batch)

        # -- Pre-allocate output arrays --------------------------------
        preds_buf  = np.empty(n_iters * bs if bs > 1 else n_iters, dtype=np.float32)
        latencies  = np.empty(n_iters, dtype=np.float64)

        # -- Inference with tegrastats logging -------------------------
        power_log_path = out_dir / "power.log"
        with TegrastatsLogger(log_path=power_log_path):
            if bs == 1:
                for i in range(n_iters):
                    out, lat      = trt.infer(X_run[i : i + 1])
                    preds_buf[i]  = out[0]
                    latencies[i]  = lat
            else:
                for i in range(n_iters):
                    out, lat                          = trt.infer(X_run[i * bs : (i + 1) * bs])
                    preds_buf[i * bs : (i + 1) * bs] = out.ravel()
                    latencies[i]                      = lat
        # tegrastats is now stopped - hardware metrics logging ends here

        del trt   # free GPU memory before loading next engine

        # -- Post-inference calculations (hardware idle) --------------- 

        # Flatten padded predictions and remove dummy outputs
        if bs > 1:
            preds = preds_buf[:n_samples]   # slice off n_pad dummy predictions
        else:
            preds = preds_buf

        # Latency stats
        mean_lat  = float(np.mean(latencies))
        p95_lat   = float(np.percentile(latencies, 95))
        throughput = bs / (mean_lat / 1000.0)           # cells / second
        norm_lat  = mean_lat / bs                       # ms / cell

        # Write logs
        _write_latency_csv(
            out_dir / "latency.csv",
            mean_lat, p95_lat, throughput, norm_lat, n_iters,
        )
        _write_accuracy_csv(out_dir / "accuracy.csv", preds, y, boundaries)

        print(f"           mean={mean_lat:.3f}ms  p95={p95_lat:.3f}ms  "
              f"throughput={throughput:.1f} cells/s  "
              f"global_rmse={float(np.sqrt(np.mean((preds - y)**2))):.4f}")

    # ------------------------------------------------------------------
    # 4. Aggregate all engine results into summary.csv
    # ------------------------------------------------------------------
    print("\n[benchmark] Aggregating results ...")
    aggregate(RESULTS_DIR)

    msg = (
        f"Benchmark complete [{FOLDER}]: {len(engines)} engines evaluated. "
        f"Summary -> results/{FOLDER}/summary.csv"
    )
    print(f"\n[benchmark] {msg}")
    send_notification(f"{msg}", DISCORD_URL)


if __name__ == "__main__":
    main()
