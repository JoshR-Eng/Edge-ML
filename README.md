# Edge-ML

TensorRT inference benchmarking suite for battery State-of-Health (SoH) models on the NVIDIA Jetson Orin Nano 4GB. Takes trained PyTorch models exported from [Battery-ML-Training](https://github.com/JoshR-ENG/Battery-ML-Training), compiles them into optimised TensorRT engines (FP32, FP16, INT8), and evaluates them using a two-pass methodology that separates accuracy/latency measurement from steady-state power profiling.

---

## Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Workflow](#workflow)
  - [Step 1 — Compile TensorRT Engines](#step-1--compile-tensorrt-engines)
  - [Step 2 — Run the Benchmark](#step-2--run-the-benchmark)
- [Benchmark Methodology](#benchmark-methodology)
  - [Pass 1: Accuracy & Latency](#pass-1-accuracy--latency)
  - [Pass 2: Steady-State Power](#pass-2-steady-state-power)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Power Modes (Jetson Orin Nano)](#power-modes-jetson-orin-nano)
- [Project Structure](#project-structure)

---

## Overview

This repository is the **inference and evaluation** stage of a two-repo pipeline:

```
Battery-ML-Training  ──(ONNX exports)──►  Edge-ML (this repo)
   PyTorch training                         TensorRT compilation
   ONNX export                              Benchmarking on Jetson
```

Models exported as `.onnx` from `Battery-ML-Training` are compiled here into `.engine` files in three precision formats:

| Precision | Description |
|-----------|-------------|
| **FP32**  | Full float32 baseline — no quantisation |
| **FP16**  | Half-precision float — ~2× speedup on Tensor Cores with minimal accuracy loss |
| **INT8**  | 8-bit integer — maximum throughput, requires calibration data |

INT8 calibration uses the **training split** of the dataset (70% of cells) to compute per-layer activation scale factors, ensuring the held-out test split is never used during calibration.

---

## Prerequisites

- NVIDIA Jetson Orin Nano 4GB running **JetPack 5.x**
- Python 3 with a virtual environment at `.venv/` (recommended)
- TensorRT and CUDA (bundled with JetPack)
- `tegrastats` available on the system path (bundled with JetPack)
- ONNX models placed under `models/<folder>/`
- Preprocessed test tensors under `data/tensor_qv/`

Install Python dependencies inside your venv before running:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # numpy, pyyaml, tensorrt bindings, etc.
```

> **Optional:** Create a `.env` file in the repo root to set `DISCORD_WEBHOOK_URL` for benchmark completion notifications.

---

## Workflow

### Step 1 — Compile TensorRT Engines

Run `quantise.sh` to compile all ONNX models in a given folder into FP16 and INT8 TensorRT engines. FP32 engines are compiled as a baseline by `onnx2engine.py` automatically.

```bash
./quantise.sh <folder>
# e.g.
./quantise.sh v4
```

This script runs two steps internally:

1. **`calibration_cache.py`** — Profiles the Q-V training data and writes per-model INT8 activation scale-factor caches to `models/<folder>/`.
2. **`onnx2engine.py`** — Compiles every `.onnx` model into optimised FP32, FP16, and INT8 `.engine` files.

> The script can be sourced (`source quantise.sh v4`) or run directly. When sourced from a tmux session, it automatically re-launches in a subshell to protect the parent shell environment.

---

### Step 2 — Run the Benchmark

Use `run_benchmark.sh` to lock hardware state and execute the full benchmark suite:

```bash
./run_benchmark.sh <model_folder> [power_mode]
# e.g.
./run_benchmark.sh v4         # MAXN mode (default)
./run_benchmark.sh v4 1       # 7W_AI efficiency mode
```

The script:
1. Activates `.venv` and loads `.env` if present
2. Locks the Jetson hardware state with `sudo nvpmodel` and `sudo jetson_clocks`
3. Sets a fixed fan PWM (default: 127/255) to eliminate thermal variability
4. Runs `benchmark.py` with elevated privileges via `sudo -E` (preserving env vars)
5. Restores automatic fan control on completion

> A 10-second startup delay is built into `benchmark.py` — this window can be used to detach from an SSH session (e.g. via `tmux`) to remove SSH overhead before inference begins.

---

## Benchmark Methodology

The benchmark uses a **two-pass approach** to eliminate the observer effect — power logging overhead must not inflate latency measurements, and latency-optimised inference must not suppress the GPU into a low-power state during power profiling.

### Pass 1: Accuracy & Latency

Runs with **no power-logging overhead**. All test samples are passed through the engine, recording per-sample predictions and wall-clock latencies.

- **Warmup:** 50 inference iterations are discarded before timing begins.
- **Padding:** For batch sizes > 1, dummy samples pad the dataset to a full batch; dummy predictions are discarded before accuracy calculation.
- **Outputs:** `accuracy.csv` and `latency.csv`

Latency metrics recorded:

| Metric | Description |
|--------|-------------|
| `mean_latency_ms` | Mean inference time per batch |
| `p95_latency_ms` | 95th-percentile latency (tail latency) |
| `throughput_cells_per_sec` | Cells processed per second |
| `norm_latency_ms_per_cell` | Mean latency normalised per cell (batch-size independent) |

Accuracy metrics (denormalised to Ah against nominal capacity of 2.4 Ah):

| Metric | Description |
|--------|-------------|
| `rmse_ah` | Root mean square error |
| `mae_ah` | Mean absolute error |
| `max_abs_error_ah` | Worst-case absolute error |

Both global and per-cell breakdowns are written.

### Pass 2: Steady-State Power

Runs inference **continuously for a fixed 10-second window** with `tegrastats` recording hardware telemetry. Predictions are discarded — only `power.log` is written.

The fixed time window is critical for fast models: a TCN with batch size 96 can complete a full dataset pass in ~7 ms, yielding far too few `tegrastats` samples (default 50 ms interval) if inference were stopped at dataset end. The 10-second window guarantees ~200 samples regardless of model speed.

`tegrastats` captures:

- `VDD_IN` — Total board input power
- `VDD_SOC` — SoC power rail
- `VDD_CPU_CV` — CPU + CV engine power rail

---

## Output Files

Results are written to `results/<run_name>/<model>/bs<N>/<precision>/`:

```
results/
└── v4_powm0/
    └── <model_name>/
        └── bs<batch_size>/
            └── <precision>/        # fp32 | fp16 | int8
                ├── accuracy.csv
                ├── latency.csv
                └── power.log
```

A final aggregated `summary.csv` is written to `results/<run_name>/summary.csv`, combining all engines into a single table for cross-model comparison.

---

## Configuration

`configs.yaml` defines the train/val/test cell splits. These splits must match those used in `Battery-ML-Training` to ensure correct INT8 calibration and unbiased evaluation.

| Split | Proportion | Purpose |
|-------|-----------|---------|
| `train` | 70% (47 cells) | Model training + INT8 calibration |
| `val`   | 15% (12 cells) | Hyperparameter tuning / early stopping |
| `test`  | 15% (13 cells) | Held-out evaluation only — **never used for calibration or training** |

The `NOMINAL_CAPACITY` constant (2.4 Ah) is used to denormalise model outputs from the [0, 1] range back into physical units for accuracy reporting.

---

## Power Modes (Jetson Orin Nano)

| Mode | `nvpmodel -m` | TDP | Description |
|------|--------------|-----|-------------|
| MAXN | `0` | ~10 W | All CPU/GPU cores unlocked, maximum performance |
| 7W\_AI | `1` | ~7 W | CPU frequency capped, GPU prioritised |
| 7W\_CPU | `2` | ~7 W | Both CPU and GPU frequency capped |

`jetson_clocks` is called after `nvpmodel` to lock clocks to their maximum within the selected power envelope, preventing dynamic frequency scaling from introducing variance in benchmark results.

---

## Project Structure

```
Edge-ML/
├── benchmark.py              # Main benchmark entry point
├── run_benchmark.sh          # Hardware-locking benchmark runner
├── quantise.sh               # TensorRT compilation pipeline
├── configs.yaml              # Train/val/test cell split definitions
├── models/                   # ONNX inputs + compiled .engine outputs
│   └── <folder>/
├── data/
│   └── tensor_qv/            # Preprocessed input tensors
├── results/                  # Benchmark output (generated at runtime)
└── src/
    ├── benchmark/
    │   ├── aggregate.py       # Aggregates per-engine CSVs into summary.csv
    │   ├── dataloader.py      # Loads and batches test tensors
    │   ├── discover_files.py  # Discovers .engine files and parses metadata
    │   ├── inference.py       # TRTWrapper — loads engine, runs inference
    │   └── power_log.py       # TegrastatsLogger context manager
    ├── quantise/
    │   ├── calibration_cache.py  # Generates INT8 activation scale caches
    │   └── onnx2engine.py        # Compiles ONNX → TensorRT .engine
    └── utils/
        └── notify.py             # Optional Discord webhook notifications
```
