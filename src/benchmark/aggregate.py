"""
DESCRIPTION:
    Scans all per-engine result directories under results/<folder>/ and
    compiles a single summary.csv from the per-engine latency.csv,
    accuracy.csv, and power.log files.

USAGE:
    Called automatically at the end of benchmark.py, or standalone:
        from src.benchmark.aggregate import aggregate
        aggregate(Path("results/v4"))
"""

import csv
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# =========================================================================
#                         Power log parsing
# =========================================================================

# Tegrastats line format (Jetson Orin Nano):
#   ... VDD_IN 3000mW/3000mW VDD_CPU_CV 500mW/500mW VDD_SOC 1000mW/1000mW ...
# The first value is instantaneous, the second is a rolling average.
# We use the instantaneous value for per-sample statistics.

_PWR_RE = {
    "VDD_IN":     re.compile(r"VDD_IN\s+(\d+)mW/\d+mW"),
    "VDD_SOC":    re.compile(r"VDD_SOC\s+(\d+)mW/\d+mW"),
    "VDD_CPU_CV": re.compile(r"VDD_CPU_CV\s+(\d+)mW/\d+mW"),
}


def _parse_power_log(log_path: Path) -> dict:
    """
    Parse a tegrastats log file and return mean/max for each power rail.
    Returns a dict with keys: mean_VDD_IN, mean_VDD_SOC, mean_VDD_CPU_CV
    and corresponding max_ entries. All values in mW, or None if unavailable.
    """
    if not log_path.exists() or log_path.stat().st_size == 0:
        return {f"{stat}_{rail}": None
                for rail in _PWR_RE for stat in ("mean", "max")}

    samples = {rail: [] for rail in _PWR_RE}

    with open(log_path, "r") as f:
        for line in f:
            for rail, pattern in _PWR_RE.items():
                m = pattern.search(line)
                if m:
                    samples[rail].append(int(m.group(1)))

    result = {}
    for rail, vals in samples.items():
        if vals:
            result[f"mean_{rail}"] = round(float(np.mean(vals)), 2)
            result[f"max_{rail}"]  = round(float(np.max(vals)), 2)
        else:
            result[f"mean_{rail}"] = None
            result[f"max_{rail}"]  = None

    return result


# =========================================================================
#                         CSV readers
# =========================================================================

def _read_latency_csv(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def _read_accuracy_csv(path: Path) -> Optional[dict]:
    """Returns dict indexed by 'cell' value -> row dict."""
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return {row["cell"]: row for row in csv.DictReader(f)}


# =========================================================================
#                           aggregate()
# =========================================================================

def aggregate(results_dir: Path) -> None:
    """
    Walk results_dir/<model>/bs<N>/<precision>/ directories, gather per-engine
    metrics, and write results_dir/summary.csv.
    """
    # Discover all leaf precision directories
    engine_dirs = sorted(
        p for p in results_dir.rglob("*")
        if p.is_dir()
        and (p / "latency.csv").exists()
    )

    if not engine_dirs:
        print(f"[aggregate] No result directories found under {results_dir}")
        return

    # Determine cell columns from the first accuracy.csv we can find
    n_cells = 0
    for d in engine_dirs:
        acc = _read_accuracy_csv(d / "accuracy.csv")
        if acc:
            n_cells = sum(1 for k in acc if k.startswith("cell_"))
            break

    cell_cols = [f"cell_{i+1:02d}_rmse_ah" for i in range(n_cells)]

    fieldnames = [
        "model", "precision", "batch_size",
        "global_rmse_ah", "global_maxe_ah",
        *cell_cols,
        "p95_latency_ms", "mean_latency_ms",
        "throughput_cells_per_sec", "norm_latency_ms_per_cell",
        "mean_power_mw_total", "mean_power_mw_gpu",
        "energy_per_inference_mj",
    ]

    rows = []

    for eng_dir in engine_dirs:
        # Parse path: results/<folder>/<model>/bs<N>/<precision>
        parts = eng_dir.relative_to(results_dir).parts
        if len(parts) < 3:
            continue
        model      = parts[0]
        bs_str     = parts[1]   # e.g. 'bs1' or 'bs32'
        precision  = parts[2]   # e.g. 'fp32'
        try:
            batch_size = int(bs_str.replace("bs", ""))
        except ValueError:
            continue

        lat  = _read_latency_csv(eng_dir / "latency.csv")
        acc  = _read_accuracy_csv(eng_dir / "accuracy.csv")
        pwr  = _parse_power_log(eng_dir / "power.log")

        # Latency fields
        mean_lat = float(lat["mean_latency_ms"])  if lat else None
        p95_lat  = float(lat["p95_latency_ms"])   if lat else None
        thru     = float(lat["throughput_cells_per_sec"]) if lat else None
        norm_lat = float(lat["norm_latency_ms_per_cell"]) if lat else None

        # Accuracy fields
        global_rmse = float(acc["global"]["rmse_ah"]) if acc and "global" in acc else None
        global_maxe = float(acc["global"]["max_abs_error_ah"]) if acc and "global" in acc else None

        cell_rmse_vals = []
        for i in range(n_cells):
            key = f"cell_{i+1:02d}"
            val = float(acc[key]["rmse_ah"]) if acc and key in acc else None
            cell_rmse_vals.append(val)

        # Power fields
        mean_pwr_total = pwr.get("mean_VDD_IN")
        mean_pwr_gpu   = pwr.get("mean_VDD_SOC")

        # Energy per inference: mW * ms / 1000 = mJ
        if mean_pwr_total is not None and mean_lat is not None:
            energy_mj = round(mean_pwr_total * mean_lat / 1000, 4)
        else:
            energy_mj = None

        row = {
            "model":                    model,
            "precision":                precision,
            "batch_size":               batch_size,
            "global_rmse_ah":           global_rmse,
            "global_maxe_ah":           global_maxe,
            "p95_latency_ms":           p95_lat,
            "mean_latency_ms":          mean_lat,
            "throughput_cells_per_sec": thru,
            "norm_latency_ms_per_cell": norm_lat,
            "mean_power_mw_total":      mean_pwr_total,
            "mean_power_mw_gpu":        mean_pwr_gpu,
            "energy_per_inference_mj":  energy_mj,
        }
        for i, val in enumerate(cell_rmse_vals):
            row[cell_cols[i]] = val

        rows.append(row)

    summary_path = results_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[aggregate] Summary written -> {summary_path}  ({len(rows)} engines)")
