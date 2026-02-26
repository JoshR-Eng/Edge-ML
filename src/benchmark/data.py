"""
Data loading utilities for the benchmark.

Loads Q-V cycles from held-out test cells into RAM before any engine is
touched, so disk I/O cannot inflate latency measurements during timing.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml


def load_test_cells(repo_root: Path) -> List[str]:
    """Return the held-out test cell IDs from configs.yaml."""
    with open(repo_root / "configs.yaml") as f:
        return yaml.safe_load(f)["splits"]["test"]


def load_test_data(data_dir: Path, cell_ids: List[str]) -> List[np.ndarray]:
    """
    Load Q-V cycles for the given cells into RAM as individual (1, 120) arrays.

    Each array is one Q-V curve — the exact shape the TRT engines expect.
    """
    all_files = sorted(data_dir.glob("*.pt"))
    samples = []

    for cell_id in cell_ids:
        pt_file = next((f for f in all_files if f.stem.startswith(f"{cell_id}_")), None)
        if pt_file is None:
            print(f"  Warning: no file found for test cell {cell_id}, skipping.")
            continue

        data = torch.load(pt_file, map_location="cpu", weights_only=True)
        x = data["X"].numpy().astype(np.float32)  # (num_cycles, 120)

        for i in range(len(x)):
            samples.append(x[i : i + 1])  # split into individual (1, 120) arrays

    print(f"  Loaded {len(samples)} Q-V cycles from {len(cell_ids)} test cells")
    return samples


def make_batches(samples: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
    """
    Group individual (1, 120) samples into (batch_size, 120) arrays.

    Used for the 32-cell pack scenario where all cells are inferred in one
    GPU call.  If the number of samples isn't divisible by batch_size, the
    list is tiled from the beginning until it is.  This is valid for a latency
    benchmark — the GPU timing depends on the arithmetic load, not the specific
    cell values.

    Example: 13 test cells → tile to 32 → one batch of shape (32, 120).
    """
    # Tile to the next multiple of batch_size if needed
    remainder = len(samples) % batch_size
    if remainder:
        samples = samples + samples[: batch_size - remainder]

    return [
        np.vstack(samples[i : i + batch_size])
        for i in range(0, len(samples), batch_size)
    ]
