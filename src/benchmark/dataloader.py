"""
DESCRIPTION:
    Loads pre-processed battery Q-V curve tensors (.pt files) for
    a given data split and returns stacked arrays for inference.
"""

import numpy as np
import torch
import yaml
from pathlib import Path
from typing import List, Tuple


def load_data(
    data_dir: Path,
    configs_path: Path,
    split: str = "test",
    nominal_capacity: float = 2.4,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    """
    Load Q-V curve tensors and capacity labels for a given split.

    Args:
        data_dir:          Directory containing the .pt tensor files.
        configs_path:      Path to configs.yaml with train/val/test cell IDs.
        split:             Which split to load ('train', 'val', or 'test').
        nominal_capacity:  Nominal cell capacity (Ah) used to normalise y.

    Returns:
        X:          float32 array (N, 120) — Q-V input curves.
        y:          float32 array (N,)    — normalised capacity (0-1).
        boundaries: list of (start, end) index pairs, one per cell in split order.
    """
    with open(configs_path, "r") as f:
        cfg = yaml.safe_load(f)

    cell_ids: List[str] = cfg["splits"][split]

    X_list, y_list, boundaries = [], [], []
    offset = 0

    for cell_id in cell_ids:
        # Files are named like '03_Rd_3C.pt' — match on the numeric prefix
        matches = sorted(Path(data_dir).glob(f"{cell_id}_*.pt"))
        if not matches:
            continue
        data = torch.load(matches[0], weights_only=True)
        x      = data["X"].numpy().astype(np.float32)   # (num_cycles, 120)
        y_raw  = data["y"].numpy().astype(np.float32)   # (num_cycles,) in Ah
        y_norm = y_raw / nominal_capacity               # normalise to 0-1

        n = len(x)
        X_list.append(x)
        y_list.append(y_norm)
        boundaries.append((offset, offset + n))
        offset += n

    if not X_list:
        raise FileNotFoundError(
            f"No .pt files found for split '{split}' in {data_dir}"
        )

    return (
        np.concatenate(X_list, axis=0),  # (N, 120)
        np.concatenate(y_list, axis=0),  # (N,)
        boundaries,
    )
