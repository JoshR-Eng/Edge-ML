"""
Utilities for discovering and describing TensorRT engine files.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


PRECISIONS = ("fp32", "fp16", "int8")


@dataclass
class EngineInfo:
    path:       Path        # Absolute path to the .engine file
    model:      str         # e.g. GRU, LSTM
    batch_size: int         # e.g. 1, 32, 96
    precision:  str         # fp32, fp16 or int8
    subdirs:    List[str]   # Original subdirs list

def find_engines(root: Path) -> List[EngineInfo]:
    """
    Recursively find all .engine files under root and return their metadata.
    """
    
    results = []


    for engine_path in sorted(root.rglob("*.engine")):
        # Directory names from root down to the file's parent
        subdirs = list(engine_path.relative_to(root).parts[:-1])

        # Extract precision from filename (e.g. GRU_fp32.engine → 'fp32')
        name = engine_path.stem.lower()
        match = re.search(r"(fp32|fp16|int8)", name)
        precision = match.group(1) if match else None

        # Dynamiccaly extract Model and Batch size from folder structure
        # Assume strict structure:
        #       "models/<suggested_dir>/<model>/bs<batch_size>/..."
        try:
            model_name = subdirs[0] # e.g. 'GRU'

            # Find the folder that starts with 'bs' and extract the integer
            bs_str = next(s for s in subdirs if s.startswith('bs'))
            batch_size = int(bs_str.replace('bs', ''))
        except (IndexError, StopIteration, ValueError):
            print(f"WARNING: Could not parse model/batch size for {engine_path}")
            continue

        results.append(EngineInfo(
            path = engine_path,
            model = model_name,
            batch_size = batch_size,
            precision = precision,
            subdirs = subdirs
        ))

    return results
