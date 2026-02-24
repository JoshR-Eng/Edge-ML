#!/usr/bin/env python3
"""
DESCRIPTION:
Generate INT8 calibration caches for TensorRT from real Q-V battery data.


OUTPUT:
    models/<FOLDER>/<ModelName>/<ModelName>.cache


REQUIREMENTS:  (available on Jetson / any TRT host)
    - tensorrt  
    - pycuda  
    - torch  
    - numpy


USAGE:
    python calibration_cache.py [--folder FOLDER] [--data DATA_DIR]
                                [--batch-size N]   [--num-batches N]
"""


# ==========================================================================
#                                IMPORTS                      
# ==========================================================================
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch



# ==========================================================================
#                                 CONFIG                     
# ==========================================================================
# Which sub-folder inside ./models/ to process (e.g. "v1", "v2")
FOLDER = "v1"


DATA_DIR = "data/tensor_qv" # Where to find the calibration tensors

# How many Q-V samples to feed per calibration step.
# Larger batches give TensorRT more signal per step but use more GPU memory.
BATCH_SIZE = 32

# How many batches to run in total (BATCH_SIZE × NUM_BATCHES = total samples seen).
# 64 batches × 32 samples = 2 048 samples, which is enough for stable INT8 scales.
NUM_BATCHES = 64


# ==========================================================================
#                                   MAIN                       
# ==========================================================================

def _import_tensorrt():
    """Import TensorRT and exit with a clear message if it isn't installed."""
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        print(
            "ERROR: tensorrt Python package not found.\n"
            "       Install it on the Jetson or TensorRT host first.",
            file=sys.stderr,
        )
        sys.exit(1)



# DATA LOADING --------------------------------------------------------------

def load_calibration_data(data_dir: Path) -> np.ndarray:
    """
    Load every .pt file in data_dir and stack the "X" tensors into one array.

    Each file holds one battery cycle's worth of Q-V measurements.
    Shape of the returned array: (total_samples, 120)  — 120 voltage points.
    The rows are shuffled so that every calibration batch is a random mix of
    cycles rather than a block of consecutive ones.
    """
    pt_files = sorted(data_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    arrays = []
    for pt_file in pt_files:
        sample = torch.load(pt_file, map_location="cpu", weights_only=True)
        # Each file's "X" key is shape (num_cycles, 120)
        arrays.append(sample["X"].numpy().astype(np.float32))

    all_data = np.concatenate(arrays, axis=0)   # (total_samples, 120)
    np.random.shuffle(all_data)                 # mix cycles from different files

    print(f"  Loaded {all_data.shape[0]:,} samples from {len(pt_files)} files")
    return all_data



# TENSOR CALIBRATOR ---------------------------------------------------------

class QVCalibrator:
    """
    Feeds real Q-V data to TensorRT's INT8 entropy calibrator.

    TensorRT calls get_batch() repeatedly during the engine build to collect
    activation statistics.  Once it has seen enough data it calls
    write_calibration_cache() to persist the computed INT8 scale factors.
    On subsequent builds read_calibration_cache() returns the saved file so
    calibration can be skipped entirely.

    This class follows the IInt8EntropyCalibrator2 interface, which uses
    entropy minimisation to pick scale factors — generally the most accurate
    option for networks that process continuous sensor data like Q-V curves.
    """

    def __init__(
        self,
        data: np.ndarray,
        cache_path: Path,
        batch_size: int,
        num_batches: Optional[int],
    ):
        # pycuda is only imported here because it requires an active CUDA
        # context, which may not exist on the machine running data prep.
        import pycuda.driver as cuda

        self._cuda = cuda
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.data = data
        self.current_batch = 0

        # If the caller didn't specify a limit, use every sample exactly once.
        self.num_batches = num_batches if num_batches else (len(data) // batch_size)

        # Pinned (page-locked) host memory allows the GPU to DMA the data
        # directly without an extra copy — faster than regular numpy arrays.
        self._host_buffer = cuda.pagelocked_empty(
            (batch_size, data.shape[1]), dtype=np.float32
        )
        # Matching device-side allocation that TensorRT reads from.
        self._device_buffer = cuda.mem_alloc(self._host_buffer.nbytes)

    
    # Tensor Calibrating Inferface
    # |
    def get_batch_size(self) -> int:
        """Tell TensorRT how many samples are in each batch we provide."""
        return self.batch_size

    def get_batch(self, names: List[str]):
        """
        Return the next batch of calibration data as a GPU pointer.

        TensorRT calls this in a loop until we return None, signalling that
        all calibration data has been consumed.  'names' contains the input
        tensor names from the ONNX graph — we ignore them because we only
        have a single input.
        """
        start = self.current_batch * self.batch_size

        # Signal end-of-calibration when we've served all requested batches
        # or run out of data (whichever comes first).
        if self.current_batch >= self.num_batches or start >= len(self.data):
            return None

        end = min(start + self.batch_size, len(self.data))
        batch = self.data[start:end]

        # If the very last slice is smaller than batch_size, pad it by
        # repeating the first few rows.  TensorRT requires a fixed size.
        if len(batch) < self.batch_size:
            shortfall = self.batch_size - len(batch)
            batch = np.vstack([batch, batch[:shortfall]])

        # Copy from CPU → pinned host buffer → GPU device buffer
        np.copyto(self._host_buffer, batch)
        self._cuda.memcpy_htod(self._device_buffer, self._host_buffer)
        self.current_batch += 1

        # Return a list of device pointers, one per model input
        return [self._device_buffer]

    def read_calibration_cache(self):
        """
        Return the cached INT8 scales from a previous run, if they exist.

        TensorRT checks this first; if it gets data back it skips re-running
        calibration entirely, making subsequent engine builds much faster.
        """
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None  # No cache yet — TensorRT will run calibration from scratch

    def write_calibration_cache(self, cache: bytes) -> None:
        """Save the INT8 scale factors TensorRT just computed to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            f.write(cache)
        print(f"  Calibration cache saved -> {self.cache_path}")




# CACHE GENERATION ---------------------------------------------------------

def generate_cache(
    onnx_path: Path,
    cache_path: Path,
    data: np.ndarray,
    batch_size: int,
    num_batches: Optional[int],
) -> bool:
    """
    Run TensorRT calibration for a single ONNX model and write the .cache file.

    We build a full TensorRT engine here, but only to drive the calibration
    loop — the engine itself is thrown away.  The valuable output is the
    .cache file written by QVCalibrator.write_calibration_cache(), which
    onnx2engine.py will pick up when building the final optimised engine.

    Returns True on success, False if the ONNX couldn't be parsed or the
    engine build failed.
    """
    import pycuda.autoinit  # noqa: F401  — initialises the CUDA context once

    trt = _import_tensorrt()
    logger = trt.Logger(trt.Logger.WARNING)

    calibrator = QVCalibrator(data, cache_path, batch_size, num_batches)

    print(
        f"  Calibrating {onnx_path.name} "
        f"({calibrator.num_batches} batches * {batch_size} samples = "
        f"{calibrator.num_batches * batch_size:,} total) …"
    )

    # Build the network inside a set of context managers so TensorRT cleans
    # up its internal resources (builders, parsers, configs) automatically.
    # Note: backslash form used here for Python 3.8/3.9 compatibility (Jetson JetPack).
    with trt.Builder(logger) as builder, \
         builder.create_network(
             # EXPLICIT_BATCH mode is required for ONNX models; it lets
             # TensorRT reason about the batch dimension at build time.
             1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
         ) as network, \
         trt.OnnxParser(network, logger) as parser, \
         builder.create_builder_config() as config:
        # ── Step 1: Parse the ONNX graph ──────────────────────────────────
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  ONNX parse error: {parser.get_error(i)}", file=sys.stderr)
                return False

        # ── Step 2: Configure INT8 calibration ────────────────────────────
        # INT8 is the target precision.  FP16 is also enabled as a fallback
        # for any layers that TensorRT can't run in INT8 (e.g. some activations).
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.int8_calibrator = calibrator
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * (1 << 20))

        # ── Step 3: Build (this triggers the calibration loop) ────────────
        engine = builder.build_serialized_network(network, config)
        if engine is None:
            print(f"  ERROR: engine build failed for {onnx_path.name}", file=sys.stderr)
            return False

    return True




# ENTRY POINT ----------------------------------------------------------------

def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument(
        "--folder", default=FOLDER,
        help="Models sub-folder inside ./models/  (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--data", default=DATA_DIR, type=Path,
        help="Directory of .pt calibration tensors  (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--batch-size", default=BATCH_SIZE, type=int,
        help="Samples per calibration batch  (default: %(default)s)",
    )
    arg_parser.add_argument(
        "--num-batches", default=NUM_BATCHES, type=int,
        help="Number of calibration batches; 0 = use all available data  (default: %(default)s)",
    )
    args = arg_parser.parse_args()

    # Resolve paths relative to the repository root so the script works
    # regardless of which directory you launch it from.
    repo_root  = Path(__file__).resolve().parents[2]
    models_dir = repo_root / "models" / args.folder
    data_dir   = repo_root / args.data if not Path(args.data).is_absolute() else Path(args.data)

    if not models_dir.exists():
        print(f"ERROR: models directory not found: {models_dir}", file=sys.stderr)
        sys.exit(1)

    onnx_files = sorted(models_dir.glob("*/*.onnx"))
    if not onnx_files:
        print(f"ERROR: no .onnx files found under {models_dir}", file=sys.stderr)
        sys.exit(1)

    # Load all calibration tensors upfront — every model shares the same
    # Q-V input domain, so we only need to do this expensive step once.
    print(f"Loading calibration data from {data_dir} …")
    calibration_data = load_calibration_data(data_dir)
    print()

    # 0 on the CLI means "use everything"; None is the internal sentinel for that.
    num_batches = args.num_batches if args.num_batches > 0 else None

    results: Dict[str, bool] = {}

    for onnx_path in onnx_files:
        model_name = onnx_path.stem
        cache_path = onnx_path.parent / f"{model_name}.cache"

        print(f"{'─' * 60}")
        print(f"  Model : {model_name}")

        if cache_path.exists():
            print(f"  Cache already exists ({cache_path.name}) — skipping.")
            results[model_name] = True
            continue

        success = generate_cache(
            onnx_path, cache_path, calibration_data, args.batch_size, num_batches
        )
        results[model_name] = success
        print(f"  {'OK Cache written' if success else 'X FAILED'}\n")



    # Final summary: 
    print(f"{'═' * 60}")
    print("  Summary")
    print(f"{'─' * 60}")
    for model_name, success in results.items():
        print(f"  {model_name:<20}  {'OK' if success else 'X'}")
    print(f"{'═' * 60}")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
