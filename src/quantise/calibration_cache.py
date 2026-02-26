#!/usr/bin/env python3
"""
Generate INT8 calibration caches for each model in models/<folder>/*.onnx.
Called by quantise.sh — not intended to be run directly.

Output: models/<folder>/<ModelName>/<ModelName>.cache
"""
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml


#  Config ----------------------------------------------------------------------

DATA_DIR      = "data/tensor_qv"
CALIB_SAMPLES = 2048  # total Q-V samples to use for INT8 calibration
              # calls = ceil(CALIB_SAMPLES / batch_size), so bs1 and bs32
              # both calibrate on the same number of samples


# Data Loading -----------------------------------------------------------------

def load_train_cells(repo_root: Path) -> List[str]:
    """
    Return the training cell IDs from configs.yaml.

    Only training cells are used for calibration — using validation or test
    cells would leak evaluation data into the quantisation process.
    """
    with open(repo_root / "configs.yaml") as f:
        splits = yaml.safe_load(f)["splits"]
    return splits["train"]


def load_calibration_data(data_dir: Path, cell_ids: List[str]) -> np.ndarray:
    """
    Load .pt files for the given cell IDs and stack their X tensors into one array.

    Files are named like '01_Rd_3C.pt' — matched by the numeric prefix so the
    protocol suffix doesn't matter.

    Returns shape (total_samples, 120), shuffled so each calibration batch
    is a random mix of cells and cycles rather than a block of one cell.
    """
    all_files = sorted(data_dir.glob("*.pt"))

    matched = []
    for cell_id in cell_ids:
        match = next((f for f in all_files if f.stem.startswith(f"{cell_id}_")), None)
        if match:
            matched.append(match)

    if not matched:
        raise RuntimeError(f"No .pt files matched cell IDs in {data_dir}")

    arrays = []
    for pt_file in matched:
        sample = torch.load(pt_file, map_location="cpu", weights_only=True)
        arrays.append(sample["X"].numpy().astype(np.float32))  # (num_cycles, 120)

    data = np.concatenate(arrays, axis=0)
    np.random.shuffle(data)

    print(f"  Loaded {data.shape[0]:,} samples from {len(matched)} cells")
    return data


# TensorRT Calibrator Class --------------------------------------------------

def _import_tensorrt():
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        print("ERROR: tensorrt not found. Install it on the Jetson first.", file=sys.stderr)
        sys.exit(1)


def _make_calibrator_class(trt):
    """
    Build QVCalibrator as a runtime subclass of trt.IInt8EntropyCalibrator2.

    The class must genuinely inherit from TensorRT's base — duck typing isn't
    accepted. We can't do this at module level because trt is only available
    on the Jetson, so we build the class here after importing trt.
    """

    class QVCalibrator(trt.IInt8EntropyCalibrator2):

        def __init__(self, data: np.ndarray, cache_path: Path,
                     batch_size: int, num_batches: Optional[int]):
            trt.IInt8EntropyCalibrator2.__init__(self)

            # pycuda requires an active CUDA context — imported here, not at
            # module level, so the script can load on non-CUDA machines.
            import pycuda.driver as cuda

            self._cuda        = cuda
            self.cache_path   = cache_path
            self.batch_size   = batch_size
            self.data         = data
            self.current_batch = 0
            self.num_batches  = num_batches or (len(data) // batch_size)

            # Page-locked host memory lets the GPU DMA data directly, avoiding
            # an extra copy versus a regular numpy array.
            self._host_buf = cuda.pagelocked_empty((batch_size, data.shape[1]),
                                                   dtype=np.float32)
            self._dev_buf  = cuda.mem_alloc(self._host_buf.nbytes)

        def get_batch_size(self) -> int:
            return self.batch_size

        def get_batch(self, names: List[str]):
            # TensorRT calls this in a loop; returning None signals end of data.
            start = self.current_batch * self.batch_size
            if self.current_batch >= self.num_batches or start >= len(self.data):
                return None

            end   = min(start + self.batch_size, len(self.data))
            batch = self.data[start:end]

            # Pad the last (possibly short) batch by repeating rows.
            if len(batch) < self.batch_size:
                shortfall = self.batch_size - len(batch)
                batch = np.vstack([batch, batch[:shortfall]])

            np.copyto(self._host_buf, batch)
            self._cuda.memcpy_htod(self._dev_buf, self._host_buf)
            self.current_batch += 1
            return [self._dev_buf]

        def read_calibration_cache(self):
            # If a cache already exists TensorRT skips re-calibrating.
            if self.cache_path.exists():
                with open(self.cache_path, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                f.write(cache)
            print(f"  Cache saved -> {self.cache_path.name}")

    return QVCalibrator


# Cache Generation ----------------------------------------------------------------

def generate_cache(onnx_path: Path, cache_path: Path,
                   data: np.ndarray,
                   calib_samples: int) -> bool:
    """
    Run TensorRT INT8 calibration for one ONNX model and write the .cache file.

    The calibration batch size is read directly from the ONNX input shape so
    it always matches the network — bs1 models get batch_size=1, bs32 models
    get batch_size=32.  The number of calibration calls is computed from
    calib_samples so both scenarios use the same total number of samples.

    A full TRT engine is built here only to drive the calibration loop; the
    engine itself is discarded.
    """
    import pycuda.autoinit  # noqa: F401 — initialises the CUDA context

    trt        = _import_tensorrt()
    logger     = trt.Logger(trt.Logger.WARNING)
    Calibrator = _make_calibrator_class(trt)

    with trt.Builder(logger) as builder, \
         builder.create_network(
             1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
         ) as network, \
         trt.OnnxParser(network, logger) as parser, \
         builder.create_builder_config() as config:

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  ONNX error: {parser.get_error(i)}", file=sys.stderr)
                return False

        # Read the batch dimension from the parsed network so the calibrator
        # always feeds the right number of samples per step:
        #   bs1  → batch_size=1,  num_calls=2048 → 2,048 samples total
        #   bs32 → batch_size=32, num_calls=64   → 2,048 samples total
        batch_size = network.get_input(0).shape[0]
        num_calls  = (calib_samples + batch_size - 1) // batch_size
        calibrator = Calibrator(data, cache_path, batch_size, num_calls)

        print(f"  Calibrating {onnx_path.name} "
              f"({num_calls} calls × {batch_size} = "
              f"{num_calls * batch_size:,} samples) ...")

        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # fallback for layers that can't run INT8
        config.int8_calibrator = calibrator
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * (1 << 20))

        engine = builder.build_serialized_network(network, config)
        if engine is None:
            print(f"  ERROR: build failed for {onnx_path.name}", file=sys.stderr)
            return False

    return True




# Script Entry Point ------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python calibration_cache.py <folder>", file=sys.stderr)
        sys.exit(1)

    folder    = sys.argv[1]
    repo_root = Path(__file__).resolve().parents[2]
    models_dir = repo_root / "models" / folder
    data_dir   = repo_root / DATA_DIR

    if not models_dir.exists():
        print(f"ERROR: models/{folder} does not exist", file=sys.stderr)
        sys.exit(1)

    onnx_files = sorted(models_dir.glob("*/*.onnx"))
    if not onnx_files:
        print(f"ERROR: no .onnx files found under {models_dir}", file=sys.stderr)
        sys.exit(1)

    cell_ids = load_train_cells(repo_root)
    print(f"Using train split ({len(cell_ids)} cells)")
    print(f"Loading calibration data from {data_dir} ...")
    calibration_data = load_calibration_data(data_dir, cell_ids)
    print()

    results: Dict[str, bool] = {}

    for onnx_path in onnx_files:
        model_name = onnx_path.stem
        cache_path = onnx_path.parent / f"{model_name}.cache"

        print(f"{'─' * 60}")
        print(f"  Model: {model_name}")

        if cache_path.exists():
            print(f"  Cache already exists, skipping.")
            results[model_name] = True
            continue

        results[model_name] = generate_cache(
            onnx_path, cache_path, calibration_data, CALIB_SAMPLES
        )
        print(f"  {'OK' if results[model_name] else 'FAILED'}\n")

    print(f"{'=' * 60}")
    print("  Summary")
    print(f"{'-' * 60}")
    for name, ok in results.items():
        print(f"  {name:<20}  {'OK' if ok else 'FAILED'}")
    print(f"{'=' * 60}")

    sys.exit(0 if all(results.values()) else 1)
