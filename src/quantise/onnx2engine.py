#!/usr/bin/env python3
"""
Compile ONNX models into FP16 and INT8 TensorRT engines.
Called by quantise.sh

Output per model:
    <name>_fp16.engine   16-bit float (half the memory of FP32, near-identical accuracy)
    <name>_int8.engine   8-bit integer (fastest; uses the .cache from calibration_cache.py)
"""
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


# Config ----------------------------------------------------------------------

TRTEXEC      = "/usr/src/tensorrt/bin/trtexec"
WORKSPACE_MB = 4096  # GPU memory budget for TRT's optimiser (Orin Nano has 8 GB)

# Flags applied to every build, tuned for the Jetson Orin Nano (Ampere/SM87)
BASE_FLAGS = [
    f"--memPoolSize=workspace:{WORKSPACE_MB}MiB",
    "--tacticSources=+CUBLAS,+CUBLAS_LT,+CUDNN",  # all fast-math backends
    "--noDataTransfers",                            # measure compute time only
    "--preview=+fasterDynamicShapes0805",           # TRT 8.5+ dynamic shape opt
]


# Engine Building ----------------------------------------------------------------

def _run(cmd: List[str]) -> bool:
    print(" $", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


def build_fp32_engine(onnx_path: Path, output_path: Path) -> bool:
    # FP32 is TRT's default precision — no flags needed.
    # Useful as a latency/accuracy baseline to compare against FP16 and INT8.
    return _run([TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={output_path}",
                 *BASE_FLAGS])


def build_fp16_engine(onnx_path: Path, output_path: Path) -> bool:
    return _run([TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={output_path}",
                 "--fp16", *BASE_FLAGS])


def build_int8_engine(onnx_path: Path, output_path: Path,
                      calib_cache: Optional[Path]) -> bool:
    # FP16 is included as a fallback for layers TRT can't run in INT8
    # (e.g. LSTM Recurrence nodes — those fall back silently, no accuracy loss).
    flags = ["--int8", "--fp16", *BASE_FLAGS]
    if calib_cache:
        flags.append(f"--calib={calib_cache}")
    else:
        print("  WARNING: no calibration cache — INT8 accuracy may be poor")
    return _run([TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={output_path}", *flags])


# Script Entry Point ------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python onnx2engine.py <folder>", file=sys.stderr)
        sys.exit(1)

    folder     = sys.argv[1]
    repo_root  = Path(__file__).resolve().parents[2]
    models_dir = repo_root / "models" / folder

    if not models_dir.exists():
        print(f"ERROR: models/{folder} does not exist", file=sys.stderr)
        sys.exit(1)

    onnx_files = sorted(models_dir.glob("*/*.onnx"))
    if not onnx_files:
        print(f"ERROR: no .onnx files found under {models_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(onnx_files)} model(s) in models/{folder}\n")

    results: Dict[str, Dict[str, bool]] = {}

    for onnx_path in onnx_files:
        name       = onnx_path.stem
        output_dir = onnx_path.parent
        cache      = output_dir / f"{name}.cache"

        print(f"{'─' * 60}")
        print(f"  Model: {name}\n")

        fp32_ok = build_fp32_engine(onnx_path, output_dir / f"{name}_fp32.engine")
        print(f"  [FP32] {'OK' if fp32_ok else 'FAILED'}\n")

        fp16_ok = build_fp16_engine(onnx_path, output_dir / f"{name}_fp16.engine")
        print(f"  [FP16] {'OK' if fp16_ok else 'FAILED'}\n")

        int8_ok = build_int8_engine(onnx_path, output_dir / f"{name}_int8.engine",
                                    cache if cache.exists() else None)
        print(f"  [INT8] {'OK' if int8_ok else 'FAILED'}\n")

        results[name] = {"fp32": fp32_ok, "fp16": fp16_ok, "int8": int8_ok}

    print(f"{'=' * 60}")
    print("  Summary")
    print(f"{'-' * 60}")
    all_ok = True
    for name, res in results.items():
        print(f"  {name:<20}  FP32 {'OK' if res['fp32'] else 'FAIL'}   "
              f"FP16 {'OK' if res['fp16'] else 'FAIL'}   "
              f"INT8 {'OK' if res['int8'] else 'FAIL'}")
        all_ok = all_ok and all(res.values())
    print(f"{'=' * 60}")

    sys.exit(0 if all_ok else 1)
