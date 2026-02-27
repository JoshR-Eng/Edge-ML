#!/usr/bin/env python3
"""
Compile ONNX models into FP32, FP16, and INT8 TensorRT engines.
Called by quantise.sh

Output per model (written to models/<folder>/<name>/):
    <name>_fp32.engine          FP32 baseline
    <name>_fp16.engine          FP16 — half the memory of FP32, near-identical accuracy
    <name>_int8.engine          INT8 — fastest; uses .cache from calibration_cache.py

    logs/quantisation_fp32.log          full trtexec build output
    logs/quantisation_fp32_layers.json  per-layer precision, memory, tactic info
    (same for fp16 and int8)
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
]


# Engine building -------------------------------------------------------------

def _run(cmd: List[str], log_path: Path) -> bool:
    """
    Run a trtexec command, streaming output live to the terminal while also
    writing it to log_path so you can review it later.

    stdout and stderr are merged into one stream — TRT writes everything to
    stderr, so capturing them separately would leave the log mostly empty.
    """
    print(" $", " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_file:
        log_file.write(" ".join(cmd) + "\n\n")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout
            text=True,
        )
        for line in proc.stdout:
            print(line, end="")       # live terminal output
            log_file.write(line)      # saved to file
        proc.wait()

    return proc.returncode == 0


def build_fp32_engine(onnx_path: Path, output_path: Path, log_dir: Path) -> bool:
    # FP32 is TRT's default precision — no flags needed.
    # Useful as a latency/accuracy baseline to compare against FP16 and INT8.
    return _run(
        [TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={output_path}",
         f"--exportLayerInfo={log_dir / 'quantisation_fp32_layers.json'}",
         *BASE_FLAGS],
        log_dir / "quantisation_fp32.log",
    )


def build_fp16_engine(onnx_path: Path, output_path: Path, log_dir: Path) -> bool:
    return _run(
        [TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={output_path}",
         "--fp16",
         f"--exportLayerInfo={log_dir / 'quantisation_fp16_layers.json'}",
         *BASE_FLAGS],
        log_dir / "quantisation_fp16.log",
    )


def build_int8_engine(onnx_path: Path, output_path: Path, log_dir: Path,
                      calib_cache: Optional[Path]) -> bool:
    # FP16 is included as a fallback for layers TRT can't run in INT8
    # (e.g. LSTM Recurrence nodes — those fall back silently, no accuracy loss).
    flags = ["--int8", "--fp16",
             f"--exportLayerInfo={log_dir / 'quantisation_int8_layers.json'}",
             *BASE_FLAGS]
    if calib_cache:
        flags.append(f"--calib={calib_cache}")
    else:
        print("  WARNING: no calibration cache — INT8 accuracy may be poor")
    return _run(
        [TRTEXEC, f"--onnx={onnx_path}", f"--saveEngine={output_path}", *flags],
        log_dir / "quantisation_int8.log",
    )


# Script entry point ----------------------------------------------------------

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

    onnx_files = sorted(models_dir.glob("*/*/*.onnx"))
    if not onnx_files:
        print(f"ERROR: no .onnx files found under {models_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(onnx_files)} model(s) in models/{folder}\n")

    results: Dict[str, Dict[str, bool]] = {}

    for onnx_path in onnx_files:
        name       = onnx_path.stem
        output_dir = onnx_path.parent
        log_dir    = output_dir / "logs"
        cache      = output_dir / f"{name}.cache"

        print(f"{'─' * 60}")
        print(f"  Model: {name}\n")

        fp32_ok = build_fp32_engine(onnx_path, output_dir / f"{name}_fp32.engine", log_dir)
        print(f"  [FP32] {'OK' if fp32_ok else 'FAILED'}\n")

        fp16_ok = build_fp16_engine(onnx_path, output_dir / f"{name}_fp16.engine", log_dir)
        print(f"  [FP16] {'OK' if fp16_ok else 'FAILED'}\n")

        int8_ok = build_int8_engine(onnx_path, output_dir / f"{name}_int8.engine", log_dir,
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
