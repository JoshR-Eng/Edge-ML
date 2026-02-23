#!/usr/bin/env bash
# DESCRIPTION:
# quantise.sh — Run the full INT8/FP16 quantisation pipeline for one model folder.
#
# USAGE:
#   ./quantise.sh <folder>          e.g.  ./quantise.sh v1
#   source quantise.sh <folder>     same, but keeps env vars in your shell
#
# FUNCTION: 
#   1. calibration_cache.py  — profiles your Q-V data and writes per-model
#                              INT8 scale-factor caches to models/<folder>/
#   2. onnx2engine.py        — compiles every ONNX model into optimised
#                              FP16 and INT8 TensorRT engines
#
# NOTE:
# Both scripts will look for models under  models/<folder>/*/*.onnx


# Source guard: when sourced, 'exit' and 'set -e' act on the parent shell and
# kill the tmux window. Re-run in a subshell instead, then return cleanly.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    bash "${BASH_SOURCE[0]}" "$@"
    return
fi

set -euo pipefail  # exit on error



# Resolve Paths -------------------------------------------------------------
# REPO_ROOT is the directory this script lives in, so it works from any cwd.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTILS_DIR="$REPO_ROOT/src/utils"



# Validate Inputs -----------------------------------------------------------
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <folder>"
    echo "  e.g. $0 v1"
    exit 1
fi

FOLDER="$1"

if [[ ! -d "$REPO_ROOT/models/$FOLDER" ]]; then
    echo "ERROR: models/$FOLDER does not exist."
    exit 1
fi


# Activate venv if exists ---------------------------------------------------
if [ -d ".venv" ]; then
    echo -e "Activating virtual environment..."
    source .venv/bin/activate
fi

# Run Quantisation Scripts --------------------------------------------------
echo "============================================================"
echo "  Quantisation pipeline  →  models/$FOLDER"
echo "============================================================"
echo

echo "--- Step 1/2: Generating INT8 calibration caches -----------"
python3 "$UTILS_DIR/calibration_cache.py" --folder "$FOLDER"
echo

echo "--- Step 2/2: Compiling TensorRT engines -------------------"
python3 "$UTILS_DIR/onnx2engine.py" --folder "$FOLDER"
echo

echo "============================================================"
echo "  Done.  Engines written to models/$FOLDER/"
echo "============================================================"
