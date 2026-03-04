#!/bin/bash
#
# DESCRIPTION:
#   Locks Jetson hardware state, activates the virtual environment,
#   and runs the benchmark suite for a given model folder.
#
# USAGE:
#   ./run_benchmark.sh <model_folder> [power_mode]
#
#   model_folder : subfolder under models/ to benchmark  (e.g. v4)
#   power_mode   : nvpmodel mode number  (default: 0)
#
#   Jetson Orin Nano 4GB power modes:
#     0 = MAXN   - maximum performance  (~10 W, all cores unlocked)
#     1 = 7W_AI  - efficiency mode      (~7 W,  CPU capped)
#     2 = 7W_CPU - CPU efficiency mode  (~7 W,  CPU & GPU capped)
#
# EXAMPLES:
#   ./run_benchmark.sh v4
#   ./run_benchmark.sh v4 1

# Prevent exit on source; re-run as subshell if sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    bash "${BASH_SOURCE[0]}" "$@"; return
fi

set -e


# --- Arguments ---------------------------------------------------------------

if [ -z "$1" ]; then
    echo "Usage: ./run_benchmark.sh <model_folder> [power_mode]"
    echo "  model_folder : e.g. v4"
    echo "  power_mode   : 0 = MAXN (default), 1 = 7W"
    exit 1
fi

FOLDER="$1"
POWER_MODE="${2:-0}"
RUN_NAME="${FOLDER}_powm${POWER_MODE}"

if ! [[ "$POWER_MODE" =~ ^[0-9]+$ ]]; then
    echo "Error: power_mode must be a number (0 = MAXN, 1 = 7W)"
    exit 1
fi

echo "============================================================"
echo "  Jetson Benchmark  ->  models/$FOLDER  (results: $RUN_NAME)"
echo "  Power mode        ->  nvpmodel -m $POWER_MODE"
echo "============================================================"


# --- Environment setup -------------------------------------------------------

if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# --- Hardware state ----------------------------------------------------------

echo "Locking hardware state (requires sudo)..."
sudo -v
sudo nvpmodel -m "$POWER_MODE"
sudo jetson_clocks           # locks clocks + sets fan to max within power mode
echo "Hardware state locked."
echo ""


# --- Run benchmark -----------------------------------------------------------

# Export Bash variables so they can pass to sudo
export BENCHMARK_FOLDER="$FOLDER" 
export BENCHMARK_RUN_NAME="$RUN_NAME" 

# Find exact python intepreter inside .venv
VENV_PYTHON=$(which python3)

# Run sudo -E flag to pass env variables to sudo user
sudo -E "$VENV_PYTHON" benchmark.py


# --- Done --------------------------------------------------------------------

echo ""
echo "============================================================"
echo "  Complete!  Results -> results/$RUN_NAME/"
echo "============================================================"
