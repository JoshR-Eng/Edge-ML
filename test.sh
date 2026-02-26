#!/bin/bash
#
# DESCRIPTION:
#   Sets up the Jetson hardware state, launches benchmark.py inside a detached
#   tmux session, sends a Discord notification on completion, then closes the
#   SSH connection so it doesn't influence measurements.
#
# USAGE:
#   ./test.sh <model_folder> [power_mode]
#
#   model_folder : subfolder under models/ to benchmark  (e.g. v2, v2_b32)
#   power_mode   : nvpmodel mode number  (default: 0)
#
#   Jetson Orin Nano 4GB power modes:
#     0 = MAXN  — maximum performance  (~10 W, all cores unlocked)
#     1 = 7W    — efficiency mode      (~7 W,  CPU/GPU capped)
#
# EXAMPLES:
#   ./test.sh v2          # MAXN mode  (single-cell engines)
#   ./test.sh v2 1        # 7W mode
#   ./test.sh v2_b32 0    # MAXN mode  (32-cell pack engines)

# Source guard: if this file is sourced instead of executed, re-run it as a
# subshell so that 'set -e' can never exit the user's interactive SSH session.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    bash "${BASH_SOURCE[0]}" "$@"; return
fi

set -e


# --- Arguments ---------------------------------------------------------------

if [ -z "$1" ]; then
    echo "Usage: ./test.sh <model_folder> [power_mode]"
    echo "  model_folder : e.g. v2, v2_b32"
    echo "  power_mode   : 0 = MAXN (default), 1 = 7W"
    exit 1
fi

MODELS="$1"
POWER_MODE="${2:-0}"

if ! [[ "$POWER_MODE" =~ ^[0-9]+$ ]]; then
    echo "Error: power_mode must be a number (0 = MAXN, 1 = 7W)"
    exit 1
fi

echo "============================================================"
echo "  Jetson Benchmarking  ->  models/$MODELS"
echo "  Power mode           ->  nvpmodel -m $POWER_MODE"
echo "============================================================"
echo ""


# --- Environment setup -------------------------------------------------------

if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Environment variables loaded from .env"
else
    echo "Warning: .env not found — Discord notifications will not be sent."
fi


# --- Hardware state ----------------------------------------------------------

echo "Requesting sudo to lock hardware state..."
sudo -v   # pre-authenticate so subsequent sudo calls don't prompt mid-script

# Set power envelope first, then lock clocks.
# jetson_clocks fixes CPU/GPU/memory to their max frequency *within* the
# chosen power mode, which removes governor-induced variance between samples.
echo "  Setting power mode: nvpmodel -m $POWER_MODE"
sudo nvpmodel -m "$POWER_MODE"

echo "  Locking clocks: jetson_clocks"
sudo jetson_clocks

echo "Hardware state locked."
echo ""


# --- tmux session ------------------------------------------------------------

SESSION_NAME="benchmarking_session"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Reusing tmux session: $SESSION_NAME"
else
    echo "Creating tmux session: $SESSION_NAME"
    # -e passes the current environment (including DISCORD_WEBHOOK_URL) into
    # the new session so notify.py can read it via os.getenv.
    tmux new-session -d -s "$SESSION_NAME" -e "DISCORD_WEBHOOK_URL=$DISCORD_WEBHOOK_URL"
fi

# Run benchmark then notify via src/utils/notify.py on success or failure.
tmux send-keys -t "$SESSION_NAME" "
python3 benchmark.py \
    --models models/$MODELS \
    --data   data/tensor_qv \
    --output results/$MODELS
" C-m


# --- Disconnect --------------------------------------------------------------

echo "============================================================"
echo "  Setup complete. Benchmark is running in tmux."
echo ""
echo "  Reconnect  :  tmux attach -t $SESSION_NAME"
echo "  Results    :  results/$MODELS/hardware_benchmark_raw.csv"
echo "  Power mode :  nvpmodel -m $POWER_MODE"
echo ""
echo "  Discord notification will be sent on completion."
echo "  You can safely disconnect now (Ctrl+D or close the terminal)."
echo "============================================================"

