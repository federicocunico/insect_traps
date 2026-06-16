#!/bin/bash
#
# tmux_experiments_entrypoint.sh
# ---------------------------------------------------------------------------
# One-shot entrypoint that runs the *entire* S. titanus experiment pipeline,
# designed to survive an unstable machine. It:
#
#   1. (Re)launches itself inside a detached tmux session so the run keeps
#      going after you disconnect.
#   2. Activates the conda env (aiprah5090).
#   3. Backfills done.txt markers from the cache (adds P/R/F1, no retraining).
#   4. Runs the full atomic experiment suite, relaunching it in an outer loop
#      until every experiment is completed (crash-safe; finished work is skipped
#      via its done.txt marker, so no computation is ever wasted).
#   5. Regenerates the result tables (experiment_results.{txt,tex}).
#
# Usage:
#   ./tmux_experiments_entrypoint.sh [GPU_DEVICE]      # default GPU 0
#
#   # follow progress:
#   tmux attach -t insect_exps
#   # or tail the log printed at launch.
# ---------------------------------------------------------------------------

set -u

SESSION="insect_exps"
DEVICE="${1:-0}"
CONDA_ENV="aiprah5090"
MAX_OUTER_LOOPS=100          # safety cap on relaunches
OUTER_RETRY_WAIT=15          # seconds between relaunches

# Resolve absolute paths so tmux re-launch and conda work from anywhere.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${REPO_DIR}/$(basename "${BASH_SOURCE[0]}")"
cd "$REPO_DIR"

# ── Phase 0: self-launch into a detached tmux session ───────────────────────
# Skipped when already inside tmux, or when re-entered via the _IN_PIPELINE flag.
if [[ -z "${TMUX:-}" && "${_IN_PIPELINE:-}" != "1" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux is not installed. Either install tmux, or run the pipeline"
        echo "       directly with:  _IN_PIPELINE=1 bash $SCRIPT_PATH $DEVICE"
        exit 1
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "A tmux session '$SESSION' is already running."
        echo "Attach with:  tmux attach -t $SESSION"
        exit 0
    fi
    echo "Launching the full experiment pipeline in detached tmux session '$SESSION' (GPU $DEVICE)…"
    tmux new-session -d -s "$SESSION" "_IN_PIPELINE=1 bash '$SCRIPT_PATH' '$DEVICE'"
    echo "Started. Follow it with:   tmux attach -t $SESSION"
    echo "Logs are written under:    $REPO_DIR/runs/experiments/pipeline_*.log"
    exit 0
fi

# ── Inside the session: set up logging ──────────────────────────────────────
mkdir -p runs/experiments
LOG_FILE="runs/experiments/pipeline_$(date +%Y%m%d_%H%M%S).log"
# Mirror all stdout/stderr to the log file as well as the tmux pane.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "############################################################################"
echo "# Insect-trap experiment pipeline"
echo "# Started:  $(date)"
echo "# Repo:     $REPO_DIR"
echo "# GPU:      $DEVICE"
echo "# Log:      $LOG_FILE"
echo "############################################################################"

# ── Phase 1: activate conda ─────────────────────────────────────────────────
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [[ -z "$CONDA_BASE" || ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "ERROR: could not locate conda. Is it installed and on PATH?"
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "ERROR: failed to activate env '$CONDA_ENV'"; exit 1; }
echo "[OK] Activated conda env: $CONDA_ENV ($(python --version 2>&1))"

# ── Phase 2: backfill done.txt from the experiment cache (no training) ───────
echo ""
echo "=== Phase 2/4: backfilling done.txt markers with full metrics (P/R/F1) ==="
python backfill_done_files.py

# ── Phase 3: run the atomic suite, relaunching until it completes cleanly ────
echo ""
echo "=== Phase 3/4: running the atomic experiment suite (crash-safe loop) ==="
ERROR_LOG="runs/experiments/error.txt"
attempt=1
suite_ok=0
while (( attempt <= MAX_OUTER_LOOPS )); do
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Atomic suite — pass #$attempt  ($(date))"
    echo "------------------------------------------------------------------------"
    # Truncate the error log so run_all_atomic.sh's exit code reflects only this
    # pass (it appends to and tests this file at the end).
    : > "$ERROR_LOG"

    bash run_all_atomic.sh "$DEVICE"
    rc=$?

    if [[ $rc -eq 0 ]]; then
        echo "[OK] Atomic suite completed with no failures on pass #$attempt."
        suite_ok=1
        break
    fi

    echo "[WARN] Atomic suite returned $rc (some experiments still failing/incomplete)."
    echo "       Relaunching in ${OUTER_RETRY_WAIT}s — already-completed work is skipped."
    sleep "$OUTER_RETRY_WAIT"
    ((attempt++))
done

if [[ $suite_ok -ne 1 ]]; then
    echo "[ERROR] Atomic suite did not fully complete after $MAX_OUTER_LOOPS passes."
    echo "        Check $ERROR_LOG for the experiments that keep failing."
fi

# ── Phase 4: regenerate the result tables ───────────────────────────────────
echo ""
echo "=== Phase 4/4: generating result tables ==="
python generate_results_tables.py

echo ""
echo "############################################################################"
echo "# PIPELINE FINISHED: $(date)"
echo "# Tables: experiment_results.txt  /  experiment_results.tex"
echo "# Full log: $LOG_FILE"
echo "############################################################################"

# Keep the tmux pane open so the final summary stays visible after completion.
echo ""
echo "(Pipeline done — press Ctrl-C or 'tmux kill-session -t $SESSION' to close.)"
exec "${SHELL:-bash}"
