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
#   ./tmux_experiments_entrypoint.sh [GPU_DEVICE] [EPOCHS]
#       GPU_DEVICE  CUDA device index             (default: 0)
#       EPOCHS      training epochs; supersedes run_all_atomic.sh's default (50)
#                   *and* the per-training defaults  (default: unset → use 50)
#
#   # e.g. run on GPU 1 for 120 epochs:
#   ./tmux_experiments_entrypoint.sh 1 120
#
#   # follow progress:
#   tmux attach -t insect_exps
#   # or tail the log printed at launch.
# ---------------------------------------------------------------------------

set -u

SESSION="insect_exps"
DEVICE="${1:-0}"
EPOCHS="${2:-}"              # optional epoch override; empty → use run_all_atomic.sh default
CONDA_ENV="aiprah5090"
MAX_OUTER_LOOPS=100          # safety cap on relaunches
OUTER_RETRY_WAIT=15          # seconds between relaunches

# ── Expected experiment counts per set (keep in sync with run_all_atomic.sh) ─
declare -A EXP_TOTALS=(
    [exp1]=90    # 3 datasets * 6 models * 5 folds
    [exp2]=120   # 3 datasets * 2 models * 4 img_sizes * 5 folds
    [exp3]=12    # 6 pairs * 2 models
    [exp4]=20    # 2 datasets * 2 models * 5 folds
    [exp5]=80    # 2 datasets * 2 models * 4 resolutions * 5 folds
)
EXP_ORDER=(exp1 exp2 exp3 exp4 exp5)
OUTPUT_DIR="runs/experiments"

# Print a per-set and global completion summary (counts done.txt markers).
print_progress() {
    local title="${1:-Progress}"
    local done_total=0 exp_total=0
    echo ""
    echo "========================================================================"
    echo "  $title"
    echo "========================================================================"
    printf "  %-6s %10s %8s %8s\n" "SET" "DONE/TOTAL" "REMAIN" "PERCENT"
    printf "  %-6s %10s %8s %8s\n" "------" "----------" "------" "-------"
    local exp total done remain
    for exp in "${EXP_ORDER[@]}"; do
        total=${EXP_TOTALS[$exp]}
        done=0
        for d in "${OUTPUT_DIR}/${exp}_"*/done.txt; do
            [[ -f "$d" ]] && ((done++))
        done
        remain=$(( total - done ))
        (( remain < 0 )) && remain=0
        printf "  %-6s %10s %8s %7.1f%%\n" \
            "$exp" "${done}/${total}" "$remain" \
            "$(awk "BEGIN{printf (${total}?100*${done}/${total}:0)}")"
        (( done_total += done ))
        (( exp_total += total ))
    done
    local global_remain=$(( exp_total - done_total ))
    (( global_remain < 0 )) && global_remain=0
    printf "  %-6s %10s %8s %8s\n" \
        "------" "----------" "------" "-------"
    printf "  %-6s %10s %8s %7.1f%%\n" \
        "ALL" "${done_total}/${exp_total}" "$global_remain" \
        "$(awk "BEGIN{printf (${exp_total}?100*${done_total}/${exp_total}:0)}")"
    echo "========================================================================"
    echo ""
}

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
    echo "Launching the full experiment pipeline in detached tmux session '$SESSION' (GPU $DEVICE, epochs ${EPOCHS:-default})…"
    tmux new-session -d -s "$SESSION" "_IN_PIPELINE=1 bash '$SCRIPT_PATH' '$DEVICE' '$EPOCHS'"
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
# Export EPOCHS so run_all_atomic.sh (and run_single_experiment.py via --epochs)
# pick it up; leaving it unset lets run_all_atomic.sh use its own default (50).
if [[ -n "$EPOCHS" ]]; then
    export EPOCHS
    echo "# Epochs:   $EPOCHS  (overriding run_all_atomic.sh + training defaults)"
else
    echo "# Epochs:   default (run_all_atomic.sh = 50)"
fi
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

# Show where we stand before kicking off any training.
print_progress "Experiment progress at launch"

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

    print_progress "Experiment progress after pass #$attempt"

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

print_progress "Final experiment progress"

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
