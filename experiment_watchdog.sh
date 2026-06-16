#!/bin/bash
#
# experiment_watchdog.sh
# ---------------------------------------------------------------------------
# Persistent, tmux-independent guardian for the S. titanus experiment pipeline.
#
# Run it periodically from cron (see install instructions at the bottom). On
# each tick it:
#
#   1. Counts completed experiments (done.txt markers) vs. the expected total.
#      → If everything is finished, it does nothing and exits.
#   2. Checks whether the tmux session is alive.
#      → If alive, it does nothing (the pipeline is still working).
#      → If dead/missing AND work remains, it relaunches the tmux pipeline.
#
# Because tmux_experiments_entrypoint.sh self-guards (it refuses to start a
# second session and skips already-completed work via done.txt), running this
# watchdog on a timer is safe and idempotent: it only ever resurrects a dead run.
#
# It is intentionally NOT dependent on tmux for its own survival — cron keeps it
# alive, so even if the whole tmux session crashes, the next tick brings it back.
#
# Usage (manual test):    ./experiment_watchdog.sh
# Usage (cron):           see "CRON SETUP" near the bottom of this file.
# ---------------------------------------------------------------------------

set -u

SESSION="insect_exps"
DEVICE="${DEVICE:-0}"          # GPU index; override via cron env if needed
EPOCHS="${EPOCHS:-}"          # optional epoch override, passed straight through
OUTPUT_DIR="runs/experiments"

# Expected experiment count — keep in sync with run_all_atomic.sh:
#   exp1=90  exp2=120  exp3=12  exp4=20  exp5=80  →  322
TOTAL_EXPECTED=322

# Resolve the repo dir from this script's location so cron can run it from /.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR" || { echo "FATAL: cannot cd to $REPO_DIR"; exit 1; }

ENTRYPOINT="${REPO_DIR}/tmux_experiments_entrypoint.sh"
WATCHDOG_LOG="${OUTPUT_DIR}/watchdog.log"
mkdir -p "$OUTPUT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$WATCHDOG_LOG"; }

# ── Make conda + tmux reachable in cron's minimal environment ───────────────
# cron does not inherit your interactive PATH, so the tmux child (which runs the
# entrypoint and calls `conda`) would otherwise fail to find conda. Put the
# conda + system bins on PATH; the detached tmux session inherits this env.
CONDA_BASE="/home/aiprah/miniconda3"
export PATH="${CONDA_BASE}/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

# ── 1. Are we already finished? ─────────────────────────────────────────────
done_count=0
for d in "${OUTPUT_DIR}"/exp{1,2,3,4,5}_*/done.txt; do
    [[ -f "$d" ]] && ((done_count++))
done

if (( done_count >= TOTAL_EXPECTED )); then
    # Nothing to do. Stay quiet to avoid spamming the log every tick.
    exit 0
fi

# ── 2. Is the pipeline still alive in tmux? ─────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    # Healthy and running — let it keep working.
    exit 0
fi

# ── 3. Session is gone and work remains → resurrect it. ─────────────────────
log "Session '$SESSION' is DOWN with $done_count/$TOTAL_EXPECTED experiments done. Relaunching pipeline (GPU $DEVICE, epochs ${EPOCHS:-default})."

# The entrypoint detaches its own tmux session and returns immediately.
DEVICE="$DEVICE" bash "$ENTRYPOINT" "$DEVICE" "$EPOCHS" >> "$WATCHDOG_LOG" 2>&1
log "Relaunch command issued (entrypoint exit=$?)."

exit 0

# ===========================================================================
# CRON SETUP
# ---------------------------------------------------------------------------
# Check every 5 minutes (also covers reboots — cron is started by the OS):
#
#   crontab -e
#   # then add this line (adjust GPU via DEVICE=, epochs via EPOCHS=):
#   */5 * * * * DEVICE=0 EPOCHS= /home/aiprah/github/insect_traps/experiment_watchdog.sh
#
# Or install non-interactively:
#
#   ( crontab -l 2>/dev/null | grep -v experiment_watchdog.sh;
#     echo '*/5 * * * * DEVICE=0 EPOCHS= /home/aiprah/github/insect_traps/experiment_watchdog.sh'
#   ) | crontab -
#
# Verify:   crontab -l
# Watch:    tail -f /home/aiprah/github/insect_traps/runs/experiments/watchdog.log
# Remove:   crontab -l | grep -v experiment_watchdog.sh | crontab -
# ===========================================================================
