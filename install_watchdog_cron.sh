#!/bin/bash
#
# install_watchdog_cron.sh
# ---------------------------------------------------------------------------
# Installs (or removes) the cron entry that keeps experiment_watchdog.sh ticking
# every 5 minutes. The watchdog relaunches the tmux experiment pipeline whenever
# it has crashed and work still remains. See CRON.md for details.
#
# Usage:
#   ./install_watchdog_cron.sh [DEVICE] [EPOCHS]   # install / update the job
#   ./install_watchdog_cron.sh --remove            # uninstall the job
#   ./install_watchdog_cron.sh --status            # show current crontab entry
#
#   DEVICE  GPU index passed to the pipeline      (default: 0)
#   EPOCHS  epoch override for the pipeline        (default: unset → script default)
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG="${REPO_DIR}/experiment_watchdog.sh"
SCHEDULE="*/5 * * * *"          # every 5 minutes (also self-heals after reboot)
TAG="experiment_watchdog.sh"    # unique marker used to find/replace our line

# Always strip any pre-existing watchdog line so re-running is idempotent.
current_without_ours() { crontab -l 2>/dev/null | grep -vF "$TAG" || true; }

case "${1:-}" in
    --remove)
        current_without_ours | crontab -
        echo "[OK] Removed watchdog cron entry."
        crontab -l 2>/dev/null | grep -F "$TAG" >/dev/null \
            && echo "[WARN] An entry still matches '$TAG'." || echo "Done."
        exit 0
        ;;
    --status)
        echo "Current watchdog cron entry (if any):"
        crontab -l 2>/dev/null | grep -F "$TAG" || echo "  (none installed)"
        exit 0
        ;;
esac

DEVICE="${1:-0}"
EPOCHS="${2:-}"

if [[ ! -x "$WATCHDOG" ]]; then
    chmod +x "$WATCHDOG" 2>/dev/null || true
fi
if [[ ! -f "$WATCHDOG" ]]; then
    echo "ERROR: watchdog not found at $WATCHDOG" >&2
    exit 1
fi

CRON_LINE="${SCHEDULE} DEVICE=${DEVICE} EPOCHS=${EPOCHS} ${WATCHDOG}"

{ current_without_ours; echo "$CRON_LINE"; } | crontab -

echo "[OK] Installed watchdog cron entry:"
echo "     $CRON_LINE"
echo ""
echo "Verify:   crontab -l"
echo "Watch:    tail -f ${REPO_DIR}/runs/experiments/watchdog.log"
echo "Remove:   ${BASH_SOURCE[0]} --remove"
echo ""
echo "Note: the first relaunch happens on the next 5-minute cron tick."
echo "      To start immediately, run:  bash ${WATCHDOG}"
