# Experiment Watchdog (cron)

Keeps the S. titanus experiment pipeline alive **without** relying on tmux to
stay up. A cron job runs `experiment_watchdog.sh` every 5 minutes; if the tmux
session has crashed and experiments remain, it relaunches it. Cron is started by
the OS, so this survives logout **and** reboot.

## Pieces

| File | Role |
|------|------|
| `tmux_experiments_entrypoint.sh` | Runs the full pipeline inside a detached tmux session (`insect_exps`). Self-guards against double-launch. |
| `experiment_watchdog.sh` | Per-tick check: finished? → exit. session alive? → exit. else → relaunch. |
| `install_watchdog_cron.sh` | Installs / removes the cron entry. |

## Install

```bash
./install_watchdog_cron.sh [DEVICE] [EPOCHS]   # e.g. ./install_watchdog_cron.sh 0 120
```

- `DEVICE` — GPU index (default `0`)
- `EPOCHS` — epoch override (default unset → pipeline default of 50)

The cron line installed:

```
*/5 * * * * DEVICE=0 EPOCHS= /home/aiprah/github/insect_traps/experiment_watchdog.sh
```

First relaunch happens on the next 5-min tick. To start **now** without waiting:

```bash
bash experiment_watchdog.sh
```

## How it decides (each tick)

1. Count `done.txt` markers vs. `322` total → if complete, do nothing.
2. `tmux has-session insect_exps`? → if yes, leave it running.
3. Otherwise (down + work left) → relaunch via the entrypoint.

Idempotent and safe to run on a timer: finished work is skipped via `done.txt`,
and the entrypoint refuses to start a second session.

## Operate

```bash
crontab -l                                  # show all cron jobs
./install_watchdog_cron.sh --status         # show just this job
tail -f runs/experiments/watchdog.log       # watch relaunch activity
tmux attach -t insect_exps                  # attach to the live pipeline
./install_watchdog_cron.sh --remove         # uninstall
```

## Notes

- Restarts a **crashed/dead** session. It does not detect an alive-but-hung
  session — add a heartbeat check if hangs are the failure mode.
- `TOTAL_EXPECTED=322` in `experiment_watchdog.sh` mirrors the loop counts in
  `run_all_atomic.sh`; update it if those change.
- Cron has a minimal `PATH`, so the watchdog hard-sets conda at
  `/home/aiprah/miniconda3` for the relaunched pipeline.
