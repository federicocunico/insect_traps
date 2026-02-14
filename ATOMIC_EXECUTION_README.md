# Atomic Experiment Execution

This directory now contains robust atomic experiment execution scripts that handle unstable environments and potential OOM (Out-Of-Memory) issues.

## Problem Solved

The original `run_experiments.py --all` would run all experiments in a single monolithic Python process, which caused:
- **Memory leaks**: Models and data stayed in memory between experiments
- **No recovery**: If a process was killed (OOM, system instability), you lost all progress
- **Hard to debug**: Couldn't identify which specific experiment failed

## Solution: Atomic Execution

Each individual experiment (including each fold) runs as a separate, isolated process with:
- ✅ **Automatic retry logic**: Up to 3 attempts per experiment
- ✅ **Progress tracking**: `done.txt` markers prevent re-running completed experiments
- ✅ **Memory cleanup**: Explicit GPU cache clearing and garbage collection
- ✅ **Error logging**: Failed experiments logged to `error.txt`
- ✅ **Resumable**: Can stop and restart without losing progress

## Files Created

### `run_single_experiment.py`
Runs a single atomic experiment (one fold or one cross-dataset eval) with cleanup.

```bash
# Example: Run exp1_hi_res_yolov8s_fold0
python run_single_experiment.py \
    --group exp1 \
    --dataset hi_res \
    --model yolov8s \
    --fold 0 \
    --device 0
```

### `run_all_atomic.sh`
Orchestrates all experiments (exp1-exp5) with retry logic.

```bash
# Run all experiments on GPU 0
./run_all_atomic.sh 0

# Run all experiments on GPU 1
./run_all_atomic.sh 1
```

Total experiments: **254 atomic runs**
- EXP1: 90 experiments (3 datasets × 6 models × 5 folds)
- EXP2: 120 experiments (3 datasets × 2 models × 4 img_sizes × 5 folds)
- EXP3: 12 experiments (6 pairs × 2 models)
- EXP4: 20 experiments (2 datasets × 2 models × 5 folds)
- EXP5: 12 experiments (2 datasets × 2 models × 3 folds)

### `run_exp1_atomic.sh`
Run only EXP1 experiments (baseline performance).

```bash
# Run only EXP1 on GPU 0
./run_exp1_atomic.sh 0
```

## How It Works

### 1. Done File Tracking
Each experiment creates a `done.txt` file upon successful completion:
```
runs/experiments/exp1_hi_res_yolov8s_fold0/done.txt
```

Before running, the script checks for `done.txt`. If it exists, the experiment is skipped.

### 2. Retry Logic
If an experiment fails:
1. Retry up to 3 times automatically
2. Wait 5 seconds between retries
3. After 3 failures, log to `error.txt` and continue

### 3. Memory Cleanup
After each experiment, the following cleanup happens:
- Delete model objects
- Delete optimizer/scheduler (for non-YOLO)
- Run Python garbage collection
- Clear CUDA cache
- Synchronize GPU

### 4. Error Logging
Failed experiments are logged to `runs/experiments/error.txt`:
```
[ERROR] exp1_hi_res_yolov8s_fold2 failed after 3 attempts
Timestamp: Mon Feb 10 14:32:15 2026
---
```

## Improvements Made to `experiment_runner.py`

Added explicit memory cleanup to all trainers:

```python
# After training/validation
del model
del results
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
```

This is added to:
- `YOLOTrainer.train()` and `validate()`
- `FasterRCNNTrainer.train()` and `validate()`
- `RTDETRTrainer.train()` and `validate()`

## Usage Examples

### Run all experiments
```bash
./run_all_atomic.sh 0
```

### Run only EXP1
```bash
./run_exp1_atomic.sh 0
```

### Run a specific experiment manually
```bash
python run_single_experiment.py \
    --group exp1 \
    --dataset hi_res \
    --model yolov8s \
    --fold 0 \
    --img-size 1024 \
    --epochs 100 \
    --device 0
```

### Resume after interruption
Just run the same command again! Completed experiments (with `done.txt`) are automatically skipped.

### Check progress
```bash
# Count completed experiments
find runs/experiments -name "done.txt" | wc -l

# Check for errors
cat runs/experiments/error.txt
```

### Force re-run a specific experiment
```bash
# Delete the done.txt file
rm runs/experiments/exp1_hi_res_yolov8s_fold0/done.txt

# Then run the script again
./run_exp1_atomic.sh 0
```

## Monitoring

While experiments run, you can monitor:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Count completed experiments
watch -n 10 'find runs/experiments -name "done.txt" | wc -l'

# Check errors in real-time
tail -f runs/experiments/error.txt

# Check latest experiment output
ls -lt runs/experiments/ | head -20
```

## Benefits

1. **Robustness**: System crashes don't lose progress
2. **Resumable**: Stop and restart anytime
3. **Debuggable**: Know exactly which experiment failed
4. **Memory efficient**: Each experiment starts fresh
5. **Parallel ready**: Can easily modify to run multiple experiments in parallel
6. **Production ready**: Handles failures gracefully

## Original Script

The original `run_experiments.py` is still available for running single experiments or groups:

```bash
# Still works for testing
python run_experiments.py --test

# Still works for single groups
python run_experiments.py --group exp1
```

However, for production runs on unstable systems, use the atomic scripts instead.
