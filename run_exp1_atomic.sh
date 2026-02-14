#!/bin/bash

# Run only EXP1: Intra-Dataset Baseline Performance
# 3 datasets * 6 models * 5 folds = 90 experiments

set -u

DEVICE=${1:-0}
OUTPUT_DIR="runs/experiments"
ERROR_LOG="${OUTPUT_DIR}/exp1_errors.txt"
MAX_RETRIES=3
EPOCHS=100

echo "Starting EXP1: Intra-Dataset Baseline Performance"
echo "Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
echo "Max retries: $MAX_RETRIES"
echo "================================"
echo ""

mkdir -p "$OUTPUT_DIR"
> "$ERROR_LOG"

run_experiment() {
    local dataset="$1"
    local model="$2"
    local fold="$3"
    
    local exp_name="exp1_${dataset}_${model}_fold${fold}"
    local done_file="${OUTPUT_DIR}/${exp_name}/done.txt"
    
    if [[ -f "$done_file" ]]; then
        echo "[SKIP] $exp_name - already completed"
        return 0
    fi
    
    echo ""
    echo "========================================================================"
    echo "Experiment: $exp_name"
    echo "========================================================================"
    
    local attempt=1
    while [[ $attempt -le $MAX_RETRIES ]]; do
        echo "Attempt $attempt/$MAX_RETRIES for $exp_name"
        
        conda activate aiprah5090
        
        python run_single_experiment.py \
            --group exp1 \
            --dataset "$dataset" \
            --model "$model" \
            --fold "$fold" \
            --img-size 1024 \
            --epochs $EPOCHS \
            --device $DEVICE \
            --output-dir "$OUTPUT_DIR" \
            --type fold
        
        local exit_code=$?
        
        if [[ $exit_code -eq 0 && -f "$done_file" ]]; then
            echo "[SUCCESS] $exp_name completed successfully"
            return 0
        fi
        
        echo "[RETRY] $exp_name failed (attempt $attempt/$MAX_RETRIES)"
        ((attempt++))
        
        sleep 5
    done
    
    echo "[ERROR] $exp_name failed after $MAX_RETRIES attempts" | tee -a "$ERROR_LOG"
    echo "Timestamp: $(date)" >> "$ERROR_LOG"
    echo "---" >> "$ERROR_LOG"
    
    return 1
}

# Run all EXP1 experiments
for dataset in hi_res low_res literature; do
    for model in yolov5s yolov5m yolov8s yolov8m yolo11s yolo11m; do
        for fold in {0..4}; do
            run_experiment $dataset $model $fold
        done
    done
done

echo ""
echo "========================================================================"
echo "EXP1 completed!"
echo "========================================================================"
echo ""

if [[ -s "$ERROR_LOG" ]]; then
    echo "⚠️  Some experiments failed. Check $ERROR_LOG for details:"
    cat "$ERROR_LOG"
    exit 1
else
    echo "✅ All EXP1 experiments completed successfully!"
    exit 0
fi
