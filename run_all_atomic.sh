#!/bin/bash

# Atomic experiment runner with retry logic
# Each experiment runs independently with cleanup between runs

set -u

DEVICE=${1:-0}
OUTPUT_DIR="runs/experiments"
ERROR_LOG="${OUTPUT_DIR}/error.txt"
MAX_RETRIES=3
EPOCHS=100

echo "Starting atomic experiment runner"
echo "Device: $DEVICE"
echo "Output: $OUTPUT_DIR"
echo "Max retries: $MAX_RETRIES"
echo "================================"
echo ""

mkdir -p "$OUTPUT_DIR"
# Append to error log instead of truncating (safe for re-runs)
echo "" >> "$ERROR_LOG"
echo "=== Run started at $(date) ===" >> "$ERROR_LOG"

run_experiment() {
    local exp_type="$1"
    local group="$2"
    local model="$3"
    local img_size="$4"
    shift 4
    
    local exp_name=""
    local python_args=""
    
    if [[ "$exp_type" == "fold" ]]; then
        local dataset="$1"
        local fold="$2"
        exp_name="${group}_${dataset}_${model}"
        if [[ "$img_size" != "1024" ]]; then
            exp_name="${exp_name}_img${img_size}"
        fi
        exp_name="${exp_name}_fold${fold}"
        python_args="--type fold --dataset $dataset --fold $fold"
    elif [[ "$exp_type" == "cross" ]]; then
        local train_ds="$1"
        local test_ds="$2"
        exp_name="${group}_${train_ds}_to_${test_ds}_${model}"
        python_args="--type cross --train-dataset $train_ds --test-dataset $test_ds"
    fi
    
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
        
        # conda activate aiprah5090
        
        python run_single_experiment.py \
            --group "$group" \
            --model "$model" \
            --img-size "$img_size" \
            --epochs $EPOCHS \
            --device $DEVICE \
            --output-dir "$OUTPUT_DIR" \
            $python_args
        
        local exit_code=$?
        
        if [[ $exit_code -eq 0 && -f "$done_file" ]]; then
            echo "[SUCCESS] $exp_name completed successfully"
            return 0
        fi
        
        echo "[RETRY] $exp_name failed (attempt $attempt/$MAX_RETRIES)"
        ((attempt++))
        
        # Wait for GPU memory to fully release before retry
        sleep 10
    done
    
    echo "[ERROR] $exp_name failed after $MAX_RETRIES attempts" | tee -a "$ERROR_LOG"
    echo "Timestamp: $(date)" >> "$ERROR_LOG"
    echo "---" >> "$ERROR_LOG"
    
    return 1
}

# EXP1: Intra-Dataset Baseline Performance
# 3 datasets * 6 models * 5 folds = 90 experiments
echo "=========================================="
echo "EXP1: Intra-Dataset Baseline Performance"
echo "=========================================="

for dataset in hi_res low_res literature; do
    for model in yolov5s yolov5m yolov8s yolov8m yolo11s yolo11m; do
        for fold in {0..4}; do
            run_experiment fold exp1 $model 1024 $dataset $fold
        done
    done
done

# EXP2: Resolution Impact Analysis
# 3 datasets * 2 models * 4 img_sizes * 5 folds = 120 experiments
echo ""
echo "=========================================="
echo "EXP2: Resolution Impact Analysis"
echo "=========================================="

for dataset in hi_res low_res literature; do
    for model in yolov8s yolo11s; do
        for img_size in 512 640 768 1024; do
            for fold in {0..4}; do
                run_experiment fold exp2 $model $img_size $dataset $fold
            done
        done
    done
done

# EXP3: Cross-Dataset Generalization
# 6 pairs * 2 models = 12 experiments
echo ""
echo "=========================================="
echo "EXP3: Cross-Dataset Generalization"
echo "=========================================="

cross_pairs=(
    "hi_res literature"
    "hi_res low_res"
    "low_res literature"
    "low_res hi_res"
    "literature hi_res"
    "literature low_res"
)

for pair in "${cross_pairs[@]}"; do
    read -r train_ds test_ds <<< "$pair"
    for model in yolov8s yolo11s; do
        run_experiment cross exp3 $model 1024 $train_ds $test_ds
    done
done

# EXP4: Dataset Combination Strategies
# 2 datasets * 2 models * 5 folds = 20 experiments
echo ""
echo "=========================================="
echo "EXP4: Dataset Combination Strategies"
echo "=========================================="

for dataset in hi_res_low_res combined; do
    for model in yolov8s yolo11s; do
        for fold in {0..4}; do
            run_experiment fold exp4 $model 1024 $dataset $fold
        done
    done
done

# EXP5: Alternative Models
# 2 datasets * 2 models * 3 folds = 12 experiments
echo ""
echo "=========================================="
echo "EXP5: Alternative Models"
echo "=========================================="

for dataset in hi_res low_res; do
    for model in fasterrcnn_resnet50 rtdetr_l; do
        for fold in {0..2}; do
            run_experiment fold exp5 $model 640 $dataset $fold
        done
    done
done

echo ""
echo "========================================================================"
echo "All experiments completed!"
echo "========================================================================"
echo ""

if [[ -s "$ERROR_LOG" ]]; then
    echo "⚠️  Some experiments failed. Check $ERROR_LOG for details:"
    cat "$ERROR_LOG"
    exit 1
else
    echo "✅ All experiments completed successfully!"
    exit 0
fi
