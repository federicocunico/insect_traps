# YOLO Model Training and Comparison

This directory contains scripts for training and comparing different YOLO models on the hi_res insect detection dataset.

## Scripts Overview

### 1. `train_compare_models.py` - Train All Models and Compare
Main script that trains 9 different YOLO models and generates comparison results.

**Models trained:**
- YOLOv5: n, s, m (nano, small, medium)
- YOLOv8: n, s, m
- YOLOv11: n, s, m

**Features:**
- Trains each model sequentially
- Extracts metrics from training results
- Creates comparison DataFrame
- Saves results to CSV and Excel
- Shows top 3 models
- Provides summary statistics by YOLO version

**Usage:**
```bash
conda activate aiprah5090
cd /home/aiprah/github/insect_traps
python detector/yolo/train_compare_models.py
```

**Output:**
- `runs/detect_comparison/[model_name]/` - Individual model results
- `runs/detect_comparison/model_comparison_results.csv` - Summary table
- `runs/detect_comparison/model_comparison_results.xlsx` - Excel format

### 2. `analyze_results.py` - Analyze Existing Results
Analyzes already-trained models without retraining.

**Usage:**
```bash
conda activate aiprah5090
python detector/yolo/analyze_results.py [project_dir]
```

**Example:**
```bash
python detector/yolo/analyze_results.py runs/detect_comparison
```

### 3. `visualize_comparison.py` - Create Visualizations
Generates comparison plots from results CSV.

**Visualizations created:**
- Bar charts comparing all metrics
- Line plots showing performance by model size
- Precision vs Recall scatter plot
- mAP50-95 heatmap

**Usage:**
```bash
conda activate aiprah5090
python detector/yolo/visualize_comparison.py [results_csv] [output_dir]
```

**Example:**
```bash
python detector/yolo/visualize_comparison.py runs/detect_comparison/model_comparison_results.csv
```

**Output:**
- `model_comparison_bars.png` - Bar chart comparison
- `model_comparison_lines.png` - Line plot comparison
- `precision_recall_scatter.png` - Scatter plot
- `map_heatmap.png` - Heatmap visualization

### 4. `yolo_train.py` - Train Single Model
Train a single YOLO model (original script).

**Usage:**
```bash
conda activate aiprah5090
python detector/yolo/yolo_train.py
```

### 5. `train_hi_res.py` - Example Training Script
Simple example for training on hi_res dataset.

## Metrics Tracked

- **mAP50**: Mean Average Precision at IoU threshold 0.5
- **mAP50-95**: Mean Average Precision at IoU thresholds 0.5 to 0.95
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **Box Loss**: Bounding box regression loss
- **Epochs Trained**: Number of training epochs completed

## Configuration

Default training parameters:
```python
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640
DEVICE = 1  # GPU device
PATIENCE = 20  # Early stopping patience
```

## Workflow

### Full Training and Comparison Pipeline:

1. **Train all models:**
   ```bash
   python detector/yolo/train_compare_models.py
   ```

2. **Analyze results:**
   ```bash
   python detector/yolo/analyze_results.py runs/detect_comparison
   ```

3. **Create visualizations:**
   ```bash
   python detector/yolo/visualize_comparison.py runs/detect_comparison/model_comparison_results.csv
   ```

### Analyze Existing Results Only:

If you've already trained models and just want to regenerate the comparison:

```bash
python detector/yolo/analyze_results.py runs/detect_comparison
python detector/yolo/visualize_comparison.py runs/detect_comparison/model_comparison_results.csv
```

## Expected Results Format

The comparison table includes:

| model | epochs_trained | final_mAP50 | final_mAP50-95 | best_mAP50 | best_mAP50-95 | final_precision | final_recall | best_precision | best_recall | final_train_loss | final_val_loss | batch_size | img_size |
|-------|----------------|-------------|----------------|------------|---------------|-----------------|--------------|----------------|-------------|------------------|----------------|------------|----------|
| yolov11m | 100 | 0.8543 | 0.6234 | 0.8621 | 0.6301 | 0.8234 | 0.7821 | 0.8412 | 0.8021 | 0.0234 | 0.0312 | 16 | 640 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## Tips

- **GPU Memory**: If you encounter OOM errors, reduce `BATCH_SIZE`
- **Training Time**: Larger models (m) take significantly longer than nano (n)
- **Early Stopping**: Training will stop if validation doesn't improve for 20 epochs
- **Resuming**: If training is interrupted, YOLO can resume from last checkpoint

## Requirements

```bash
pip install ultralytics pandas openpyxl matplotlib seaborn pyyaml
```
