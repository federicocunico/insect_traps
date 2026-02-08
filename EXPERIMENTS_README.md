# Insect Detection Experiment Framework

This document describes the experiment framework for *Scaphoideus titanus* detection research, implementing the research plan for evaluating high-resolution controlled acquisition datasets against field-acquired and literature datasets.

## Quick Start

```bash
# Activate environment
conda activate aiprah5090

# Run quick test to verify setup (2 epochs, ~2 minutes)
python run_experiments.py --test

# List available experiments
python run_experiments.py --list

# Run a specific experiment group
python run_experiments.py --group exp1 --device 0

# Run all experiments
python run_experiments.py --all --device 1
```

## Project Structure

```
insect_traps/
├── run_experiments.py              # Main experiment runner script
├── detector/
│   ├── datasets/
│   │   ├── data_loader.py          # Unified data loader with caching
│   │   └── pytorch_dataset.py      # PyTorch dataset for Faster R-CNN
│   ├── experiments/
│   │   └── experiment_runner.py    # Core experiment framework
│   └── data/
│       ├── hi_res/                 # High-resolution controlled dataset
│       ├── low_res/                # Field-acquired low-res dataset
│       │   ├── part1-5/            # Annotation batches (CVAT XML)
│       │   └── merged/             # Auto-generated merged dataset
│       └── InsectDetectionDataset/ # Literature dataset
├── tests/
│   └── test_experiments.py         # Test suite
└── runs/
    └── experiments/                # Experiment outputs
```

## Datasets

| Dataset | Description | Format | Images | Annotations |
|---------|-------------|--------|--------|-------------|
| **hi_res** | High-resolution controlled acquisition | YOLO | ~1760 | ~1000 |
| **low_res** | Field-acquired smartphone images | CVAT XML → YOLO | 3800 | 13142 |
| **literature** | Checola et al. (2024) dataset | YOLO | ~615 | ~1329 |
| **combined** | All datasets merged | YOLO | ~6175 | ~15471 |
| **hi_res_low_res** | Novel datasets only | YOLO | ~5560 | ~14142 |

### Data Preparation

The framework automatically handles:
- Merging CVAT XML annotations from low_res parts
- Converting coordinates to YOLO format
- Creating stratified k-fold splits
- Caching processed datasets

```python
from detector.datasets.data_loader import DatasetManager

manager = DatasetManager(Path('detector/data'))

# Merge low_res parts (cached)
output_dir, stats = manager.merge_low_res_parts()

# Prepare dataset with k-fold splits
dataset_dir, splits = manager.prepare_dataset('hi_res', n_folds=5)
```

## Models

### YOLO Family (Ultralytics)
- YOLOv5: `yolov5n`, `yolov5s`, `yolov5m`
- YOLOv8: `yolov8n`, `yolov8s`, `yolov8m`
- YOLO11: `yolo11n`, `yolo11s`, `yolo11m`

### Alternative Models
- **Faster R-CNN**: `fasterrcnn_resnet50` (torchvision)
- **RT-DETR**: `rtdetr_l`, `rtdetr_x` (Ultralytics)

## Experiment Groups

### Exp1: Intra-Dataset Baseline Performance
5-fold cross-validation on each dataset independently.
```bash
python run_experiments.py --group exp1
```

### Exp2: Resolution Impact Analysis
Tests image sizes: 512, 640, 768, 1024
```bash
python run_experiments.py --group exp2
```

### Exp3: Cross-Dataset Generalization
Train on one dataset, evaluate on another.
```bash
python run_experiments.py --group exp3
```

### Exp4: Dataset Combination Strategies
Combined dataset training.
```bash
python run_experiments.py --group exp4
```

### Exp5: Alternative Models
Non-YOLO architectures (Faster R-CNN, RT-DETR).
```bash
python run_experiments.py --group exp5
```

## Caching System

All experiment results are cached to avoid re-computation:
- **Dataset caching**: Processed datasets stored in `detector/data/.cache/`
- **Experiment caching**: Results stored in `runs/experiments/.cache/`

To force re-run:
```bash
python run_experiments.py --group exp1 --force
```

## Configuration

### Command Line Options
```
--group {exp1,exp2,exp3,exp4,exp5,test}  Run specific experiment group
--all                                     Run all experiments
--test                                    Quick test (2 epochs)
--device DEVICE                           GPU device (default: 0)
--epochs EPOCHS                           Training epochs (default: 100)
--patience PATIENCE                       Early stopping (default: 20)
--seed SEED                               Random seed (default: 42)
--force                                   Force re-run cached experiments
--list                                    List available experiments
```

### Programmatic Usage
```python
from detector.experiments import ExperimentRunner, MODEL_CONFIGS

runner = ExperimentRunner(device=0)

# Run k-fold experiment
results = runner.run_kfold_experiment(
    name='my_experiment',
    dataset='hi_res',
    model_name='yolov8s',
    n_folds=5,
    epochs=100,
    img_size=1024
)

# Aggregate results
aggregated = runner.aggregate_kfold_results(results)
print(f"mAP50: {aggregated['mAP50'][0]:.4f} ± {aggregated['mAP50'][1]:.4f}")
```

## Output Metrics

For each experiment, the following metrics are computed:
- **mAP@0.5**: Primary comparison metric
- **mAP@0.75**: Stricter localization
- **mAP@[0.5:0.95]**: COCO-style comprehensive metric
- **Precision**: False positive rate assessment
- **Recall**: False negative rate assessment
- **F1-score**: Harmonic mean

Results are saved to:
- `runs/experiments/{group}_results.csv`
- `runs/experiments/{group}_results.tex` (LaTeX table)

## Testing

```bash
# Run quick integration test
python tests/test_experiments.py --quick

# Run full test suite
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_experiments.py::TestDataLoader -v
```

## Requirements

Core dependencies:
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
albumentations>=1.3.0
torchmetrics>=1.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
```

Install with:
```bash
pip install -r detector/requirements.txt
```

## Research Plan Mapping

| Research Plan Section | Implementation |
|----------------------|----------------|
| Experiment 1: Intra-Dataset Baseline | `--group exp1` |
| Experiment 2: Resolution Impact | `--group exp2` |
| Experiment 3: Cross-Dataset | `--group exp3` |
| Experiment 4: Dataset Combination | `--group exp4` |
| Experiment 5: Data Quantity vs Quality | Subsampling in exp1 |
| Experiment 6: Ablation Studies | `--group exp5` + configs |

## Expected Results Format

After running experiments, results are in:
```
runs/experiments/
├── exp1_hi_res_yolov8s/
│   ├── fold_0/
│   │   ├── data.yaml
│   │   ├── train/
│   │   │   ├── weights/best.pt
│   │   │   ├── results.csv
│   │   │   └── ...
│   ├── fold_1/
│   └── ...
├── .cache/
│   └── experiments_cache.json
└── exp1_results.csv
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```python
MODEL_CONFIGS['yolov8s'].batch_size = 8
```

### Missing Dependencies
```bash
pip install albumentations torchmetrics pycocotools
```

### Data Not Found
Ensure datasets are in correct locations:
```
detector/data/
├── hi_res/
│   ├── images/
│   ├── labels/
│   └── hi_res.yaml
└── low_res/
    └── part1-5/
```
