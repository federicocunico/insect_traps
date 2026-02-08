# Research Plan: High-Resolution and Field-Acquired Dataset Integration for Scaphoideus titanus Detection

## Executive Summary

This research extends the state-of-the-art (Checola et al., 2024) by introducing two novel datasets: a high-resolution (hi-res) dataset acquired under controlled conditions and a field-acquired low-resolution (field-lr) dataset captured with mobile phones. The study aims to demonstrate that controlled high-resolution acquisition substantially improves detection accuracy while also showing that larger, field-acquired datasets can complement or exceed existing approaches despite lower resolution.

---

## 1. Research Objectives

### Primary Objectives
1. **Demonstrate superiority of high-resolution controlled acquisition** over existing methods for S. titanus detection
2. **Validate the utility of field-acquired low-resolution data** for practical deployment scenarios
3. **Establish optimal dataset combination strategies** that balance quality and quantity
4. **Provide reproducible benchmarks** against literature baseline (Checola et al., 2024)

### Secondary Objectives
- Investigate resolution-accuracy trade-offs in insect detection
- Assess domain transfer between acquisition methods
- Develop deployment-ready recommendations for practitioners

---

## 2. Dataset Description

### 2.1 Literature Dataset (LIT)
- **Source**: Checola et al., 2024
- **Size**: 615 images, ~1,329 ST annotations, ~1,506 OI annotations
- **Acquisition**: Mixed (field photos, laboratory, scanned traps, smart-trap)
- **Resolution**: Varied (mostly medium, some high from scans)
- **Controlled conditions**: No (heterogeneous sources)

### 2.2 High-Resolution Dataset (HI-RES)
- **Source**: Custom acquisition box (current work)
- **Size**: [TO BE SPECIFIED] images, [NUMBER] ST annotations
- **Acquisition**: Controlled imaging box with fixed geometry, LED illumination
- **Resolution**: High (~26.6 µm/pixel, 4056×3040 sensor)
- **Controlled conditions**: Yes (standardized illumination, geometry, focus)

### 2.3 Field Low-Resolution Dataset (FIELD-LR)
- **Source**: Mobile phone acquisition in field conditions
- **Size**: [TO BE SPECIFIED] images, [NUMBER] ST annotations (substantially larger than LIT)
- **Acquisition**: Smartphone photography in vineyard settings
- **Resolution**: Variable, generally lower than HI-RES
- **Controlled conditions**: No (realistic field deployment scenario)

### 2.4 Combined Datasets
- **HI-RES + FIELD-LR**: Full novel contribution
- **ALL (HI-RES + FIELD-LR + LIT)**: Maximum data scenario

---

## 3. Model Selection

All models selected for ease of implementation, reproducibility, and established performance in small-object detection.

### Primary Models (YOLO Family)
1. **YOLOv5s** - Lightweight, fast, excellent baseline
2. **YOLOv5m** - Medium capacity, balanced performance
3. **YOLOv8s** - Latest improvements, anchor-free
4. **YOLOv8m** - Higher capacity variant
5. **YOLO11s** - State-of-the-art, latest architecture
6. **YOLO11m** - Higher capacity YOLO11

### Rationale
- **YOLOv5**: Proven track record, extensive validation in literature
- **YOLOv8**: Improved architecture, anchor-free detection, better small-object handling
- **YOLO11**: Latest advancements, optimal for current benchmarking
- All models available via Ultralytics Python package for easy implementation

### Implementation Framework
```python
# Primary framework: Ultralytics
from ultralytics import YOLO

# Models to test
models = [
    'yolov5s.pt', 'yolov5m.pt',
    'yolov8s.pt', 'yolov8m.pt',
    'yolo11s.pt', 'yolo11m.pt'
]
```

---

## 4. Experimental Design

### 4.1 Experiment Structure Overview

```
├── Experiment 1: Intra-Dataset Baseline Performance
│   ├── 1.1: LIT dataset (5-fold CV) - Reproduce literature
│   ├── 1.2: HI-RES dataset (5-fold CV) - Controlled acquisition
│   └── 1.3: FIELD-LR dataset (5-fold CV) - Field acquisition
│
├── Experiment 2: Resolution Impact Analysis
│   ├── 2.1: HI-RES at multiple resolutions (1024, 768, 640, 512)
│   ├── 2.2: FIELD-LR at matched resolutions
│   └── 2.3: LIT at matched resolutions
│
├── Experiment 3: Cross-Dataset Generalization
│   ├── 3.1: Train HI-RES → Test LIT
│   ├── 3.2: Train HI-RES → Test FIELD-LR
│   ├── 3.3: Train FIELD-LR → Test LIT
│   ├── 3.4: Train FIELD-LR → Test HI-RES
│   ├── 3.5: Train LIT → Test HI-RES
│   └── 3.6: Train LIT → Test FIELD-LR
│
├── Experiment 4: Dataset Combination Strategies
│   ├── 4.1: Train (HI-RES + FIELD-LR) → Test on both separately
│   ├── 4.2: Train (HI-RES + LIT) → Test on all three
│   ├── 4.3: Train (FIELD-LR + LIT) → Test on all three
│   └── 4.4: Train ALL → Test on each dataset
│
├── Experiment 5: Data Quantity vs Quality
│   ├── 5.1: HI-RES subsampling (100%, 75%, 50%, 25%)
│   ├── 5.2: FIELD-LR subsampling (100%, 75%, 50%, 25%)
│   └── 5.3: Quality-quantity trade-off analysis
│
└── Experiment 6: Ablation Studies
    ├── 6.1: Augmentation impact per dataset
    ├── 6.2: Pre-processing impact (controlled vs non-controlled)
    └── 6.3: Model architecture sensitivity
```

---

## 5. Detailed Experimental Protocols

### 5.1 Experiment 1: Intra-Dataset Baseline Performance

**Objective**: Establish baseline performance for each dataset independently and validate against literature.

#### Protocol
- **Method**: 5-fold stratified cross-validation
- **Models**: All six YOLO variants
- **Image sizes**: 1024px (primary), 768px, 640px
- **Augmentation**: Default YOLO augmentation enabled
- **Training**: 
  - Epochs: 100 (with early stopping)
  - Batch size: 16 (adjust for GPU memory)
  - Optimizer: SGD (YOLO default)
  - Learning rate: Default YOLO schedule

#### LIT Dataset Specific Requirements
- **Critical**: Must reproduce Checola et al. (2024) results
- Use exact same data splits if available, otherwise document random seed
- Compare metrics directly: mAP@0.5, mAP@0.75, mAP@[0.5:0.95], Precision, Recall, F1
- Expected baseline: YOLOv8s should achieve ~0.92 mAP@0.5

#### Evaluation Metrics
```
Primary Metrics:
- mAP@0.5 (main comparison point with literature)
- mAP@0.75 (stricter localization)
- mAP@[0.5:0.95] (COCO-style comprehensive metric)
- Precision (FP rate assessment)
- Recall (FN rate assessment)
- F1-score (harmonic mean)

Secondary Metrics:
- Inference time (FPS)
- Model size (parameters, disk size)
- Training time
```

#### Expected Outcomes
- HI-RES should outperform LIT (hypothesis: controlled conditions + resolution)
- FIELD-LR performance relative to LIT will indicate if quantity compensates for quality
- All results should include mean ± std across folds

---

### 5.2 Experiment 2: Resolution Impact Analysis

**Objective**: Isolate the effect of image resolution on detection performance.

#### Protocol
- **Datasets**: LIT, HI-RES, FIELD-LR
- **Image sizes tested**: 1024, 768, 640, 512 pixels
- **Models**: YOLOv5m, YOLOv8s, YOLO11s (representative selection)
- **Evaluation**: 5-fold CV for each combination

#### Specific Tests
1. **HI-RES resolution sweep**
   - Determine optimal resolution for controlled acquisition
   - Assess diminishing returns of higher resolution
   
2. **FIELD-LR resolution sweep**
   - Understand resolution limits for field acquisition
   - Compare with HI-RES at matched resolutions

3. **LIT resolution sweep**
   - Reproduce literature at multiple scales
   - Compare heterogeneous dataset behavior

#### Analysis
- Plot mAP vs resolution for each dataset
- Statistical comparison at each resolution point
- Identify resolution threshold for reliable detection
- Calculate computational cost vs accuracy trade-off

---

### 5.3 Experiment 3: Cross-Dataset Generalization

**Objective**: Assess domain transfer and robustness across acquisition methods.

#### Protocol
Each experiment follows: Train on Dataset A → Validate on Dataset A → Test on Dataset B

#### Critical Comparisons

**3.1: HI-RES → LIT**
- Tests if controlled acquisition generalizes to heterogeneous real-world data
- Success metric: Performance drop < 10% vs LIT baseline

**3.2: HI-RES → FIELD-LR**
- Tests if controlled data generalizes to field conditions
- Indicates deployment feasibility

**3.3: FIELD-LR → LIT**
- Tests if field data captures sufficient diversity
- Large dataset vs smaller heterogeneous dataset

**3.4: FIELD-LR → HI-RES**
- Tests if field training can handle high-quality data
- Reverse of 3.2

**3.5: LIT → HI-RES**
- Literature baseline generalization to high-quality data
- Reference for improvement quantification

**3.6: LIT → FIELD-LR**
- Literature baseline generalization to field data
- Validates FIELD-LR as legitimate test set

#### Evaluation
- Full test set evaluation (no CV on target domain)
- Per-class analysis if multiple classes present
- Failure case analysis: qualitative review of FP and FN
- Domain gap quantification: performance drop percentage

---

### 5.4 Experiment 4: Dataset Combination Strategies

**Objective**: Determine optimal dataset mixing for maximum performance.

#### Protocol

**4.1: HI-RES + FIELD-LR (Novel Combined Dataset)**
- Train on union of both novel datasets
- Test on HI-RES holdout, FIELD-LR holdout separately
- Evaluate if combination improves generalization

**4.2: HI-RES + LIT**
- Augment high-quality data with literature diversity
- Test on all three dataset holdouts

**4.3: FIELD-LR + LIT**
- Large field dataset + diverse literature dataset
- Test generalization across all domains

**4.4: ALL (HI-RES + FIELD-LR + LIT)**
- Maximum data strategy
- Test if more data always helps
- Individual test set performance for each domain

#### Mixing Strategies
1. **Naive mixing**: Simply concatenate datasets
2. **Balanced sampling**: Equalize representation across sources
3. **Weighted sampling**: Prioritize high-quality annotations

#### Analysis
- Compare each combination against best single-dataset baseline
- Assess if combination is better than best individual dataset
- Cost-benefit analysis: improvement vs data collection effort

---

### 5.5 Experiment 5: Data Quantity vs Quality

**Objective**: Understand the trade-off between dataset size and acquisition quality.

#### Protocol

**5.1: HI-RES Subsampling**
- Create subsets: 100%, 75%, 50%, 25%, 10% of HI-RES
- Maintain annotation distribution
- Train and evaluate each subset (3-fold CV due to smaller data)

**5.2: FIELD-LR Subsampling**
- Same subsampling strategy
- Compare learning curves against HI-RES

#### Analysis
- **Learning curves**: Plot mAP vs dataset size
- **Crossover analysis**: At what size does FIELD-LR match HI-RES performance?
- **Quality coefficient**: Quantify how many field images equal one high-res image
- **Practical recommendations**: Minimum dataset sizes for deployment

#### Key Question
*"Is it better to collect 1000 field images or 100 controlled high-res images?"*

---

### 5.6 Experiment 6: Ablation Studies

**Objective**: Isolate impact of specific methodological choices.

#### 6.1: Augmentation Impact
- **Test**: Default augmentation vs no augmentation vs aggressive augmentation
- **Datasets**: All three
- **Expected**: Controlled HI-RES may need less augmentation; FIELD-LR may benefit more

#### 6.2: Pre-processing Impact
- **LIT**: Test with/without brightness enhancement (as per Checola et al.)
- **HI-RES**: Minimal pre-processing expected (controlled acquisition)
- **FIELD-LR**: Test various pre-processing pipelines

Pre-processing variants:
```
1. None (raw)
2. Auto-crop (remove background)
3. Brightness/contrast adjustment
4. Sharpening filter
5. Combinations of above
```

#### 6.3: Model Architecture Sensitivity
- Compare YOLO variants (v5, v8, v11)
- Analyze which architecture benefits most from high-resolution
- Determine if certain models handle domain shift better

---

## 6. Evaluation Protocol for Literature Comparison

### 6.1 Direct Reproduction Requirements

To ensure fair comparison with Checola et al. (2024):

1. **Exact Metric Definitions**
   - IoU threshold: 0.5 for mAP@0.5
   - IoU threshold: 0.75 for mAP@0.75
   - IoU range: 0.5:0.95:0.05 for mAP@[0.5:0.95]
   - Use COCO evaluation protocol

2. **Statistical Reporting**
   - Report mean ± std for all metrics
   - Use same CV folds (if available) or same random seed
   - Report confidence intervals where appropriate

3. **Training Configuration**
   - Match image resolution (primarily 1280px for comparison)
   - Use default augmentation settings
   - Report training hyperparameters explicitly

### 6.2 Performance Comparison Table Template

```
| Dataset      | Model    | mAP@0.5      | mAP@0.75     | mAP@[0.5:0.95] | Precision    | Recall       | F1          |
|--------------|----------|--------------|--------------|----------------|--------------|--------------|-------------|
| LIT (ours)   | YOLOv8s  | 0.92 ± 0.02  | 0.81 ± 0.03  | 0.66 ± 0.02    | 0.89 ± 0.05  | 0.89 ± 0.03  | 0.89 ± 0.03 |
| LIT (lit)    | YOLOv8s  | 0.92 ± 0.02  | 0.81 ± 0.03  | 0.66 ± 0.02    | 0.89 ± 0.05  | 0.89 ± 0.03  | 0.89 ± 0.03 |
| HI-RES       | YOLOv8s  | X.XX ± X.XX  | X.XX ± X.XX  | X.XX ± X.XX    | X.XX ± X.XX  | X.XX ± X.XX  | X.XX ± X.XX |
| FIELD-LR     | YOLOv8s  | X.XX ± X.XX  | X.XX ± X.XX  | X.XX ± X.XX    | X.XX ± X.XX  | X.XX ± X.XX  | X.XX ± X.XX |
| HI+FIELD     | YOLOv8s  | X.XX ± X.XX  | X.XX ± X.XX  | X.XX ± X.XX    | X.XX ± X.XX  | X.XX ± X.XX  | X.XX ± X.XX |
| ALL          | YOLOv8s  | X.XX ± X.XX  | X.XX ± X.XX  | X.XX ± X.XX    | X.XX ± X.XX  | X.XX ± X.XX  | X.XX ± X.XX |
```

### 6.3 Statistical Significance Testing

For all comparisons against literature baseline:
- **Paired t-test** across CV folds (when same folds used)
- **Wilcoxon signed-rank test** as non-parametric alternative
- **Effect size** (Cohen's d) for meaningful difference assessment
- **Significance level**: α = 0.05
- **Multiple comparison correction**: Bonferroni when testing multiple models

### 6.4 Qualitative Analysis Requirements

1. **Visual Examples**
   - Show detection results on each dataset
   - Include success cases and failure cases
   - Highlight morphological details visible in HI-RES vs others

2. **Error Analysis**
   - Categorize false positives (confusion with other species, debris, artifacts)
   - Categorize false negatives (degraded specimens, occlusions, small/distant insects)
   - Dataset-specific error patterns

3. **Morphological Detail Assessment**
   - Visual comparison of wing venation visibility
   - Body proportion clarity
   - Color pattern distinguishability

---

## 7. Implementation Details

### 7.1 Software Stack

```python
# Core dependencies
ultralytics==8.x.x  # YOLO models
opencv-python==4.x.x  # Image processing
numpy==1.24.x
pandas==2.x.x
matplotlib==3.x.x
seaborn==0.12.x
scikit-learn==1.3.x  # CV splits, metrics
scipy==1.11.x  # Statistical tests

# Annotation and evaluation
pycocotools==2.x.x  # COCO metrics
albumentations==1.3.x  # Augmentation (if needed beyond YOLO defaults)

# Experiment tracking
wandb==0.15.x  # Recommended for tracking experiments
tensorboard  # Alternative/complementary
```

### 7.2 Hardware Requirements

- **Minimum**: NVIDIA RTX 3090 (24GB VRAM)
- **Recommended**: NVIDIA RTX 4090 or A100
- **Storage**: ~500GB for datasets, models, results
- **RAM**: 64GB recommended for large-batch processing

### 7.3 Directory Structure

```
project/
├── data/
│   ├── literature/
│   │   ├── images/
│   │   ├── labels/
│   │   └── splits/
│   ├── hi-res/
│   │   ├── images/
│   │   ├── labels/
│   │   └── splits/
│   ├── field-lr/
│   │   ├── images/
│   │   ├── labels/
│   │   └── splits/
│   └── combined/
├── models/
│   ├── pretrained/
│   └── trained/
│       ├── exp1/
│       ├── exp2/
│       └── ...
├── results/
│   ├── metrics/
│   ├── visualizations/
│   └── analysis/
├── scripts/
│   ├── prepare_datasets.py
│   ├── train.py
│   ├── evaluate.py
│   ├── cross_dataset_eval.py
│   └── analysis.py
└── configs/
    ├── exp1_config.yaml
    ├── exp2_config.yaml
    └── ...
```

### 7.4 Training Script Template

```python
from ultralytics import YOLO
import yaml
from pathlib import Path

def train_model(config):
    """
    Train YOLO model with specified configuration.
    """
    model = YOLO(config['model'])
    
    results = model.train(
        data=config['data_yaml'],
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        name=config['experiment_name'],
        project=config['project_path'],
        exist_ok=True,
        pretrained=True,
        optimizer=config.get('optimizer', 'auto'),
        seed=config.get('seed', 42),
        deterministic=True,  # For reproducibility
        workers=config.get('workers', 8),
        # CV-specific
        kfold=config.get('kfold', 5) if config.get('use_cv') else None,
    )
    
    return results

# Example config
config = {
    'model': 'yolov8s.pt',
    'data_yaml': 'data/hi-res/data.yaml',
    'epochs': 100,
    'imgsz': 1024,
    'batch': 16,
    'experiment_name': 'exp1_hires_yolov8s',
    'project_path': 'models/trained',
    'seed': 42,
    'use_cv': True,
    'kfold': 5
}
```

### 7.5 Evaluation Script Template

```python
def evaluate_cross_dataset(train_dataset, test_dataset, model_path):
    """
    Evaluate model trained on one dataset, tested on another.
    """
    model = YOLO(model_path)
    
    # Load test dataset
    test_data_yaml = f'data/{test_dataset}/data.yaml'
    
    # Run evaluation
    metrics = model.val(
        data=test_data_yaml,
        split='test',
        imgsz=1024,
        batch=16,
        save_json=True,  # For COCO metrics
        plots=True
    )
    
    # Extract metrics
    results = {
        'mAP50': metrics.box.map50,
        'mAP75': metrics.box.map75,
        'mAP50-95': metrics.box.map,
        'precision': metrics.box.mp,
        'recall': metrics.box.mr,
        'f1': 2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr)
    }
    
    return results, metrics
```

---

## 8. Expected Outcomes and Hypotheses

### 8.1 Primary Hypotheses

**H1: Resolution Superiority**
- HI-RES will significantly outperform LIT baseline (ΔmAP@0.5 > 5%)
- Justification: Controlled acquisition + higher resolution enables finer morphological detail capture

**H2: Quantity Compensation**
- FIELD-LR will match or exceed LIT performance despite lower resolution
- Justification: Larger dataset size compensates for reduced quality

**H3: Domain Transfer**
- Models trained on HI-RES will generalize well to FIELD-LR (performance drop < 15%)
- Models trained on FIELD-LR will generalize moderately to HI-RES (performance drop < 20%)

**H4: Optimal Combination**
- HI-RES + FIELD-LR will outperform any single dataset
- ALL (including LIT) will show marginal improvement over HI-RES + FIELD-LR

**H5: Diminishing Returns**
- Resolution beyond 1024px shows diminishing returns for FIELD-LR
- HI-RES benefits from full resolution (1024px+)

### 8.2 Performance Targets

Relative to LIT baseline (YOLOv8s: mAP@0.5 = 0.92):

| Dataset          | Target mAP@0.5 | Target mAP@[0.5:0.95] | Justification                    |
|------------------|----------------|----------------------|----------------------------------|
| LIT (reproduced) | 0.92 ± 0.02    | 0.66 ± 0.02          | Match literature                 |
| HI-RES           | ≥ 0.95         | ≥ 0.75               | Controlled high-res advantage    |
| FIELD-LR         | ≥ 0.90         | ≥ 0.65               | Quantity compensates quality     |
| HI-RES + FIELD-LR| ≥ 0.96         | ≥ 0.78               | Combined strengths               |
| ALL              | ≥ 0.96         | ≥ 0.78               | Maximum diversity                |

---

## 9. Analysis and Visualization Plan

### 9.1 Quantitative Analysis

**Primary Tables**
1. **Intra-dataset performance comparison** (Experiment 1)
2. **Resolution impact analysis** (Experiment 2)
3. **Cross-dataset generalization matrix** (Experiment 3)
4. **Dataset combination results** (Experiment 4)
5. **Learning curve statistics** (Experiment 5)
6. **Ablation study results** (Experiment 6)

**Statistical Tests Summary Table**
- Each comparison with literature
- P-values, effect sizes, confidence intervals

### 9.2 Visualizations

**Required Figures**
1. **Sample images from each dataset** with annotations
   - Show visual quality differences
   - Highlight morphological details

2. **Performance comparison bar charts**
   - mAP@0.5, mAP@0.75, mAP@[0.5:0.95]
   - Error bars (std or CI)
   - Statistical significance markers

3. **Resolution vs performance curves** (Experiment 2)
   - Separate curves for each dataset
   - Identify optimal resolution points

4. **Cross-dataset generalization heatmap** (Experiment 3)
   - Rows: training dataset
   - Columns: test dataset
   - Color: mAP@0.5

5. **Learning curves** (Experiment 5)
   - X-axis: dataset size (percentage or absolute)
   - Y-axis: mAP
   - Separate curves for HI-RES vs FIELD-LR

6. **Precision-Recall curves**
   - Per dataset
   - Per model
   - Comparison overlay

7. **Confusion analysis**
   - FP/FN distribution
   - Per-dataset error patterns

8. **Qualitative results grid**
   - Best/worst detections per dataset
   - Side-by-side comparisons

### 9.3 Supplementary Material

- Complete hyperparameter configurations
- Detailed CV fold compositions
- Per-fold results (not just aggregates)
- Training curves (loss, mAP over epochs)
- Inference time benchmarks
- Full error case gallery

---

## 10. Timeline and Milestones

### Phase 1: Data Preparation (Weeks 1-2)
- [ ] Finalize HI-RES dataset annotation
- [ ] Complete FIELD-LR dataset annotation
- [ ] Obtain and prepare LIT dataset
- [ ] Create CV splits for all datasets
- [ ] Validate annotation quality (inter-annotator agreement if multiple annotators)

### Phase 2: Baseline Experiments (Weeks 3-4)
- [ ] Experiment 1: Intra-dataset baselines (all models, all datasets)
- [ ] Validate LIT reproduction against literature
- [ ] Document any discrepancies

### Phase 3: Core Experiments (Weeks 5-8)
- [ ] Experiment 2: Resolution impact analysis
- [ ] Experiment 3: Cross-dataset generalization (all pairs)
- [ ] Experiment 4: Dataset combination strategies
- [ ] Preliminary analysis and mid-point review

### Phase 4: Advanced Experiments (Weeks 9-10)
- [ ] Experiment 5: Quantity vs quality analysis
- [ ] Experiment 6: Ablation studies
- [ ] Additional experiments based on preliminary findings

### Phase 5: Analysis and Writing (Weeks 11-12)
- [ ] Statistical analysis
- [ ] Generate all visualizations
- [ ] Compile results tables
- [ ] Qualitative error analysis
- [ ] Draft manuscript sections

### Phase 6: Refinement (Weeks 13-14)
- [ ] Respond to internal review
- [ ] Additional experiments if needed
- [ ] Finalize figures and tables
- [ ] Complete manuscript

---

## 11. Reproducibility Checklist

### Code and Data
- [ ] Public repository with all training/evaluation code
- [ ] Dataset release (or access protocol if privacy-restricted)
- [ ] Pretrained model weights
- [ ] Exact environment specification (requirements.txt, Docker image)

### Documentation
- [ ] Complete hyperparameter documentation
- [ ] Random seed reporting for all experiments
- [ ] CV split indices or generation procedure
- [ ] Hardware specification

### Metrics
- [ ] Use established evaluation protocols (COCO metrics)
- [ ] Report all metrics, not just best-case
- [ ] Include variance/confidence intervals
- [ ] Provide raw predictions for external evaluation

### Comparison
- [ ] Use same evaluation code as literature when possible
- [ ] Document any differences in evaluation protocol
- [ ] Provide direct comparison on identical test sets
- [ ] Statistical significance testing

---

## 12. Risk Mitigation

### Potential Issues and Solutions

**Issue 1: HI-RES doesn't significantly outperform LIT**
- *Solution*: Emphasize combination benefits, deployment trade-offs, cost-benefit analysis
- *Fallback*: Focus on standardization and reproducibility contribution

**Issue 2: FIELD-LR underperforms due to quality issues**
- *Solution*: Highlight importance of controlled acquisition, provide quality guidelines
- *Alternative angle*: Present as practical baseline for "worst-case" field deployment

**Issue 3: Domain gap too large for cross-dataset generalization**
- *Solution*: Investigate domain adaptation techniques, fine-tuning strategies
- *Additional experiments*: Few-shot adaptation, progressive fine-tuning

**Issue 4: Dataset combination doesn't improve over best single dataset**
- *Solution*: Analyze data redundancy, investigate smart sampling strategies
- *Focus shift*: Emphasize robustness over peak performance

**Issue 5: Unable to reproduce LIT results**
- *Critical*: Document all differences systematically
- *Action*: Contact original authors for clarification
- *Alternative*: Use our LIT run as baseline if consistent across multiple attempts

---

## 13. Success Criteria

### Minimum Viable Contribution
1. **Reproduce LIT baseline** within ±2% mAP@0.5
2. **Demonstrate HI-RES advantage** (any statistically significant improvement)
3. **Show FIELD-LR is viable** (within 10% of LIT baseline)
4. **Provide reproducible protocol** for future work

### Target Contribution
1. HI-RES outperforms LIT by ≥5% mAP@0.5
2. FIELD-LR matches or exceeds LIT performance
3. Combined dataset shows improvement over any single source
4. Clear practical recommendations for practitioners

### Exceptional Contribution
1. HI-RES achieves ≥0.95 mAP@0.5 (vs 0.92 baseline)
2. FIELD-LR significantly exceeds LIT despite lower resolution
3. Demonstrate effective cross-dataset transfer learning
4. Establish new benchmark that advances the field substantially

---

## 14. Publication Strategy

### Target Venues
- **Primary**: Frontiers in Plant Science (same as Checola et al.)
- **Alternative 1**: Computers and Electronics in Agriculture
- **Alternative 2**: Precision Agriculture
- **Dataset paper**: Scientific Data (Nature)

### Key Messages
1. **High-resolution controlled acquisition significantly improves detection accuracy**
2. **Large-scale field-acquired data provides practical deployment path**
3. **Combined approach balances quality and scalability**
4. **Reproducible framework for insect detection benchmark**

### Narrative Structure
1. Introduction: FD threat, current monitoring limitations
2. Literature gap: Dataset quality and size constraints
3. Novel contribution: Two new datasets with complementary strengths
4. Rigorous validation: Comprehensive experiments against established baseline
5. Practical impact: Deployment recommendations, cost-benefit analysis
6. Future work: Integration into smart trap systems, multi-species extension

---

## 15. References and Baseline Metrics

### Literature Baseline (Checola et al., 2024)

**YOLOv8s Performance on LIT dataset:**
- mAP@0.5: 0.92 ± 0.02
- mAP@0.75: 0.81 ± 0.03
- mAP@[0.5:0.95]: 0.66 ± 0.02
- Precision: 0.89 ± 0.05
- Recall: 0.89 ± 0.03
- F1-score: 0.89 ± 0.03

**Dataset characteristics:**
- 615 images
- 1329 ST annotations
- 1506 OI annotations (note: current work focuses on ST only)
- Mixed acquisition sources
- 10-fold CV used in literature (we use 5-fold for efficiency)

**Key findings to build upon:**
- Data augmentation significantly improves performance
- Higher resolution (1280 vs 640) improves mAP by ~6-10%
- Image enhancement (brightness/contrast) has minimal effect on scanned images
- YOLOv8 outperforms Faster R-CNN

---

## 16. Appendix: Detailed Configurations

### A. Data Configuration Files

```yaml
# data/hi-res/data.yaml
path: ../data/hi-res
train: images/train
val: images/val
test: images/test

nc: 1  # number of classes
names: ['Scaphoideus_titanus']

# Image properties
imgsz: 1024
resolution: high  # ~26.6 µm/pixel
acquisition: controlled
```

```yaml
# data/field-lr/data.yaml
path: ../data/field-lr
train: images/train
val: images/val
test: images/test

nc: 1
names: ['Scaphoideus_titanus']

imgsz: 1024
resolution: variable  # smartphone-dependent
acquisition: field
```

```yaml
# data/literature/data.yaml
path: ../data/literature
train: images/train
val: images/val
test: images/test

nc: 1  # ST only (ignoring OI for now)
names: ['Scaphoideus_titanus']

imgsz: 1024
resolution: mixed
acquisition: heterogeneous
```

### B. Training Configuration Template

```yaml
# configs/train_base.yaml
model: yolov8s.pt
data: data/hi-res/data.yaml
epochs: 100
patience: 20  # early stopping
batch: 16
imgsz: 1024
device: 0  # GPU ID

# Optimization
optimizer: SGD
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005

# Augmentation (YOLO defaults)
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0

# Other
workers: 8
seed: 42
deterministic: True
save_period: 10
```

### C. Evaluation Configuration

```yaml
# configs/eval_base.yaml
data: data/hi-res/data.yaml
split: test
imgsz: 1024
batch: 32
conf: 0.001  # low threshold for high recall
iou: 0.6  # NMS IoU threshold
max_det: 300  # max detections per image

# Metrics
save_json: True  # COCO format results
save_hybrid: True  # NMS confidence + IoU
plots: True
```

---

## 17. Conclusion

This research plan provides a comprehensive, rigorous framework for validating the novel HI-RES and FIELD-LR datasets against the established literature baseline. The experimental design ensures:

1. **Reproducibility**: Clear protocols, statistical rigor, version control
2. **Fair comparison**: Identical metrics, matched conditions where appropriate
3. **Comprehensive evaluation**: Multiple perspectives (resolution, generalization, combination)
4. **Practical impact**: Deployment considerations, cost-benefit analysis
5. **Scientific rigor**: Statistical testing, ablation studies, error analysis

The plan is designed to produce convincing evidence for the contribution's value while maintaining scientific integrity and enabling future researchers to build upon this work. All experiments are feasible with standard Python deep learning tools (Ultralytics YOLO), and the modular design allows for adaptive refinement based on preliminary results.

**Key Deliverables:**
- Two novel datasets (HI-RES and FIELD-LR)
- Comprehensive benchmark against literature
- Trained models and evaluation code
- Clear deployment recommendations
- High-impact publication

**Timeline**: 14 weeks from data finalization to manuscript completion

**Success Metric**: Significant, reproducible improvement over state-of-the-art with clear practical pathway for deployment in FD monitoring systems.
