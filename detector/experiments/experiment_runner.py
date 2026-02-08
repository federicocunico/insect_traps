"""
Experiment Framework for Insect Detection Research.

This module provides a comprehensive, cached experiment runner following
the research plan for S. titanus detection with multiple datasets and models.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

from detector.datasets.data_loader import DatasetManager, DatasetSplit, DATASET_REGISTRY


class ModelFamily(Enum):
    YOLO = "yolo"
    FASTER_RCNN = "faster_rcnn"
    DETR = "detr"
    RTDETR = "rtdetr"


@dataclass
class ModelConfig:
    """Configuration for a detection model."""
    name: str
    family: ModelFamily
    weights: str  # pretrained weights path or name
    img_size: int = 1024
    batch_size: int = 16
    extra_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    name: str
    dataset: str
    model: ModelConfig
    fold: int = 0
    epochs: int = 100
    patience: int = 20
    seed: int = 42
    device: Union[int, str] = 0
    
    def get_hash(self) -> str:
        """Get unique hash for this experiment configuration."""
        config_str = f"{self.name}_{self.dataset}_{self.model.name}_{self.model.img_size}_{self.fold}_{self.seed}"
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


@dataclass
class ExperimentResults:
    """Results from a single experiment."""
    config: ExperimentConfig
    metrics: Dict[str, float]
    training_time: float
    model_path: Optional[str] = None
    logs_path: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ExperimentCache:
    """Caching system for experiment results."""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'experiments_cache.json'
        self._load_cache()
    
    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file) as f:
                self._cache = json.load(f)
        else:
            self._cache = {}
    
    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self._cache, f, indent=2, default=str)
    
    def get(self, config: ExperimentConfig) -> Optional[ExperimentResults]:
        """Get cached results for an experiment configuration."""
        key = config.get_hash()
        if key in self._cache:
            data = self._cache[key]
            return ExperimentResults(
                config=config,
                metrics=data['metrics'],
                training_time=data['training_time'],
                model_path=data.get('model_path'),
                logs_path=data.get('logs_path'),
                timestamp=data.get('timestamp', '')
            )
        return None
    
    def set(self, results: ExperimentResults):
        """Cache experiment results."""
        key = results.config.get_hash()
        self._cache[key] = {
            'metrics': results.metrics,
            'training_time': results.training_time,
            'model_path': results.model_path,
            'logs_path': results.logs_path,
            'timestamp': results.timestamp
        }
        self._save_cache()
    
    def has(self, config: ExperimentConfig) -> bool:
        """Check if experiment is cached."""
        return config.get_hash() in self._cache
    
    def clear(self, config: ExperimentConfig = None):
        """Clear cache for specific config or all."""
        if config:
            key = config.get_hash()
            if key in self._cache:
                del self._cache[key]
        else:
            self._cache = {}
        self._save_cache()


class BaseModelTrainer:
    """Base class for model trainers."""
    
    def __init__(self, config: ModelConfig, device: Union[int, str] = 0):
        self.config = config
        self.device = device
    
    def train(
        self,
        data_yaml: Path,
        output_dir: Path,
        epochs: int = 100,
        patience: int = 20
    ) -> Dict[str, float]:
        """Train the model and return metrics."""
        raise NotImplementedError
    
    def validate(self, model_path: Path, data_yaml: Path) -> Dict[str, float]:
        """Validate model and return metrics."""
        raise NotImplementedError


class YOLOTrainer(BaseModelTrainer):
    """Trainer for YOLO models (v5, v8, v11)."""
    
    def train(
        self,
        data_yaml: Path,
        output_dir: Path,
        epochs: int = 100,
        patience: int = 20
    ) -> Dict[str, float]:
        from ultralytics import YOLO
        
        model = YOLO(self.config.weights)
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=self.config.img_size,
            batch=self.config.batch_size,
            project=str(output_dir.parent),
            name=output_dir.name,
            device=self.device,
            patience=patience,
            exist_ok=True,
            verbose=True,
            **self.config.extra_args
        )
        
        return self._extract_metrics(results)
    
    def validate(self, model_path: Path, data_yaml: Path) -> Dict[str, float]:
        from ultralytics import YOLO
        
        model = YOLO(str(model_path))
        metrics = model.val(data=str(data_yaml), device=self.device)
        
        return {
            'mAP50': float(metrics.box.map50),
            'mAP75': float(metrics.box.map75) if hasattr(metrics.box, 'map75') else 0.0,
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'f1': 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)
        }
    
    def _extract_metrics(self, results) -> Dict[str, float]:
        """Extract metrics from YOLO training results."""
        try:
            results_dir = Path(results.save_dir)
            results_csv = results_dir / 'results.csv'
            
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()
                last = df.iloc[-1]
                
                return {
                    'mAP50': float(last.get('metrics/mAP50(B)', 0)),
                    'mAP75': float(last.get('metrics/mAP75(B)', 0)) if 'metrics/mAP75(B)' in df.columns else 0.0,
                    'mAP50-95': float(last.get('metrics/mAP50-95(B)', 0)),
                    'precision': float(last.get('metrics/precision(B)', 0)),
                    'recall': float(last.get('metrics/recall(B)', 0)),
                    'f1': 0.0  # calculated later
                }
        except Exception as e:
            print(f"Warning: Could not extract metrics: {e}")
        
        return {'mAP50': 0.0, 'mAP50-95': 0.0, 'precision': 0.0, 'recall': 0.0}


class FasterRCNNTrainer(BaseModelTrainer):
    """Trainer for Faster R-CNN models using torchvision."""
    
    def train(
        self,
        data_yaml: Path,
        output_dir: Path,
        epochs: int = 100,
        patience: int = 20
    ) -> Dict[str, float]:
        import torch
        import torchvision
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data config
        with open(data_yaml) as f:
            data_config = yaml.safe_load(f)
        
        num_classes = data_config.get('nc', 1) + 1  # +1 for background
        
        # Create model
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        device = torch.device(f'cuda:{self.device}' if isinstance(self.device, int) else self.device)
        model.to(device)
        
        # Create data loaders
        train_loader, val_loader = self._create_dataloaders(data_config)
        
        # Training setup
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        best_map = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            
            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
            
            lr_scheduler.step()
            
            # Validation
            metrics = self._evaluate(model, val_loader, device)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - mAP50: {metrics['mAP50']:.4f}")
            
            # Early stopping
            if metrics['mAP50'] > best_map:
                best_map = metrics['mAP50']
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / 'best.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Save final model
        torch.save(model.state_dict(), output_dir / 'last.pt')
        
        return metrics
    
    def validate(self, model_path: Path, data_yaml: Path) -> Dict[str, float]:
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        
        with open(data_yaml) as f:
            data_config = yaml.safe_load(f)
        
        num_classes = data_config.get('nc', 1) + 1
        
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model.load_state_dict(torch.load(model_path))
        
        device = torch.device(f'cuda:{self.device}' if isinstance(self.device, int) else self.device)
        model.to(device)
        
        _, val_loader = self._create_dataloaders(data_config, val_only=True)
        
        return self._evaluate(model, val_loader, device)
    
    def _create_dataloaders(self, data_config: dict, val_only: bool = False):
        """Create PyTorch dataloaders for Faster R-CNN."""
        from torch.utils.data import DataLoader
        from detector.datasets.pytorch_dataset import InsectDataset, collate_fn
        
        data_path = Path(data_config['path'])
        
        if not val_only:
            train_txt = data_path / data_config.get('train', 'train.txt')
            train_dataset = InsectDataset(
                data_path, train_txt, img_size=self.config.img_size, augment=True
            )
            train_loader = DataLoader(
                train_dataset, batch_size=self.config.batch_size,
                shuffle=True, collate_fn=collate_fn, num_workers=4
            )
        else:
            train_loader = None
        
        val_txt = data_path / data_config.get('val', 'val.txt')
        val_dataset = InsectDataset(
            data_path, val_txt, img_size=self.config.img_size, augment=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, collate_fn=collate_fn, num_workers=4
        )
        
        return train_loader, val_loader
    
    def _evaluate(self, model, data_loader, device) -> Dict[str, float]:
        """Evaluate Faster R-CNN model."""
        import torch
        from torchmetrics.detection import MeanAveragePrecision
        
        model.eval()
        map_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                
                preds = []
                gts = []
                for out, tgt in zip(outputs, targets):
                    preds.append({
                        'boxes': out['boxes'].cpu(),
                        'scores': out['scores'].cpu(),
                        'labels': out['labels'].cpu()
                    })
                    gts.append({
                        'boxes': tgt['boxes'],
                        'labels': tgt['labels']
                    })
                
                map_metric.update(preds, gts)
        
        result = map_metric.compute()
        
        return {
            'mAP50': float(result['map_50']),
            'mAP75': float(result['map_75']),
            'mAP50-95': float(result['map']),
            'precision': 0.0,  # Not directly available
            'recall': 0.0
        }


class RTDETRTrainer(BaseModelTrainer):
    """Trainer for RT-DETR models (via Ultralytics)."""
    
    def train(
        self,
        data_yaml: Path,
        output_dir: Path,
        epochs: int = 100,
        patience: int = 20
    ) -> Dict[str, float]:
        from ultralytics import RTDETR
        
        model = RTDETR(self.config.weights)
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=self.config.img_size,
            batch=self.config.batch_size,
            project=str(output_dir.parent),
            name=output_dir.name,
            device=self.device,
            patience=patience,
            exist_ok=True,
            **self.config.extra_args
        )
        
        return YOLOTrainer._extract_metrics(self, results)
    
    def validate(self, model_path: Path, data_yaml: Path) -> Dict[str, float]:
        from ultralytics import RTDETR
        
        model = RTDETR(str(model_path))
        metrics = model.val(data=str(data_yaml), device=self.device)
        
        return {
            'mAP50': float(metrics.box.map50),
            'mAP75': float(metrics.box.map75) if hasattr(metrics.box, 'map75') else 0.0,
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'f1': 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr + 1e-8)
        }


def get_trainer(config: ModelConfig, device: Union[int, str] = 0) -> BaseModelTrainer:
    """Factory function to get appropriate trainer for model family."""
    if config.family == ModelFamily.YOLO:
        return YOLOTrainer(config, device)
    elif config.family == ModelFamily.FASTER_RCNN:
        return FasterRCNNTrainer(config, device)
    elif config.family in (ModelFamily.DETR, ModelFamily.RTDETR):
        return RTDETRTrainer(config, device)
    else:
        raise ValueError(f"Unknown model family: {config.family}")


# Pre-defined model configurations
MODEL_CONFIGS = {
    # YOLO v5
    'yolov5n': ModelConfig('yolov5n', ModelFamily.YOLO, 'yolov5nu.pt'),
    'yolov5s': ModelConfig('yolov5s', ModelFamily.YOLO, 'yolov5su.pt'),
    'yolov5m': ModelConfig('yolov5m', ModelFamily.YOLO, 'yolov5mu.pt'),
    
    # YOLO v8
    'yolov8n': ModelConfig('yolov8n', ModelFamily.YOLO, 'yolov8n.pt'),
    'yolov8s': ModelConfig('yolov8s', ModelFamily.YOLO, 'yolov8s.pt'),
    'yolov8m': ModelConfig('yolov8m', ModelFamily.YOLO, 'yolov8m.pt'),
    
    # YOLO v11
    'yolo11n': ModelConfig('yolo11n', ModelFamily.YOLO, 'yolo11n.pt'),
    'yolo11s': ModelConfig('yolo11s', ModelFamily.YOLO, 'yolo11s.pt'),
    'yolo11m': ModelConfig('yolo11m', ModelFamily.YOLO, 'yolo11m.pt'),
    
    # Faster R-CNN
    'fasterrcnn_resnet50': ModelConfig('fasterrcnn_resnet50', ModelFamily.FASTER_RCNN, 'fasterrcnn_resnet50_fpn_v2'),
    
    # RT-DETR
    'rtdetr_l': ModelConfig('rtdetr_l', ModelFamily.RTDETR, 'rtdetr-l.pt'),
    'rtdetr_x': ModelConfig('rtdetr_x', ModelFamily.RTDETR, 'rtdetr-x.pt'),
}


class ExperimentRunner:
    """Main experiment runner with caching support."""
    
    def __init__(
        self,
        base_dir: Path = Path('detector'),
        output_dir: Path = Path('runs/experiments'),
        device: Union[int, str] = 0
    ):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.data_manager = DatasetManager(self.base_dir / 'data')
        self.cache = ExperimentCache(self.output_dir / '.cache')
        
        self.results_df = None
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        force: bool = False
    ) -> ExperimentResults:
        """Run a single experiment."""
        
        # Check cache
        if not force and self.cache.has(config):
            print(f"[CACHED] {config.name} - {config.model.name} - fold {config.fold}")
            return self.cache.get(config)
        
        print(f"\n{'='*80}")
        print(f"Running: {config.name}")
        print(f"Model: {config.model.name} | Dataset: {config.dataset} | Fold: {config.fold}")
        print(f"{'='*80}\n")
        
        # Prepare dataset
        dataset_dir, splits = self.data_manager.prepare_dataset(
            config.dataset,
            n_folds=5,
            seed=config.seed
        )
        
        split = splits[config.fold]
        
        # Create fold-specific yaml
        fold_dir = self.output_dir / config.name / f'fold_{config.fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        data_yaml = fold_dir / 'data.yaml'
        self.data_manager.create_fold_yaml(dataset_dir, split, data_yaml)
        
        # Get trainer
        trainer = get_trainer(config.model, self.device)
        
        # Train
        start_time = time.time()
        metrics = trainer.train(
            data_yaml=data_yaml,
            output_dir=fold_dir / 'train',
            epochs=config.epochs,
            patience=config.patience
        )
        training_time = time.time() - start_time
        
        # Find best model path
        model_path = None
        for candidate in [
            fold_dir / 'train' / 'weights' / 'best.pt',
            fold_dir / 'train' / 'best.pt'
        ]:
            if candidate.exists():
                model_path = str(candidate)
                break
        
        results = ExperimentResults(
            config=config,
            metrics=metrics,
            training_time=training_time,
            model_path=model_path,
            logs_path=str(fold_dir)
        )
        
        # Cache results
        self.cache.set(results)
        
        return results
    
    def run_kfold_experiment(
        self,
        name: str,
        dataset: str,
        model_name: str,
        n_folds: int = 5,
        epochs: int = 100,
        img_size: int = 1024,
        seed: int = 42,
        force: bool = False
    ) -> List[ExperimentResults]:
        """Run k-fold cross-validation experiment."""
        
        model_config = MODEL_CONFIGS.get(model_name)
        if model_config is None:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        
        model_config.img_size = img_size
        
        all_results = []
        
        for fold in range(n_folds):
            config = ExperimentConfig(
                name=f"{name}_fold{fold}",
                dataset=dataset,
                model=model_config,
                fold=fold,
                epochs=epochs,
                seed=seed,
                device=self.device
            )
            
            results = self.run_experiment(config, force=force)
            all_results.append(results)
        
        return all_results
    
    def run_cross_dataset_eval(
        self,
        train_dataset: str,
        test_dataset: str,
        model_name: str,
        epochs: int = 100,
        img_size: int = 1024,
        seed: int = 42,
        force: bool = False
    ) -> ExperimentResults:
        """Train on one dataset, evaluate on another."""
        
        name = f"cross_{train_dataset}_to_{test_dataset}_{model_name}"
        
        model_config = MODEL_CONFIGS.get(model_name)
        if model_config is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config.img_size = img_size
        
        config = ExperimentConfig(
            name=name,
            dataset=train_dataset,
            model=model_config,
            fold=0,  # Use fold 0 for cross-dataset
            epochs=epochs,
            seed=seed,
            device=self.device
        )
        
        # Run training
        results = self.run_experiment(config, force=force)
        
        if results.model_path:
            # Evaluate on test dataset
            test_dataset_dir, test_splits = self.data_manager.prepare_dataset(
                test_dataset, seed=seed
            )
            
            test_yaml = self.output_dir / name / 'test_data.yaml'
            self.data_manager.create_fold_yaml(
                test_dataset_dir, test_splits[0], test_yaml
            )
            
            trainer = get_trainer(model_config, self.device)
            cross_metrics = trainer.validate(Path(results.model_path), test_yaml)
            
            # Add cross-dataset metrics
            results.metrics['cross_mAP50'] = cross_metrics['mAP50']
            results.metrics['cross_mAP50-95'] = cross_metrics['mAP50-95']
            
            self.cache.set(results)
        
        return results
    
    def aggregate_kfold_results(
        self,
        results: List[ExperimentResults]
    ) -> Dict[str, Tuple[float, float]]:
        """Aggregate k-fold results to mean ± std."""
        
        metrics_keys = list(results[0].metrics.keys())
        
        aggregated = {}
        for key in metrics_keys:
            values = [r.metrics.get(key, 0) for r in results]
            aggregated[key] = (np.mean(values), np.std(values))
        
        return aggregated
    
    def results_to_dataframe(
        self,
        results: List[ExperimentResults]
    ) -> pd.DataFrame:
        """Convert results list to pandas DataFrame."""
        
        rows = []
        for r in results:
            row = {
                'experiment': r.config.name,
                'dataset': r.config.dataset,
                'model': r.config.model.name,
                'fold': r.config.fold,
                'img_size': r.config.model.img_size,
                'training_time': r.training_time,
            }
            row.update(r.metrics)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_results_csv(self, results: List[ExperimentResults], output_path: Path):
        """Save results to CSV."""
        df = self.results_to_dataframe(results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


def create_experiment_suite() -> Dict[str, List[ExperimentConfig]]:
    """Create the full experiment suite based on research plan."""
    
    experiments = {}
    
    # Experiment 1: Intra-Dataset Baseline (5-fold CV)
    for dataset in ['hi_res', 'low_res', 'literature']:
        for model in ['yolov5s', 'yolov5m', 'yolov8s', 'yolov8m', 'yolo11s', 'yolo11m']:
            name = f"exp1_{dataset}_{model}"
            experiments[name] = {
                'type': 'kfold',
                'dataset': dataset,
                'model': model,
                'n_folds': 5
            }
    
    # Experiment 2: Resolution Impact
    for dataset in ['hi_res', 'low_res', 'literature']:
        for img_size in [512, 640, 768, 1024]:
            for model in ['yolov8s', 'yolo11s']:
                name = f"exp2_{dataset}_{model}_img{img_size}"
                experiments[name] = {
                    'type': 'kfold',
                    'dataset': dataset,
                    'model': model,
                    'img_size': img_size,
                    'n_folds': 5
                }
    
    # Experiment 3: Cross-Dataset Generalization
    cross_pairs = [
        ('hi_res', 'literature'),
        ('hi_res', 'low_res'),
        ('low_res', 'literature'),
        ('low_res', 'hi_res'),
        ('literature', 'hi_res'),
        ('literature', 'low_res'),
    ]
    for train_ds, test_ds in cross_pairs:
        for model in ['yolov8s', 'yolo11s']:
            name = f"exp3_{train_ds}_to_{test_ds}_{model}"
            experiments[name] = {
                'type': 'cross_dataset',
                'train_dataset': train_ds,
                'test_dataset': test_ds,
                'model': model
            }
    
    # Experiment 4: Dataset Combination
    for combo in ['hi_res_low_res', 'combined']:
        for model in ['yolov8s', 'yolo11s']:
            name = f"exp4_{combo}_{model}"
            experiments[name] = {
                'type': 'kfold',
                'dataset': combo,
                'model': model,
                'n_folds': 5
            }
    
    return experiments


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run insect detection experiments')
    parser.add_argument('--experiment', type=str, help='Specific experiment to run')
    parser.add_argument('--list', action='store_true', help='List available experiments')
    parser.add_argument('--force', action='store_true', help='Force re-run cached experiments')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    
    args = parser.parse_args()
    
    experiments = create_experiment_suite()
    
    if args.list:
        print("Available experiments:")
        for name in sorted(experiments.keys()):
            print(f"  {name}")
        exit(0)
    
    runner = ExperimentRunner(device=args.device)
    
    if args.experiment:
        if args.experiment not in experiments:
            print(f"Unknown experiment: {args.experiment}")
            exit(1)
        
        exp = experiments[args.experiment]
        
        if exp['type'] == 'kfold':
            results = runner.run_kfold_experiment(
                name=args.experiment,
                dataset=exp['dataset'],
                model_name=exp['model'],
                n_folds=exp.get('n_folds', 5),
                img_size=exp.get('img_size', 1024),
                epochs=args.epochs,
                force=args.force
            )
            
            agg = runner.aggregate_kfold_results(results)
            print("\nAggregated Results:")
            for k, (mean, std) in agg.items():
                print(f"  {k}: {mean:.4f} ± {std:.4f}")
        
        elif exp['type'] == 'cross_dataset':
            results = runner.run_cross_dataset_eval(
                train_dataset=exp['train_dataset'],
                test_dataset=exp['test_dataset'],
                model_name=exp['model'],
                epochs=args.epochs,
                force=args.force
            )
            
            print("\nResults:")
            for k, v in results.metrics.items():
                print(f"  {k}: {v:.4f}")
    else:
        print("Please specify an experiment with --experiment or use --list")
