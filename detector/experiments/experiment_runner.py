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


class MetricsCalculator:
    """Calculate detection metrics using sklearn for reliability."""
    
    @staticmethod
    def compute_detection_metrics(
        predictions: List[Dict],
        ground_truths: List[Dict],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute detection metrics from predictions and ground truths.
        
        Args:
            predictions: List of dicts with 'boxes', 'scores', 'labels'
            ground_truths: List of dicts with 'boxes', 'labels'
            iou_threshold: IoU threshold for matching
            
        Returns:
            Dictionary with precision, recall, f1 metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        all_y_true = []
        all_y_pred = []
        
        for preds, gts in zip(predictions, ground_truths):
            pred_boxes = preds.get('boxes', [])
            pred_scores = preds.get('scores', [])
            gt_boxes = gts.get('boxes', [])
            
            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
            
            matched_gt = set()
            
            if len(pred_boxes) > 0 and len(pred_scores) > 0:
                sorted_indices = np.argsort(pred_scores)[::-1]
                
                for idx in sorted_indices:
                    if idx >= len(pred_boxes):
                        continue
                    pred_box = pred_boxes[idx]
                    
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in matched_gt:
                            continue
                        iou = MetricsCalculator._compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= iou_threshold and best_gt_idx >= 0:
                        all_y_true.append(1)
                        all_y_pred.append(1)
                        matched_gt.add(best_gt_idx)
                    else:
                        all_y_true.append(0)
                        all_y_pred.append(1)
            
            for gt_idx in range(len(gt_boxes)):
                if gt_idx not in matched_gt:
                    all_y_true.append(1)
                    all_y_pred.append(0)
        
        if not all_y_true or not all_y_pred:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        precision = precision_score(all_y_true, all_y_pred, zero_division=0)
        recall = recall_score(all_y_true, all_y_pred, zero_division=0)
        f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    @staticmethod
    def _compute_iou(box1, box2) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        box1 = np.array(box1)
        box2 = np.array(box2)
        
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0


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
        import gc
        import torch
        
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
        
        # Extract metrics from training CSV
        metrics = self._extract_metrics(results)
        
        # Run validation to get mAP75 and update metrics
        best_weights = output_dir / 'weights' / 'best.pt'
        if best_weights.exists():
            val_metrics = self.validate(best_weights, data_yaml)
            metrics.update(val_metrics)
        
        del model
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return metrics
    
    def validate(self, model_path: Path, data_yaml: Path) -> Dict[str, float]:
        from ultralytics import YOLO
        import gc
        import torch
        
        model = YOLO(str(model_path))
        metrics = model.val(data=str(data_yaml), device=self.device, verbose=False)
        
        result = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }
        
        # Extract mAP75
        if hasattr(metrics.box, 'map75'):
            result['mAP75'] = float(metrics.box.map75)
        elif hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 5:
            result['mAP75'] = float(metrics.box.maps[5])
        else:
            result['mAP75'] = 0.0
        
        p, r = result['precision'], result['recall']
        result['f1'] = 2 * p * r / (p + r + 1e-8)
        
        del model
        del metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
    
    def _extract_metrics(self, results) -> Dict[str, float]:
        """Extract metrics from YOLO training results CSV."""
        try:
            results_dir = Path(results.save_dir)
            results_csv = results_dir / 'results.csv'
            
            if results_csv.exists():
                df = pd.read_csv(results_csv)
                df.columns = df.columns.str.strip()
                
                # Get best epoch (highest mAP50) instead of last
                if 'metrics/mAP50(B)' in df.columns:
                    best_idx = df['metrics/mAP50(B)'].idxmax()
                    best = df.iloc[best_idx]
                else:
                    best = df.iloc[-1]
                
                metrics = {
                    'mAP50': float(best.get('metrics/mAP50(B)', 0)),
                    'mAP50-95': float(best.get('metrics/mAP50-95(B)', 0)),
                    'precision': float(best.get('metrics/precision(B)', 0)),
                    'recall': float(best.get('metrics/recall(B)', 0)),
                }
                
                # Calculate F1 from precision and recall
                p, r = metrics['precision'], metrics['recall']
                metrics['f1'] = 2 * p * r / (p + r + 1e-8)
                
                # mAP75 not available in training CSV, will be added during validation
                metrics['mAP75'] = 0.0
                
                return metrics
        except Exception as e:
            print(f"Warning: Could not extract metrics from training: {e}")
        
        return {'mAP50': 0.0, 'mAP50-95': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mAP75': 0.0}


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
        
        torch.save(model.state_dict(), output_dir / 'last.pt')
        
        del model
        del optimizer
        del lr_scheduler
        del train_loader
        del val_loader
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return metrics
    
    def validate(self, model_path: Path, data_yaml: Path) -> Dict[str, float]:
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        import gc
        
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
        
        metrics = self._evaluate(model, val_loader, device)
        
        del model
        del val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return metrics
    
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
        """Evaluate Faster R-CNN model with sklearn metrics."""
        import torch
        from torchmetrics.detection import MeanAveragePrecision
        
        model.eval()
        map_metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
        
        all_preds = []
        all_gts = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                outputs = model(images)
                
                for out, tgt in zip(outputs, targets):
                    pred = {
                        'boxes': out['boxes'].cpu(),
                        'scores': out['scores'].cpu(),
                        'labels': out['labels'].cpu()
                    }
                    gt = {
                        'boxes': tgt['boxes'],
                        'labels': tgt['labels']
                    }
                    map_metric.update([pred], [gt])
                    all_preds.append(pred)
                    all_gts.append(gt)
        
        result = map_metric.compute()
        
        # Use sklearn-based metrics for precision/recall/f1
        sklearn_metrics = MetricsCalculator.compute_detection_metrics(
            all_preds, all_gts, iou_threshold=0.5
        )
        
        return {
            'mAP50': float(result['map_50']),
            'mAP75': float(result['map_75']),
            'mAP50-95': float(result['map']),
            'precision': sklearn_metrics['precision'],
            'recall': sklearn_metrics['recall'],
            'f1': sklearn_metrics['f1']
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
        import gc
        import torch
        
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
        
        metrics = YOLOTrainer._extract_metrics(self, results)
        
        best_weights = output_dir / 'weights' / 'best.pt'
        if best_weights.exists():
            val_metrics = self.validate(best_weights, data_yaml)
            metrics.update(val_metrics)
        
        del model
        del results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        return metrics
    
    def validate(self, model_path: Path, data_yaml: Path) -> Dict[str, float]:
        from ultralytics import RTDETR
        import gc
        import torch
        
        model = RTDETR(str(model_path))
        metrics = model.val(data=str(data_yaml), device=self.device, verbose=False)
        
        result = {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
        }
        
        if hasattr(metrics.box, 'map75'):
            result['mAP75'] = float(metrics.box.map75)
        elif hasattr(metrics.box, 'maps') and len(metrics.box.maps) > 5:
            result['mAP75'] = float(metrics.box.maps[5])
        else:
            result['mAP75'] = 0.0
        
        p, r = result['precision'], result['recall']
        result['f1'] = 2 * p * r / (p + r + 1e-8)
        
        del model
        del metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result


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
    'yolov5n': ModelConfig('yolov5n', ModelFamily.YOLO, 'yolov5nu.pt', batch_size=64),
    'yolov5s': ModelConfig('yolov5s', ModelFamily.YOLO, 'yolov5su.pt', batch_size=48),
    'yolov5m': ModelConfig('yolov5m', ModelFamily.YOLO, 'yolov5mu.pt', batch_size=32),
    
    # YOLO v8
    'yolov8n': ModelConfig('yolov8n', ModelFamily.YOLO, 'yolov8n.pt', batch_size=64),
    'yolov8s': ModelConfig('yolov8s', ModelFamily.YOLO, 'yolov8s.pt', batch_size=48),
    'yolov8m': ModelConfig('yolov8m', ModelFamily.YOLO, 'yolov8m.pt', batch_size=32),
    
    # YOLO v11
    'yolo11n': ModelConfig('yolo11n', ModelFamily.YOLO, 'yolo11n.pt', batch_size=64),
    'yolo11s': ModelConfig('yolo11s', ModelFamily.YOLO, 'yolo11s.pt', batch_size=48),
    'yolo11m': ModelConfig('yolo11m', ModelFamily.YOLO, 'yolo11m.pt', batch_size=32),
    
    # Faster R-CNN
    'fasterrcnn_resnet50': ModelConfig('fasterrcnn_resnet50', ModelFamily.FASTER_RCNN, 'fasterrcnn_resnet50_fpn_v2', batch_size=16),
    
    # RT-DETR
    'rtdetr_l': ModelConfig('rtdetr_l', ModelFamily.RTDETR, 'rtdetr-l.pt', batch_size=16),
    'rtdetr_x': ModelConfig('rtdetr_x', ModelFamily.RTDETR, 'rtdetr-x.pt', batch_size=12),
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
    
    def _create_fold_data_yaml(
        self,
        dataset_dir: Path,
        split: DatasetSplit,
        fold_dir: Path
    ) -> Path:
        """
        Create a YOLO data.yaml for a specific fold with ABSOLUTE paths.
        
        This is the key fix - we use absolute paths to image files so that
        each fold correctly uses its own train/val split regardless of
        where the data.yaml path field points.
        """
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        def resolve_path(img_path):
            """Resolve image path to absolute, handling various input formats."""
            path = Path(img_path)
            if path.is_absolute():
                return str(path)
            # Paths from DatasetManager are relative to workspace root
            # Just resolve them directly without prepending dataset_dir
            return str(path.absolute())
        
        # Write train.txt with absolute paths
        train_txt = fold_dir / 'train.txt'
        with open(train_txt, 'w') as f:
            for img_path in split.train:
                f.write(f"{resolve_path(img_path)}\n")
        
        # Write val.txt with absolute paths
        val_txt = fold_dir / 'val.txt'
        with open(val_txt, 'w') as f:
            for img_path in split.val:
                f.write(f"{resolve_path(img_path)}\n")
        
        # Write test.txt if available
        test_txt = None
        if split.test:
            test_txt = fold_dir / 'test.txt'
            with open(test_txt, 'w') as f:
                for img_path in split.test:
                    f.write(f"{resolve_path(img_path)}\n")
        
        # CRITICAL: Set path to fold_dir so YOLO reads our fold-specific txt files!
        data_yaml = fold_dir / 'data.yaml'
        config = {
            'path': str(fold_dir.absolute()),
            'train': 'train.txt',
            'val': 'val.txt',
            'nc': 1,
            'names': {0: 'ScafoideusTitanus'}
        }
        if test_txt:
            config['test'] = 'test.txt'
        
        with open(data_yaml, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return data_yaml
    
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
        
        # Create fold-specific data.yaml with ABSOLUTE paths
        # This is the critical fix - each fold gets its own correct data!
        fold_dir = self.output_dir / config.name / f'fold_{config.fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        data_yaml = self._create_fold_data_yaml(dataset_dir, split, fold_dir)
        
        # Log fold info for verification
        print(f"Fold {config.fold} data:")
        print(f"  Train: {len(split.train)} images")
        print(f"  Val: {len(split.val)} images")
        print(f"  Test: {len(split.test)} images")
        
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
            
            # Create test data yaml with ABSOLUTE paths (use first split's val as test)
            test_fold_dir = self.output_dir / name / 'test_eval'
            test_fold_dir.mkdir(parents=True, exist_ok=True)
            
            # For cross-dataset eval, use val split as test data
            test_split = test_splits[0]
            test_yaml = self._create_fold_data_yaml(
                test_dataset_dir, test_split, test_fold_dir
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
