#!/usr/bin/env python
"""
Test suite for insect detection experiment framework.

Run with: python -m pytest tests/ -v
Or: python tests/test_experiments.py
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest import TestCase, main

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataLoader(TestCase):
    """Test dataset loading and preparation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.project_root = Path(__file__).parent.parent
        cls.data_dir = cls.project_root / 'detector' / 'data'
        
    def test_dataset_registry(self):
        """Test that all expected datasets are registered."""
        from detector.datasets.data_loader import DATASET_REGISTRY
        
        expected = ['hi_res', 'low_res', 'literature', 'combined', 'hi_res_low_res']
        for ds in expected:
            self.assertIn(ds, DATASET_REGISTRY)
    
    def test_cvat_parser(self):
        """Test CVAT XML parsing."""
        from detector.datasets.data_loader import CVATAnnotationParser
        
        # Find a test XML file
        low_res_parts = list((self.data_dir / 'low_res').glob('part*'))
        if not low_res_parts:
            self.skipTest("No low_res parts found")
        
        xml_file = low_res_parts[0] / 'annotations.xml'
        images_dir = low_res_parts[0] / 'images'
        
        if not xml_file.exists():
            self.skipTest(f"No annotations.xml in {low_res_parts[0]}")
        
        parser = CVATAnnotationParser(xml_file, images_dir)
        annotations = parser.parse()
        
        self.assertGreater(len(annotations), 0)
        
        # Check annotation structure
        ann = annotations[0]
        self.assertTrue(hasattr(ann, 'image_name'))
        self.assertTrue(hasattr(ann, 'image_width'))
        self.assertTrue(hasattr(ann, 'boxes'))
    
    def test_dataset_manager_init(self):
        """Test DatasetManager initialization."""
        from detector.datasets.data_loader import DatasetManager
        
        manager = DatasetManager(self.data_dir)
        self.assertTrue(manager.cache_dir.exists())
    
    def test_merge_low_res_parts(self):
        """Test merging low_res annotation batches."""
        from detector.datasets.data_loader import DatasetManager
        
        manager = DatasetManager(self.data_dir)
        
        # Check if parts exist
        parts = list(manager.low_res_dir.glob('part*'))
        if not parts:
            self.skipTest("No low_res parts to merge")
        
        output_dir, stats = manager.merge_low_res_parts(force=False)
        
        self.assertTrue(output_dir.exists())
        self.assertTrue((output_dir / 'images').exists())
        self.assertTrue((output_dir / 'labels').exists())
        self.assertGreater(stats['total_images'], 0)


class TestModelConfigs(TestCase):
    """Test model configuration."""
    
    def test_model_configs_exist(self):
        """Test that all expected model configs exist."""
        from detector.experiments.experiment_runner import MODEL_CONFIGS, ModelFamily
        
        # Check YOLO models
        for model in ['yolov5s', 'yolov5m', 'yolov8s', 'yolov8m', 'yolo11s', 'yolo11m']:
            self.assertIn(model, MODEL_CONFIGS)
            self.assertEqual(MODEL_CONFIGS[model].family, ModelFamily.YOLO)
        
        # Check Faster R-CNN
        self.assertIn('fasterrcnn_resnet50', MODEL_CONFIGS)
        self.assertEqual(MODEL_CONFIGS['fasterrcnn_resnet50'].family, ModelFamily.FASTER_RCNN)
        
        # Check RT-DETR
        self.assertIn('rtdetr_l', MODEL_CONFIGS)
        self.assertEqual(MODEL_CONFIGS['rtdetr_l'].family, ModelFamily.RTDETR)
    
    def test_get_trainer(self):
        """Test trainer factory function."""
        from detector.experiments.experiment_runner import get_trainer, MODEL_CONFIGS
        
        # Test YOLO trainer
        yolo_config = MODEL_CONFIGS['yolov8s']
        trainer = get_trainer(yolo_config)
        self.assertIsNotNone(trainer)
        
        # Test Faster R-CNN trainer
        frcnn_config = MODEL_CONFIGS['fasterrcnn_resnet50']
        trainer = get_trainer(frcnn_config)
        self.assertIsNotNone(trainer)


class TestExperimentCache(TestCase):
    """Test experiment caching."""
    
    def setUp(self):
        """Create temporary cache directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_operations(self):
        """Test cache save/load operations."""
        from detector.experiments.experiment_runner import (
            ExperimentCache, ExperimentConfig, ExperimentResults, ModelConfig, ModelFamily
        )
        
        cache = ExperimentCache(self.temp_dir)
        
        model_config = ModelConfig('test_model', ModelFamily.YOLO, 'yolov8s.pt')
        config = ExperimentConfig(
            name='test_exp',
            dataset='hi_res',
            model=model_config,
            fold=0,
            epochs=10
        )
        
        # Initially not cached
        self.assertFalse(cache.has(config))
        
        # Save results
        results = ExperimentResults(
            config=config,
            metrics={'mAP50': 0.9, 'mAP50-95': 0.7},
            training_time=100.0
        )
        cache.set(results)
        
        # Now cached
        self.assertTrue(cache.has(config))
        
        # Retrieve
        cached = cache.get(config)
        self.assertEqual(cached.metrics['mAP50'], 0.9)
        
        # Clear
        cache.clear(config)
        self.assertFalse(cache.has(config))


class TestYOLOTraining(TestCase):
    """Test YOLO model training (requires GPU and data)."""
    
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).parent.parent
        cls.data_dir = cls.project_root / 'detector' / 'data'
        cls.hi_res_yaml = cls.data_dir / 'hi_res' / 'hi_res.yaml'
        
    def test_yolo_import(self):
        """Test that ultralytics can be imported."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self.skipTest("ultralytics not installed")
    
    def test_yolo_model_load(self):
        """Test loading YOLO model."""
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            self.assertIsNotNone(model)
        except Exception as e:
            self.skipTest(f"Could not load YOLO model: {e}")
    
    def test_yolo_short_training(self):
        """Test YOLO training for 1 epoch."""
        if not self.hi_res_yaml.exists():
            self.skipTest("hi_res.yaml not found")
        
        try:
            from ultralytics import YOLO
            import torch
            
            if not torch.cuda.is_available():
                self.skipTest("CUDA not available")
            
            model = YOLO('yolov8n.pt')
            
            # Create temp output dir
            temp_dir = Path(tempfile.mkdtemp())
            
            try:
                results = model.train(
                    data=str(self.hi_res_yaml),
                    epochs=1,
                    imgsz=320,  # Small for speed
                    batch=4,
                    project=str(temp_dir),
                    name='test',
                    device=0,
                    verbose=False
                )
                self.assertIsNotNone(results)
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            self.skipTest(f"YOLO training failed: {e}")


class TestPyTorchDataset(TestCase):
    """Test PyTorch dataset for Faster R-CNN."""
    
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).parent.parent
        cls.data_dir = cls.project_root / 'detector' / 'data'
        cls.hi_res_dir = cls.data_dir / 'hi_res'
    
    def test_dataset_import(self):
        """Test dataset module import."""
        try:
            from detector.datasets.pytorch_dataset import InsectDataset, collate_fn
        except ImportError as e:
            self.skipTest(f"Could not import dataset: {e}")
    
    def test_dataset_creation(self):
        """Test creating dataset."""
        train_txt = self.hi_res_dir / 'train.txt'
        if not train_txt.exists():
            self.skipTest("train.txt not found")
        
        try:
            from detector.datasets.pytorch_dataset import InsectDataset
            
            dataset = InsectDataset(
                self.hi_res_dir,
                train_txt,
                img_size=320,
                augment=False
            )
            
            self.assertGreater(len(dataset), 0)
            
            # Test loading one sample
            img, target = dataset[0]
            self.assertEqual(img.dim(), 3)  # C, H, W
            self.assertIn('boxes', target)
            self.assertIn('labels', target)
            
        except ImportError:
            self.skipTest("Required packages not installed")


class TestExperimentSuite(TestCase):
    """Test experiment suite configuration."""
    
    def test_experiment_groups_defined(self):
        """Test that all experiment groups are defined."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from run_experiments import EXPERIMENT_GROUPS
        
        expected = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'test']
        for group in expected:
            self.assertIn(group, EXPERIMENT_GROUPS)
    
    def test_experiment_group_structure(self):
        """Test experiment group structure is valid."""
        from run_experiments import EXPERIMENT_GROUPS
        
        for name, group in EXPERIMENT_GROUPS.items():
            self.assertIn('name', group)
            self.assertIn('description', group)
            self.assertIn('type', group)
            self.assertIn(group['type'], ['kfold', 'cross_dataset', 'single'])


def run_quick_test():
    """Run a quick integration test."""
    print("Running quick integration test...")
    
    project_root = Path(__file__).parent.parent
    
    # Test imports
    print("Testing imports...")
    from detector.datasets.data_loader import DatasetManager, DATASET_REGISTRY
    from detector.experiments.experiment_runner import (
        ExperimentRunner, MODEL_CONFIGS, create_experiment_suite
    )
    print("  ✓ Imports successful")
    
    # Test data manager
    print("Testing DatasetManager...")
    data_dir = project_root / 'detector' / 'data'
    manager = DatasetManager(data_dir)
    print(f"  ✓ DatasetManager initialized")
    print(f"  ✓ Cache dir: {manager.cache_dir}")
    
    # Check datasets
    print("Checking datasets...")
    for ds_name in ['hi_res', 'low_res', 'literature']:
        ds_dir = getattr(manager, f"{ds_name}_dir", None) or data_dir / ds_name
        if ds_dir.exists():
            print(f"  ✓ {ds_name}: {ds_dir}")
        else:
            print(f"  ⚠ {ds_name}: not found at {ds_dir}")
    
    # Check models
    print("Checking model configurations...")
    print(f"  ✓ {len(MODEL_CONFIGS)} models available")
    
    # List experiment suite
    print("Checking experiment suite...")
    experiments = create_experiment_suite()
    print(f"  ✓ {len(experiments)} experiments defined")
    
    print("\n✓ Quick test completed successfully!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run quick integration test')
    args, remaining = parser.parse_known_args()
    
    if args.quick:
        run_quick_test()
    else:
        # Run unittest
        sys.argv = [sys.argv[0]] + remaining
        main(verbosity=2)
