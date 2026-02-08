#!/usr/bin/env python
"""
Main script to run all insect detection experiments.

This script implements the research plan for S. titanus detection
across multiple datasets and model architectures.

Usage:
    # Run all experiments
    python run_experiments.py --all
    
    # Run specific experiment group
    python run_experiments.py --group exp1  # Intra-dataset baselines
    python run_experiments.py --group exp2  # Resolution analysis
    python run_experiments.py --group exp3  # Cross-dataset
    python run_experiments.py --group exp4  # Dataset combinations
    
    # Run single experiment
    python run_experiments.py --experiment exp1_hi_res_yolov8s
    
    # Run quick test (2 epochs)
    python run_experiments.py --test
    
    # List available experiments
    python run_experiments.py --list
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from detector.experiments.experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    ModelConfig,
    ModelFamily,
    MODEL_CONFIGS,
    create_experiment_suite,
)
from detector.datasets.data_loader import DatasetManager, DATASET_REGISTRY


# Experiment Groups
EXPERIMENT_GROUPS = {
    'exp1': {
        'name': 'Intra-Dataset Baseline Performance',
        'description': '5-fold CV on each dataset independently',
        'datasets': ['hi_res', 'low_res', 'literature'],
        'models': ['yolov5s', 'yolov5m', 'yolov8s', 'yolov8m', 'yolo11s', 'yolo11m'],
        'type': 'kfold',
        'n_folds': 5,
        'img_size': 1024,
    },
    'exp2': {
        'name': 'Resolution Impact Analysis',
        'description': 'Test multiple image sizes',
        'datasets': ['hi_res', 'low_res', 'literature'],
        'models': ['yolov8s', 'yolo11s'],
        'img_sizes': [512, 640, 768, 1024],
        'type': 'kfold',
        'n_folds': 5,
    },
    'exp3': {
        'name': 'Cross-Dataset Generalization',
        'description': 'Train on one dataset, test on another',
        'pairs': [
            ('hi_res', 'literature'),
            ('hi_res', 'low_res'),
            ('low_res', 'literature'),
            ('low_res', 'hi_res'),
            ('literature', 'hi_res'),
            ('literature', 'low_res'),
        ],
        'models': ['yolov8s', 'yolo11s'],
        'type': 'cross_dataset',
        'img_size': 1024,
    },
    'exp4': {
        'name': 'Dataset Combination Strategies',
        'description': 'Combined dataset training',
        'datasets': ['hi_res_low_res', 'combined'],
        'models': ['yolov8s', 'yolo11s'],
        'type': 'kfold',
        'n_folds': 5,
        'img_size': 1024,
    },
    'exp5': {
        'name': 'Alternative Models',
        'description': 'Non-YOLO model architectures',
        'datasets': ['hi_res', 'low_res'],
        'models': ['fasterrcnn_resnet50', 'rtdetr_l'],
        'type': 'kfold',
        'n_folds': 3,  # Fewer folds due to training time
        'img_size': 640,  # Smaller for memory
    },
    'test': {
        'name': 'Quick Test',
        'description': 'Verify all models train correctly',
        'datasets': ['hi_res'],
        'models': ['yolov8s', 'yolo11s', 'fasterrcnn_resnet50'],
        'type': 'single',
        'epochs': 2,
        'img_size': 640,
    },
}


class ExperimentSuite:
    """Manages running groups of experiments."""
    
    def __init__(
        self,
        output_dir: Path = Path('runs/experiments'),
        device: int = 0,
        epochs: int = 100,
        patience: int = 20,
        seed: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        
        self.runner = ExperimentRunner(
            output_dir=self.output_dir,
            device=device
        )
        
        self.results = []
        self.start_time = None
    
    def run_group(self, group_name: str, force: bool = False) -> List[Dict]:
        """Run a group of experiments."""
        if group_name not in EXPERIMENT_GROUPS:
            raise ValueError(f"Unknown group: {group_name}")
        
        group = EXPERIMENT_GROUPS[group_name]
        print(f"\n{'#'*80}")
        print(f"# {group['name']}")
        print(f"# {group['description']}")
        print(f"{'#'*80}\n")
        
        self.start_time = time.time()
        results = []
        
        if group['type'] == 'kfold':
            results = self._run_kfold_group(group_name, group, force)
        elif group['type'] == 'cross_dataset':
            results = self._run_cross_dataset_group(group_name, group, force)
        elif group['type'] == 'single':
            results = self._run_single_experiments(group_name, group, force)
        
        elapsed = time.time() - self.start_time
        print(f"\n{'='*80}")
        print(f"Group {group_name} completed in {elapsed/3600:.2f} hours")
        print(f"{'='*80}\n")
        
        return results
    
    def _run_kfold_group(self, group_name: str, group: Dict, force: bool) -> List[Dict]:
        """Run k-fold experiments for a group."""
        results = []
        
        for dataset in group['datasets']:
            for model in group['models']:
                img_sizes = group.get('img_sizes', [group.get('img_size', 1024)])
                
                for img_size in img_sizes:
                    exp_name = f"{group_name}_{dataset}_{model}"
                    if len(img_sizes) > 1:
                        exp_name += f"_img{img_size}"
                    
                    try:
                        fold_results = self.runner.run_kfold_experiment(
                            name=exp_name,
                            dataset=dataset,
                            model_name=model,
                            n_folds=group.get('n_folds', 5),
                            epochs=group.get('epochs', self.epochs),
                            img_size=img_size,
                            seed=self.seed,
                            force=force
                        )
                        
                        # Aggregate results
                        agg = self.runner.aggregate_kfold_results(fold_results)
                        
                        result = {
                            'group': group_name,
                            'experiment': exp_name,
                            'dataset': dataset,
                            'model': model,
                            'img_size': img_size,
                            'n_folds': len(fold_results),
                        }
                        
                        for metric, (mean, std) in agg.items():
                            result[f'{metric}_mean'] = mean
                            result[f'{metric}_std'] = std
                        
                        results.append(result)
                        
                        print(f"\n{exp_name}:")
                        print(f"  mAP50: {agg['mAP50'][0]:.4f} ± {agg['mAP50'][1]:.4f}")
                        print(f"  mAP50-95: {agg['mAP50-95'][0]:.4f} ± {agg['mAP50-95'][1]:.4f}")
                        
                    except Exception as e:
                        print(f"ERROR in {exp_name}: {e}")
                        results.append({
                            'group': group_name,
                            'experiment': exp_name,
                            'dataset': dataset,
                            'model': model,
                            'error': str(e)
                        })
        
        return results
    
    def _run_cross_dataset_group(self, group_name: str, group: Dict, force: bool) -> List[Dict]:
        """Run cross-dataset experiments."""
        results = []
        
        for train_ds, test_ds in group['pairs']:
            for model in group['models']:
                exp_name = f"{group_name}_{train_ds}_to_{test_ds}_{model}"
                
                try:
                    exp_result = self.runner.run_cross_dataset_eval(
                        train_dataset=train_ds,
                        test_dataset=test_ds,
                        model_name=model,
                        epochs=self.epochs,
                        img_size=group.get('img_size', 1024),
                        seed=self.seed,
                        force=force
                    )
                    
                    result = {
                        'group': group_name,
                        'experiment': exp_name,
                        'train_dataset': train_ds,
                        'test_dataset': test_ds,
                        'model': model,
                        **exp_result.metrics
                    }
                    results.append(result)
                    
                    print(f"\n{exp_name}:")
                    print(f"  Train mAP50: {exp_result.metrics.get('mAP50', 0):.4f}")
                    print(f"  Cross mAP50: {exp_result.metrics.get('cross_mAP50', 0):.4f}")
                    
                except Exception as e:
                    print(f"ERROR in {exp_name}: {e}")
                    results.append({
                        'group': group_name,
                        'experiment': exp_name,
                        'error': str(e)
                    })
        
        return results
    
    def _run_single_experiments(self, group_name: str, group: Dict, force: bool) -> List[Dict]:
        """Run single (non-CV) experiments for testing."""
        results = []
        
        for dataset in group['datasets']:
            for model in group['models']:
                exp_name = f"{group_name}_{dataset}_{model}"
                
                try:
                    model_config = MODEL_CONFIGS.get(model)
                    if model_config is None:
                        print(f"Unknown model: {model}")
                        continue
                    
                    model_config.img_size = group.get('img_size', 640)
                    
                    config = ExperimentConfig(
                        name=exp_name,
                        dataset=dataset,
                        model=model_config,
                        fold=0,
                        epochs=group.get('epochs', 2),
                        patience=5,
                        seed=self.seed,
                        device=self.device
                    )
                    
                    exp_result = self.runner.run_experiment(config, force=force)
                    
                    result = {
                        'group': group_name,
                        'experiment': exp_name,
                        'dataset': dataset,
                        'model': model,
                        'status': 'success',
                        **exp_result.metrics
                    }
                    results.append(result)
                    
                    print(f"\n{exp_name}: SUCCESS")
                    print(f"  Training time: {exp_result.training_time:.1f}s")
                    
                except Exception as e:
                    print(f"ERROR in {exp_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        'group': group_name,
                        'experiment': exp_name,
                        'dataset': dataset,
                        'model': model,
                        'status': 'failed',
                        'error': str(e)
                    })
        
        return results
    
    def run_all(self, force: bool = False, exclude_test: bool = True) -> pd.DataFrame:
        """Run all experiment groups."""
        all_results = []
        
        for group_name in EXPERIMENT_GROUPS:
            if exclude_test and group_name == 'test':
                continue
            
            results = self.run_group(group_name, force=force)
            all_results.extend(results)
        
        df = pd.DataFrame(all_results)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = self.output_dir / f'all_results_{timestamp}.csv'
        df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
        
        return df
    
    def save_results_table(self, results: List[Dict], output_path: Path):
        """Save results as formatted table."""
        df = pd.DataFrame(results)
        
        # Save CSV
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
        # Save LaTeX table if applicable
        if 'mAP50_mean' in df.columns:
            latex_df = df[['dataset', 'model', 'mAP50_mean', 'mAP50_std', 
                          'mAP50-95_mean', 'mAP50-95_std']].copy()
            latex_df['mAP50'] = latex_df.apply(
                lambda r: f"{r['mAP50_mean']:.3f} ± {r['mAP50_std']:.3f}", axis=1
            )
            latex_df['mAP50-95'] = latex_df.apply(
                lambda r: f"{r['mAP50-95_mean']:.3f} ± {r['mAP50-95_std']:.3f}", axis=1
            )
            
            latex_path = output_path.with_suffix('.tex')
            latex_df[['dataset', 'model', 'mAP50', 'mAP50-95']].to_latex(
                latex_path, index=False, escape=False
            )
            print(f"LaTeX table saved to {latex_path}")


def list_experiments():
    """Print available experiments."""
    print("\nAvailable Experiment Groups:")
    print("=" * 60)
    
    for name, group in EXPERIMENT_GROUPS.items():
        print(f"\n{name}:")
        print(f"  Name: {group['name']}")
        print(f"  Description: {group['description']}")
        print(f"  Type: {group['type']}")
        
        if 'datasets' in group:
            print(f"  Datasets: {', '.join(group['datasets'])}")
        if 'models' in group:
            print(f"  Models: {', '.join(group['models'])}")
        if 'pairs' in group:
            pairs_str = ', '.join([f"{a}→{b}" for a, b in group['pairs']])
            print(f"  Pairs: {pairs_str}")
    
    print("\n" + "=" * 60)
    print("\nAvailable Datasets:")
    for name, desc in DATASET_REGISTRY.items():
        print(f"  {name}: {desc}")
    
    print("\nAvailable Models:")
    for name, config in MODEL_CONFIGS.items():
        print(f"  {name}: {config.family.value}")


def main():
    parser = argparse.ArgumentParser(
        description='Run insect detection experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--group', type=str, choices=list(EXPERIMENT_GROUPS.keys()),
                       help='Run specific experiment group')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiment groups')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test to verify setup')
    parser.add_argument('--list', action='store_true',
                       help='List available experiments')
    
    parser.add_argument('--output-dir', type=str, default='runs/experiments',
                       help='Output directory')
    parser.add_argument('--device', type=int, default=0,
                       help='GPU device')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--force', action='store_true',
                       help='Force re-run cached experiments')
    
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    suite = ExperimentSuite(
        output_dir=Path(args.output_dir),
        device=args.device,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed
    )
    
    if args.test:
        print("\nRunning quick test to verify all models...")
        results = suite.run_group('test', force=args.force)
        
        # Check results
        failed = [r for r in results if r.get('status') == 'failed']
        if failed:
            print(f"\n⚠️  {len(failed)} tests FAILED:")
            for f in failed:
                print(f"  - {f['experiment']}: {f.get('error', 'Unknown error')}")
            sys.exit(1)
        else:
            print(f"\n✓ All {len(results)} tests PASSED!")
        return
    
    if args.all:
        df = suite.run_all(force=args.force)
        print("\nFinal Results Summary:")
        print(df.to_string())
        return
    
    if args.group:
        results = suite.run_group(args.group, force=args.force)
        suite.save_results_table(
            results,
            Path(args.output_dir) / f'{args.group}_results'
        )
        return
    
    # Default: show help
    parser.print_help()


if __name__ == '__main__':
    main()
