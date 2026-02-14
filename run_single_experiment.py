#!/usr/bin/env python
"""
Run a single atomic experiment with proper cleanup.
"""
import argparse
import copy
import gc
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detector.experiments.experiment_runner import (
    ExperimentRunner,
    ExperimentConfig,
    MODEL_CONFIGS,
)


def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def run_single_fold(
    group_name: str,
    dataset: str,
    model_name: str,
    fold: int,
    img_size: int,
    epochs: int,
    device: int,
    output_dir: Path
):
    """Run a single fold experiment with cleanup."""
    exp_name = f"{group_name}_{dataset}_{model_name}"
    if img_size != 1024:
        exp_name += f"_img{img_size}"
    exp_name += f"_fold{fold}"
    
    exp_output_dir = output_dir / exp_name
    done_file = exp_output_dir / "done.txt"
    
    if done_file.exists():
        print(f"[SKIP] {exp_name} - already completed (done.txt exists)")
        return True
    
    try:
        print(f"\n{'='*80}")
        print(f"Running: {exp_name}")
        print(f"{'='*80}\n")
        
        model_config = MODEL_CONFIGS.get(model_name)
        if model_config is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = copy.deepcopy(model_config)
        model_config.img_size = img_size
        
        runner = ExperimentRunner(output_dir=output_dir, device=device)
        
        config = ExperimentConfig(
            name=exp_name,
            dataset=dataset,
            model=model_config,
            fold=fold,
            epochs=epochs,
            patience=20,
            seed=42,
            device=device
        )
        
        # Clear any stale cache from a previous failed run
        runner.cache.clear(config)
        
        result = runner.run_experiment(config, force=False)
        
        del runner
        cleanup_memory()
        
        done_file.parent.mkdir(parents=True, exist_ok=True)
        with open(done_file, 'w') as f:
            f.write(f"Completed at: {result.timestamp}\n")
            f.write(f"Training time: {result.training_time:.2f}s\n")
            f.write(f"mAP50: {result.metrics.get('mAP50', 0):.4f}\n")
            f.write(f"mAP50-95: {result.metrics.get('mAP50-95', 0):.4f}\n")
        
        print(f"\n[SUCCESS] {exp_name}")
        print(f"  mAP50: {result.metrics.get('mAP50', 0):.4f}")
        print(f"  mAP50-95: {result.metrics.get('mAP50-95', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] {exp_name}: {e}")
        import traceback
        traceback.print_exc()
        # Remove partial outputs so retry starts clean
        partial_dir = output_dir / exp_name
        if partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)
            print(f"  Cleaned partial output: {partial_dir}")
        cleanup_memory()
        return False


def run_cross_dataset(
    group_name: str,
    train_dataset: str,
    test_dataset: str,
    model_name: str,
    img_size: int,
    epochs: int,
    device: int,
    output_dir: Path
):
    """Run cross-dataset experiment with cleanup."""
    exp_name = f"{group_name}_{train_dataset}_to_{test_dataset}_{model_name}"
    
    exp_output_dir = output_dir / exp_name
    done_file = exp_output_dir / "done.txt"
    
    if done_file.exists():
        print(f"[SKIP] {exp_name} - already completed (done.txt exists)")
        return True
    
    try:
        print(f"\n{'='*80}")
        print(f"Running: {exp_name}")
        print(f"{'='*80}\n")
        
        model_config = MODEL_CONFIGS.get(model_name)
        if model_config is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_config = copy.deepcopy(model_config)
        model_config.img_size = img_size
        
        runner = ExperimentRunner(output_dir=output_dir, device=device)
        
        result = runner.run_cross_dataset_eval(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model_name=model_name,
            epochs=epochs,
            img_size=img_size,
            seed=42,
            force=False
        )
        
        del runner
        cleanup_memory()
        
        done_file.parent.mkdir(parents=True, exist_ok=True)
        with open(done_file, 'w') as f:
            f.write(f"Completed at: {result.timestamp}\n")
            f.write(f"Training time: {result.training_time:.2f}s\n")
            f.write(f"Train mAP50: {result.metrics.get('mAP50', 0):.4f}\n")
            f.write(f"Cross mAP50: {result.metrics.get('cross_mAP50', 0):.4f}\n")
        
        print(f"\n[SUCCESS] {exp_name}")
        print(f"  Train mAP50: {result.metrics.get('mAP50', 0):.4f}")
        print(f"  Cross mAP50: {result.metrics.get('cross_mAP50', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n[FAILED] {exp_name}: {e}")
        import traceback
        traceback.print_exc()
        # Remove partial outputs so retry starts clean
        partial_dir = output_dir / exp_name
        if partial_dir.exists():
            shutil.rmtree(partial_dir, ignore_errors=True)
            print(f"  Cleaned partial output: {partial_dir}")
        cleanup_memory()
        return False


def main():
    parser = argparse.ArgumentParser(description='Run single atomic experiment')
    parser.add_argument('--group', required=True, help='Experiment group (exp1, exp2, etc)')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--train-dataset', help='Train dataset for cross-dataset')
    parser.add_argument('--test-dataset', help='Test dataset for cross-dataset')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--fold', type=int, help='Fold number')
    parser.add_argument('--img-size', type=int, default=1024, help='Image size')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--output-dir', default='runs/experiments', help='Output directory')
    parser.add_argument('--type', choices=['fold', 'cross'], default='fold', help='Experiment type')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if args.type == 'fold':
        if not args.dataset or args.fold is None:
            parser.error("--dataset and --fold required for fold experiments")
        
        success = run_single_fold(
            group_name=args.group,
            dataset=args.dataset,
            model_name=args.model,
            fold=args.fold,
            img_size=args.img_size,
            epochs=args.epochs,
            device=args.device,
            output_dir=output_dir
        )
    elif args.type == 'cross':
        if not args.train_dataset or not args.test_dataset:
            parser.error("--train-dataset and --test-dataset required for cross experiments")
        
        success = run_cross_dataset(
            group_name=args.group,
            train_dataset=args.train_dataset,
            test_dataset=args.test_dataset,
            model_name=args.model,
            img_size=args.img_size,
            epochs=args.epochs,
            device=args.device,
            output_dir=output_dir
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
