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
    get_safe_batch_size,
)


# Metric keys written to done.txt for fold experiments, in a stable order.
_FOLD_METRIC_KEYS = ['mAP50', 'mAP50-95', 'mAP75', 'precision', 'recall', 'f1']


def write_done_file(done_file: Path, result, extra_lines: list = None):
    """Write a done.txt marker with the full metric set.

    Records every metric the trainers compute (mAP@50, mAP@50-95, mAP@75,
    precision, recall, F1) so downstream table generation has Precision/Recall/F1
    without needing to retrain. ``extra_lines`` lets cross-dataset runs append
    their target-domain metrics.
    """
    done_file.parent.mkdir(parents=True, exist_ok=True)
    metrics = result.metrics
    with open(done_file, 'w') as f:
        f.write(f"Completed at: {result.timestamp}\n")
        f.write(f"Training time: {result.training_time:.2f}s\n")
        for line in (extra_lines or []):
            f.write(line + "\n")
        for key in _FOLD_METRIC_KEYS:
            if key in metrics:
                f.write(f"{key}: {metrics[key]:.4f}\n")


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
    output_dir: Path,
    batch_size: int = None
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
        # Resolve a memory-safe batch size (explicit override > per-model/res table > config default)
        model_config.batch_size = batch_size if batch_size else get_safe_batch_size(
            model_name, img_size, default=model_config.batch_size
        )
        print(f"  Using batch size: {model_config.batch_size} (model={model_name}, img={img_size})")

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

        write_done_file(done_file, result)

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
    output_dir: Path,
    batch_size: int = None
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
        model_config.batch_size = batch_size if batch_size else get_safe_batch_size(
            model_name, img_size, default=model_config.batch_size
        )
        print(f"  Using batch size: {model_config.batch_size} (model={model_name}, img={img_size})")

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

        m = result.metrics
        # Keep the legacy "Train/Cross mAP50" keys the table parser expects, and add
        # the full target-domain metric set. write_done_file also appends the
        # train-domain mAP/precision/recall/f1 under the standard keys.
        cross_lines = [
            f"Train mAP50: {m.get('mAP50', 0):.4f}",
            f"Cross mAP50: {m.get('cross_mAP50', 0):.4f}",
            f"Cross mAP50-95: {m.get('cross_mAP50-95', 0):.4f}",
            f"Cross precision: {m.get('cross_precision', 0):.4f}",
            f"Cross recall: {m.get('cross_recall', 0):.4f}",
            f"Cross f1: {m.get('cross_f1', 0):.4f}",
        ]
        write_done_file(done_file, result, extra_lines=cross_lines)

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
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default: memory-safe per model/resolution)')
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
            output_dir=output_dir,
            batch_size=args.batch_size
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
            output_dir=output_dir,
            batch_size=args.batch_size
        )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
