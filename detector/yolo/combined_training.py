import os
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml
try:
    from tabulate import tabulate
except Exception:
    tabulate = None
import shutil
import argparse


def get_dataset_root(name: str):
    """Return dataset root path for a given dataset key."""
    key = name.lower()
    if key in ('hi_res', 'hires', 'hi-res', 'hiresdataset'):
        return Path('detector/data/hi_res')
    if key in ('literature', 'insectdetectiondataset', 'literature_dataset'):
        return Path('detector/data/InsectDetectionDataset')
    return Path(name)


def filter_labels_keep_class0(label_path: Path, output_path: Path):
    """Filter label file to keep only class 0 annotations."""
    if not label_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch()
        return
    
    filtered_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5 and parts[0] == '0':
                    filtered_lines.append(line)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for line in filtered_lines:
            f.write(line + '\n')


def create_combined_dataset_yaml(output_path: Path, for_training: bool = True):
    """Create YAML for combined dataset (both hi_res and literature class 0).
    
    Args:
        output_path: Path to save the YAML file
        for_training: If True, combine train sets. If False, use for evaluation only.
    """
    hires_root = get_dataset_root('hi_res')
    lit_root = get_dataset_root('literature')
    
    # Create combined dataset structure
    combined_root = output_path.parent / 'combined_dataset'
    combined_images_dir = combined_root / 'images'
    combined_labels_dir = combined_root / 'labels'
    combined_images_dir.mkdir(parents=True, exist_ok=True)
    combined_labels_dir.mkdir(parents=True, exist_ok=True)
    
    train_images = []
    test_hires_images = []
    test_lit_images = []
    
    print(f"Creating combined dataset...")
    
    # Add hi_res images
    if for_training:
        # For training: use train + val sets
        split_files = [hires_root / 'train.txt', hires_root / 'val.txt']
        split_name = "train+val"
    else:
        # For evaluation: use test set
        split_files = [hires_root / 'test.txt']
        split_name = "test"
    
    for split_file in split_files:
        if split_file.exists():
            with open(split_file, 'r') as f:
                for line in f:
                    img_path = Path(line.strip())
                    if img_path.exists():
                        img_name = f"hires_{img_path.name}"
                        
                        # Create symlink
                        target_img = combined_images_dir / img_name
                        if not target_img.exists():
                            try:
                                target_img.symlink_to(img_path.absolute())
                            except:
                                shutil.copy2(img_path, target_img)
                        
                        # Copy label
                        label_path = Path(str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt'))
                        output_label = combined_labels_dir / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
                        if label_path.exists():
                            shutil.copy2(label_path, output_label)
                        else:
                            output_label.touch()
                        
                        if for_training:
                            train_images.append(f"./images/{img_name}")
                        else:
                            test_hires_images.append(f"./images/{img_name}")
    
    print(f"  Added {len(train_images) if for_training else len(test_hires_images)} hi_res {split_name} images")
    
    # Add literature images (class 0 only)
    lit_images_dir = lit_root / 'images'
    all_lit_images = list(lit_images_dir.glob('*.jpg')) + list(lit_images_dir.glob('*.png'))
    
    for img_file in all_lit_images:
        img_name = f"lit_{img_file.name}"
        
        # Create symlink
        target_img = combined_images_dir / img_name
        if not target_img.exists():
            try:
                target_img.symlink_to(img_file.absolute())
            except:
                shutil.copy2(img_file, target_img)
        
        # Filter label to class 0
        label_path = lit_root / 'labels' / img_file.name.replace('.jpg', '.txt').replace('.png', '.txt')
        output_label = combined_labels_dir / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        filter_labels_keep_class0(label_path, output_label)
        
        if for_training:
            train_images.append(f"./images/{img_name}")
        else:
            test_lit_images.append(f"./images/{img_name}")
    
    print(f"  Added {len(all_lit_images)} literature images (class 0 filtered)")
    
    # Create train.txt / test files
    if for_training:
        train_txt = combined_root / 'train.txt'
        with open(train_txt, 'w') as f:
            for img_path in train_images:
                f.write(f"{img_path}\n")
        
        yaml_content = {
            'path': str(combined_root.absolute()),
            'train': 'train.txt',
            'val': 'train.txt',  # Use same for val
            'nc': 1,
            'names': {0: 'ScafoideusTitanus'}
        }
    else:
        # Create separate test files for each dataset
        test_hires_txt = combined_root / 'test_hires.txt'
        test_lit_txt = combined_root / 'test_lit.txt'
        
        with open(test_hires_txt, 'w') as f:
            for img_path in test_hires_images:
                f.write(f"{img_path}\n")
        
        with open(test_lit_txt, 'w') as f:
            for img_path in test_lit_images:
                f.write(f"{img_path}\n")
        
        yaml_content = {
            'path': str(combined_root.absolute()),
            'test_hires': 'test_hires.txt',
            'test_lit': 'test_lit.txt',
            'nc': 1,
            'names': {0: 'ScafoideusTitanus'}
        }
    
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Combined dataset created: {combined_root}")
    return str(output_path.absolute())


def create_augmented_dataset_yaml(output_path: Path):
    """Create YAML for augmented dataset (literature train + ALL hi_res as augmentation)."""
    hires_root = get_dataset_root('hi_res')
    lit_root = get_dataset_root('literature')
    
    # Create augmented dataset structure
    aug_root = output_path.parent / 'augmented_dataset'
    aug_images_dir = aug_root / 'images'
    aug_labels_dir = aug_root / 'labels'
    aug_images_dir.mkdir(parents=True, exist_ok=True)
    aug_labels_dir.mkdir(parents=True, exist_ok=True)
    
    train_images = []
    test_images = []
    
    print(f"Creating augmented dataset (literature + all hi_res)...")
    
    # Add ALL literature images to train (class 0 only)
    lit_images_dir = lit_root / 'images'
    all_lit_images = list(lit_images_dir.glob('*.jpg')) + list(lit_images_dir.glob('*.png'))
    
    for img_file in all_lit_images:
        img_name = f"lit_{img_file.name}"
        
        # Create symlink
        target_img = aug_images_dir / img_name
        if not target_img.exists():
            try:
                target_img.symlink_to(img_file.absolute())
            except:
                shutil.copy2(img_file, target_img)
        
        # Filter label to class 0
        label_path = lit_root / 'labels' / img_file.name.replace('.jpg', '.txt').replace('.png', '.txt')
        output_label = aug_labels_dir / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        filter_labels_keep_class0(label_path, output_label)
        
        train_images.append(f"./images/{img_name}")
    
    print(f"  Added {len(all_lit_images)} literature images (train set)")
    
    # Add ALL hi_res images to train (as augmentation)
    for split_name in ['train', 'val', 'test']:
        split_file = hires_root / f'{split_name}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                for line in f:
                    img_path = Path(line.strip())
                    if img_path.exists():
                        img_name = f"hires_{split_name}_{img_path.name}"
                        
                        # Create symlink
                        target_img = aug_images_dir / img_name
                        if not target_img.exists():
                            try:
                                target_img.symlink_to(img_path.absolute())
                            except:
                                shutil.copy2(img_path, target_img)
                        
                        # Copy label
                        label_path = Path(str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt'))
                        output_label = aug_labels_dir / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
                        if label_path.exists():
                            shutil.copy2(label_path, output_label)
                        else:
                            output_label.touch()
                        
                        train_images.append(f"./images/{img_name}")
    
    print(f"  Added ALL hi_res images as augmentation (total train: {len(train_images)} images)")
    
    # For test: use literature images only (separate copy for clarity)
    test_root = aug_root / 'test_literature'
    test_images_dir = test_root / 'images'
    test_labels_dir = test_root / 'labels'
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file in all_lit_images:
        img_name = img_file.name
        
        # Create symlink
        target_img = test_images_dir / img_name
        if not target_img.exists():
            try:
                target_img.symlink_to(img_file.absolute())
            except:
                shutil.copy2(img_file, target_img)
        
        # Filter label to class 0
        label_path = lit_root / 'labels' / img_file.name.replace('.jpg', '.txt').replace('.png', '.txt')
        output_label = test_labels_dir / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        filter_labels_keep_class0(label_path, output_label)
        
        test_images.append(f"./images/{img_name}")
    
    # Create train.txt and test.txt
    train_txt = aug_root / 'train.txt'
    with open(train_txt, 'w') as f:
        for img_path in train_images:
            f.write(f"{img_path}\n")
    
    test_txt = test_root / 'test.txt'
    with open(test_txt, 'w') as f:
        for img_path in test_images:
            f.write(f"{img_path}\n")
    
    yaml_content = {
        'path': str(aug_root.absolute()),
        'train': 'train.txt',
        'val': 'train.txt',  # Use same for val during training
        'test': str(test_txt.absolute()),
        'nc': 1,
        'names': {0: 'ScafoideusTitanus'}
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Augmented dataset created: {aug_root}")
    return str(output_path.absolute())


def train_model(
    model_name: str,
    data_yaml: str,
    experiment_name: str,
    project: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 1024,
    device: int = 1,
    patience: int = 20,
):
    """Train a YOLO model."""
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}\n")
    
    model = YOLO(f"{model_name}.pt")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=experiment_name,
        project=project,
        device=device,
        patience=patience,
        save=True,
        plots=True,
        verbose=True
    )
    
    return results


def evaluate_model(model_path: str, data_yaml: str, split: str, device: int = 1):
    """Evaluate a trained model on a specific split."""
    print(f"\n{'='*80}")
    print(f"Evaluating {model_path}")
    print(f"Split: {split}")
    print(f"{'='*80}\n")
    
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split=split, device=device)
    
    return {
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map,
        'precision': metrics.box.mp,
        'recall': metrics.box.mr,
    }


def extract_train_metrics(experiment_dir: Path):
    """Extract training metrics from results.csv."""
    results_csv = experiment_dir / 'results.csv'
    if not results_csv.exists():
        return {}
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    return {
        'train_mAP50': df['metrics/mAP50(B)'].max() if 'metrics/mAP50(B)' in df.columns else None,
        'train_mAP50-95': df['metrics/mAP50-95(B)'].max() if 'metrics/mAP50-95(B)' in df.columns else None,
        'epochs_trained': len(df),
    }


def main(
    project: str = 'runs/combined_training',
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 1024,
    device: int = 1,
    patience: int = 20,
    models: list = None,
):
    """Run combined training experiments."""
    
    if models is None:
        models = [
            'yolov5s',
            'yolov5m',
            'yolov8s',
            'yolov8m',
            'yolo11s',
            'yolo11m',
        ]
    
    print(f"\n{'#'*100}")
    print(f"# COMBINED TRAINING EXPERIMENTS")
    print(f"# Models: {models}")
    print(f"# Image Size: {img_size}")
    print(f"# Epochs: {epochs}")
    print(f"# Batch Size: {batch_size}")
    print(f"{'#'*100}\n")
    
    project_path = Path(project)
    project_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # EXPERIMENT 1: Train on both datasets, evaluate on both test sets
    # ========================================================================
    
    print("\n" + "="*100)
    print("EXPERIMENT 1: Train on BOTH datasets → Evaluate on BOTH test sets")
    print("="*100 + "\n")
    
    # Create combined training dataset YAML
    combined_train_yaml = create_combined_dataset_yaml(
        project_path / 'combined_train.yaml',
        for_training=True
    )
    
    # Create combined test dataset YAML (with separate test sets)
    combined_test_yaml = create_combined_dataset_yaml(
        project_path / 'combined_test.yaml',
        for_training=False
    )
    
    exp1_results = []
    
    for model_name in models:
        experiment_name = f"{model_name}_combined"
        experiment_dir = project_path / experiment_name
        model_path = experiment_dir / 'weights' / 'best.pt'
        
        try:
            # Check if already trained
            if model_path.exists():
                print(f"\n{'='*80}")
                print(f"Model already trained: {experiment_name}")
                print(f"Loading existing model...")
                print(f"{'='*80}\n")
            else:
                # Train on combined dataset
                train_model(
                    model_name=model_name,
                    data_yaml=combined_train_yaml,
                    experiment_name=experiment_name,
                    project=project,
                    epochs=epochs,
                    batch_size=batch_size,
                    img_size=img_size,
                    device=device,
                    patience=patience,
                )
            
            # Extract training metrics
            train_metrics = extract_train_metrics(experiment_dir)
            
            # Evaluate on hi_res test set
            eval_hires = evaluate_model(
                model_path=str(model_path),
                data_yaml=combined_test_yaml,
                split='test_hires',
                device=device,
            )
            
            # Evaluate on literature test set
            eval_lit = evaluate_model(
                model_path=str(model_path),
                data_yaml=combined_test_yaml,
                split='test_lit',
                device=device,
            )
            
            result = {
                'model': model_name,
                'train_dataset': 'hi_res+literature',
                'train_mAP50': train_metrics.get('train_mAP50'),
                'train_mAP50-95': train_metrics.get('train_mAP50-95'),
                'hires_test_mAP50': eval_hires['mAP50'],
                'hires_test_mAP50-95': eval_hires['mAP50-95'],
                'hires_test_precision': eval_hires['precision'],
                'hires_test_recall': eval_hires['recall'],
                'lit_test_mAP50': eval_lit['mAP50'],
                'lit_test_mAP50-95': eval_lit['mAP50-95'],
                'lit_test_precision': eval_lit['precision'],
                'lit_test_recall': eval_lit['recall'],
                'epochs': train_metrics.get('epochs_trained'),
            }
            exp1_results.append(result)
            
        except Exception as e:
            print(f"\nERROR with {experiment_name}: {e}\n")
            continue
    
    # Save Experiment 1 results
    if exp1_results:
        exp1_df = pd.DataFrame(exp1_results)
        
        print("\n" + "="*100)
        print("EXPERIMENT 1 RESULTS")
        print("="*100 + "\n")
        
        try:
            if tabulate is not None:
                psql_table = tabulate(exp1_df, headers='keys', tablefmt='psql', showindex=False)
            else:
                psql_table = exp1_df.to_string(index=False)
        except Exception:
            psql_table = exp1_df.to_string(index=False)
        
        print(psql_table)
        
        # Save results
        exp1_csv = project_path / 'experiment1_combined_training.csv'
        exp1_txt = project_path / 'experiment1_combined_training.txt'
        exp1_df.to_csv(exp1_csv, index=False)
        with open(exp1_txt, 'w') as f:
            f.write(psql_table + "\n")
        
        print(f"\nExperiment 1 results saved to: {exp1_csv}")
        print(f"PSQL-styled table saved to: {exp1_txt}")
    
    # ========================================================================
    # EXPERIMENT 2: Train on literature + ALL hi_res, evaluate on literature
    # ========================================================================
    
    print("\n" + "="*100)
    print("EXPERIMENT 2: Train on literature + ALL hi_res → Evaluate on literature test")
    print("="*100 + "\n")
    
    # Create augmented dataset YAML
    augmented_yaml = create_augmented_dataset_yaml(project_path / 'augmented.yaml')
    
    exp2_results = []
    
    for model_name in models:
        experiment_name = f"{model_name}_augmented"
        experiment_dir = project_path / experiment_name
        model_path = experiment_dir / 'weights' / 'best.pt'
        
        try:
            # Check if already trained
            if model_path.exists():
                print(f"\n{'='*80}")
                print(f"Model already trained: {experiment_name}")
                print(f"Loading existing model...")
                print(f"{'='*80}\n")
            else:
                # Train on augmented dataset
                train_model(
                    model_name=model_name,
                    data_yaml=augmented_yaml,
                    experiment_name=experiment_name,
                    project=project,
                    epochs=epochs,
                    batch_size=batch_size,
                    img_size=img_size,
                    device=device,
                    patience=patience,
                )
            
            # Extract training metrics
            train_metrics = extract_train_metrics(experiment_dir)
            
            # Evaluate on literature test set
            eval_metrics = evaluate_model(
                model_path=str(model_path),
                data_yaml=augmented_yaml,
                split='test',
                device=device,
            )
            
            result = {
                'model': model_name,
                'train_dataset': 'literature+all_hires',
                'train_size': 'literature_all + hires_all',
                'train_mAP50': train_metrics.get('train_mAP50'),
                'train_mAP50-95': train_metrics.get('train_mAP50-95'),
                'test_dataset': 'literature',
                'test_mAP50': eval_metrics['mAP50'],
                'test_mAP50-95': eval_metrics['mAP50-95'],
                'test_precision': eval_metrics['precision'],
                'test_recall': eval_metrics['recall'],
                'epochs': train_metrics.get('epochs_trained'),
            }
            exp2_results.append(result)
            
        except Exception as e:
            print(f"\nERROR with {experiment_name}: {e}\n")
            continue
    
    # Save Experiment 2 results
    if exp2_results:
        exp2_df = pd.DataFrame(exp2_results)
        
        print("\n" + "="*100)
        print("EXPERIMENT 2 RESULTS")
        print("="*100 + "\n")
        
        try:
            if tabulate is not None:
                psql_table = tabulate(exp2_df, headers='keys', tablefmt='psql', showindex=False)
            else:
                psql_table = exp2_df.to_string(index=False)
        except Exception:
            psql_table = exp2_df.to_string(index=False)
        
        print(psql_table)
        
        # Save results
        exp2_csv = project_path / 'experiment2_augmented_training.csv'
        exp2_txt = project_path / 'experiment2_augmented_training.txt'
        exp2_df.to_csv(exp2_csv, index=False)
        with open(exp2_txt, 'w') as f:
            f.write(psql_table + "\n")
        
        print(f"\nExperiment 2 results saved to: {exp2_csv}")
        print(f"PSQL-styled table saved to: {exp2_txt}")
    
    # Save combined Excel report
    if exp1_results or exp2_results:
        excel_file = project_path / 'combined_training_results.xlsx'
        with pd.ExcelWriter(excel_file) as writer:
            if exp1_results:
                exp1_df.to_excel(writer, sheet_name='Exp1_Combined', index=False)
            if exp2_results:
                exp2_df.to_excel(writer, sheet_name='Exp2_Augmented', index=False)
        print(f"\nExcel report saved to: {excel_file}")


def _parse_args_and_run():
    parser = argparse.ArgumentParser(description='Combined training experiments for YOLO models')
    parser.add_argument('--project', type=str, default='runs/combined_training', help='Project folder')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--models', type=str, default=None, help='Comma-separated model list')
    
    args = parser.parse_args()
    
    models_list = None
    if args.models:
        models_list = [m.strip() for m in args.models.split(',') if m.strip()]
    
    main(
        project=args.project,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        patience=args.patience,
        models=models_list,
    )


if __name__ == "__main__":
    _parse_args_and_run()
