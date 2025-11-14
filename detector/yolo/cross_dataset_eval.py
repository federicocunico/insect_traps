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
import argparse


def get_dataset_root(name: str):
    """Return dataset root path for a given dataset key."""
    key = name.lower()
    if key in ('hi_res', 'hires', 'hi-res', 'hiresdataset'):
        return Path('detector/data/hi_res')
    if key in ('literature', 'insectdetectiondataset', 'literature_dataset'):
        return Path('detector/data/InsectDetectionDataset')
    return Path(name)


def create_literature_class0_yaml(output_path: Path):
    """Create a YAML for literature dataset with class 0 filtering."""
    from pathlib import Path
    
    lit_root = get_dataset_root('literature')
    
    # Create filtered dataset structure
    filtered_root = output_path.parent / 'filtered_literature'
    filtered_images_dir = filtered_root / 'images'
    filtered_labels_dir = filtered_root / 'labels'
    filtered_images_dir.mkdir(parents=True, exist_ok=True)
    filtered_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all images from literature dataset
    images_dir = lit_root / 'images'
    all_images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    
    print(f"Processing {len(all_images)} images from literature dataset...")
    
    # Create symlinks and filter labels
    train_images = []
    for img_file in all_images:
        img_name = img_file.name
        
        # Create symlink
        target_img = filtered_images_dir / img_name
        if not target_img.exists():
            try:
                target_img.symlink_to(img_file.absolute())
            except:
                import shutil
                shutil.copy2(img_file, target_img)
        
        # Filter label to class 0 only
        label_path = lit_root / 'labels' / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        output_label = filtered_labels_dir / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        
        if label_path.exists():
            filtered_lines = []
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5 and parts[0] == '0':
                            filtered_lines.append(line)
            
            with open(output_label, 'w') as f:
                for line in filtered_lines:
                    f.write(line + '\n')
        else:
            # Create empty label file
            output_label.touch()
        
        train_images.append(f"./images/{img_name}")
    
    # Create train.txt with all images (we'll use full dataset for training)
    train_txt = filtered_root / 'train.txt'
    with open(train_txt, 'w') as f:
        for img_path in train_images:
            f.write(f"{img_path}\n")
    
    print(f"Created filtered literature dataset with {len(train_images)} images (class 0 only)")
    
    yaml_content = {
        'path': str(filtered_root.absolute()),
        'train': 'train.txt',
        'val': 'train.txt',  # Use same for val during training
        'nc': 1,
        'names': {0: 'ScafoideusTitanus'}
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    return str(output_path.absolute())


def create_hires_yaml(output_path: Path):
    """Create a YAML for hi_res dataset."""
    hires_root = get_dataset_root('hi_res')
    
    yaml_content = {
        'path': str(hires_root.absolute()),
        'train': str((hires_root / 'train.txt').absolute()),
        'val': str((hires_root / 'val.txt').absolute()),
        'test': str((hires_root / 'test.txt').absolute()),
        'nc': 1,
        'names': {0: 'ScafoideusTitanus'}
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
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
    print(f"Data: {data_yaml}")
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


def evaluate_model(
    model_path: str,
    data_yaml: str,
    device: int = 1,
):
    """Evaluate a trained model."""
    print(f"\n{'='*80}")
    print(f"Evaluating {model_path}")
    print(f"Data: {data_yaml}")
    print(f"{'='*80}\n")
    
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, device=device, split='test')
    
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
        'train_precision': df['metrics/precision(B)'].max() if 'metrics/precision(B)' in df.columns else None,
        'train_recall': df['metrics/recall(B)'].max() if 'metrics/recall(B)' in df.columns else None,
        'epochs_trained': len(df),
    }


def main(
    project: str = 'runs/cross_dataset',
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 1024,
    device: int = 1,
    patience: int = 20,
    models: list = None,
):
    """Run cross-dataset evaluation experiments."""
    
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
    print(f"# CROSS-DATASET EVALUATION")
    print(f"# Models: {models}")
    print(f"# Image Size: {img_size}")
    print(f"# Epochs: {epochs}")
    print(f"# Batch Size: {batch_size}")
    print(f"{'#'*100}\n")
    
    project_path = Path(project)
    project_path.mkdir(parents=True, exist_ok=True)
    
    # Create dataset YAMLs
    print("Creating dataset configurations...")
    hires_yaml = create_hires_yaml(project_path / 'hires.yaml')
    literature_yaml = create_literature_class0_yaml(project_path / 'literature_class0.yaml')
    
    all_results = []
    
    # Experiment 1: Train on hi_res, evaluate on literature
    print("\n" + "="*100)
    print("EXPERIMENT 1: Train on hi_res → Evaluate on literature (class 0)")
    print("="*100 + "\n")
    
    for model_name in models:
        experiment_name = f"{model_name}_hires_to_lit"
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
                # Train on hi_res
                train_model(
                    model_name=model_name,
                    data_yaml=hires_yaml,
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
            
            # Evaluate on literature (class 0 only)
            eval_metrics = evaluate_model(
                model_path=str(model_path),
                data_yaml=literature_yaml,
                device=device,
            )
            
            result = {
                'model': model_name,
                'train_dataset': 'hi_res',
                'eval_dataset': 'literature',
                'train_mAP50': train_metrics.get('train_mAP50'),
                'train_mAP50-95': train_metrics.get('train_mAP50-95'),
                'eval_mAP50': eval_metrics['mAP50'],
                'eval_mAP50-95': eval_metrics['mAP50-95'],
                'eval_precision': eval_metrics['precision'],
                'eval_recall': eval_metrics['recall'],
                'epochs': train_metrics.get('epochs_trained'),
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"\nERROR with {experiment_name}: {e}\n")
            continue
    
    # Experiment 2: Train on literature, evaluate on hi_res
    print("\n" + "="*100)
    print("EXPERIMENT 2: Train on literature (class 0) → Evaluate on hi_res")
    print("="*100 + "\n")
    
    for model_name in models:
        experiment_name = f"{model_name}_lit_to_hires"
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
                # Train on literature (class 0 only)
                train_model(
                    model_name=model_name,
                    data_yaml=literature_yaml,
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
            eval_metrics = evaluate_model(
                model_path=str(model_path),
                data_yaml=hires_yaml,
                device=device,
            )
            
            result = {
                'model': model_name,
                'train_dataset': 'literature',
                'eval_dataset': 'hi_res',
                'train_mAP50': train_metrics.get('train_mAP50'),
                'train_mAP50-95': train_metrics.get('train_mAP50-95'),
                'eval_mAP50': eval_metrics['mAP50'],
                'eval_mAP50-95': eval_metrics['mAP50-95'],
                'eval_precision': eval_metrics['precision'],
                'eval_recall': eval_metrics['recall'],
                'epochs': train_metrics.get('epochs_trained'),
            }
            all_results.append(result)
            
        except Exception as e:
            print(f"\nERROR with {experiment_name}: {e}\n")
            continue
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        print("\n" + "="*100)
        print("CROSS-DATASET EVALUATION RESULTS")
        print("="*100 + "\n")
        
        # Print and save full results
        try:
            if tabulate is not None:
                psql_table = tabulate(results_df, headers='keys', tablefmt='psql', showindex=False)
            else:
                psql_table = results_df.to_string(index=False)
        except Exception:
            psql_table = results_df.to_string(index=False)
        
        print(psql_table)
        
        # Save CSV and text
        output_csv = project_path / 'cross_dataset_results.csv'
        results_df.to_csv(output_csv, index=False)
        output_txt = project_path / 'cross_dataset_results.txt'
        with open(output_txt, 'w') as f:
            f.write(psql_table + "\n")
        
        print(f"\nResults saved to: {output_csv}")
        print(f"PSQL-styled table saved to: {output_txt}")
        
        # Create summary by direction
        print("\n" + "="*100)
        print("SUMMARY BY TRANSFER DIRECTION")
        print("="*100 + "\n")
        
        summary_rows = []
        for direction in ['hi_res→literature', 'literature→hi_res']:
            if direction == 'hi_res→literature':
                subset = results_df[results_df['train_dataset'] == 'hi_res']
            else:
                subset = results_df[results_df['train_dataset'] == 'literature']
            
            if len(subset) > 0:
                summary_rows.append({
                    'direction': direction,
                    'n_models': len(subset),
                    'mean_eval_mAP50': f"{subset['eval_mAP50'].mean():.4f}",
                    'std_eval_mAP50': f"{subset['eval_mAP50'].std():.4f}",
                    'mean_eval_mAP50-95': f"{subset['eval_mAP50-95'].mean():.4f}",
                    'std_eval_mAP50-95': f"{subset['eval_mAP50-95'].std():.4f}",
                    'best_model': subset.loc[subset['eval_mAP50-95'].idxmax(), 'model'],
                    'best_mAP50-95': f"{subset['eval_mAP50-95'].max():.4f}",
                })
        
        summary_df = pd.DataFrame(summary_rows)
        
        try:
            if tabulate is not None:
                summary_table = tabulate(summary_df, headers='keys', tablefmt='psql', showindex=False)
            else:
                summary_table = summary_df.to_string(index=False)
        except Exception:
            summary_table = summary_df.to_string(index=False)
        
        print(summary_table)
        
        summary_txt = project_path / 'cross_dataset_summary.txt'
        with open(summary_txt, 'w') as f:
            f.write(summary_table + "\n")
        print(f"Summary saved to: {summary_txt}")
        
        # Save Excel
        excel_file = project_path / 'cross_dataset_results.xlsx'
        with pd.ExcelWriter(excel_file) as writer:
            results_df.to_excel(writer, sheet_name='All Results', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        print(f"Excel report saved to: {excel_file}")
        
    else:
        print("\nNo results collected\n")


def _parse_args_and_run():
    parser = argparse.ArgumentParser(description='Cross-dataset evaluation for YOLO models')
    parser.add_argument('--project', type=str, default='runs/cross_dataset', help='Project folder')
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
