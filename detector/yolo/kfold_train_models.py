import os
import json
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
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import argparse
import datetime


def get_dataset_root(name: str):
    """Return dataset root path for a given dataset key."""
    key = name.lower()
    if key in ('hi_res', 'hires', 'hi-res', 'hiresdataset'):
        return Path('detector/data/hi_res')
    if key in ('literature', 'insectdetectiondataset', 'literature_dataset'):
        return Path('detector/data/InsectDetectionDataset')
    # default
    return Path(name)


def get_unique_project_dir(base_project: str):
    """Return a non-existing project directory by appending suffixes if needed."""
    base = Path(base_project)
    if not base.exists():
        return str(base)
    # try numbered suffixes
    for i in range(1, 1000):
        candidate = base.with_name(f"{base.name}_run{i}")
        if not candidate.exists():
            return str(candidate)
    # fallback to timestamp
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return str(base.with_name(f"{base.name}_{ts}"))


def load_dataset_info(data_root: str, filter_class: int = None):
    """Load the image information from dataset.
    
    Works with datasets that have pre-split files (train.txt, val.txt, test.txt)
    or datasets with just images/ and labels/ directories.
    
    Args:
        data_root: Path to dataset root
        filter_class: If provided, count only annotations of this class when determining has_boxes
    """
    data_root_path = Path(data_root)
    train_txt = data_root_path / 'train.txt'
    val_txt = data_root_path / 'val.txt'
    test_txt = data_root_path / 'test.txt'
    
    all_images = []
    
    # Try to load from split files first
    has_split_files = False
    for split_file in [train_txt, val_txt, test_txt]:
        if split_file.exists():
            has_split_files = True
            with open(split_file, 'r') as f:
                for line in f:
                    img_path = line.strip()
                    if img_path:
                        all_images.append(img_path)
    
    # If no split files, scan images directory directly
    if not has_split_files:
        images_dir = data_root_path / 'images'
        if images_dir.exists():
            for img_file in images_dir.glob('*.jpg'):
                # Store relative path from data_root
                all_images.append(f"images/{img_file.name}")
            for img_file in images_dir.glob('*.png'):
                all_images.append(f"images/{img_file.name}")
    
    if not all_images:
        raise ValueError(f"No images found in {data_root}. Check dataset structure.")
    
    image_data = []
    
    for img_path in all_images:
        full_img_path = data_root_path / img_path
        # Handle both absolute and relative paths
        if not full_img_path.exists():
            full_img_path = Path(img_path)
        
        label_path = Path(str(full_img_path).replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt'))
        
        num_boxes = 0
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        # If filter_class is specified, only count that class
                        if filter_class is not None:
                            try:
                                cls = int(line.strip().split()[0])
                                if cls == filter_class:
                                    num_boxes += 1
                            except (ValueError, IndexError):
                                pass
                        else:
                            num_boxes += 1
        
        has_boxes = 1 if num_boxes > 0 else 0
        image_data.append({
            'img_path': img_path,
            'file_name': Path(img_path).name,
            'num_boxes': num_boxes,
            'has_boxes': has_boxes
        })
    
    return pd.DataFrame(image_data)


def create_kfold_splits(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    """Create stratified K-fold splits based on whether images have boxes."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df, df['has_boxes'])):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        folds.append({
            'fold': fold_idx,
            'train': train_df,
            'val': val_df
        })
    
    return folds


def filter_labels_keep_class0(label_path: Path, output_path: Path):
    """Filter label file to keep only class 0 annotations."""
    if not label_path.exists():
        return
    
    filtered_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 5 and parts[0] == '0':  # Keep only class 0
                    filtered_lines.append(line)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for line in filtered_lines:
            f.write(line + '\n')


def create_fold_yaml(
    fold_data: dict,
    base_data_path: str,
    output_yaml_path: str,
    nc: int = 1,
    names: list = None,
    filter_class0_only: bool = False
):
    """Create a YAML file for a specific fold."""
    fold_dir = Path(output_yaml_path).parent
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    train_txt = fold_dir / 'train.txt'
    val_txt = fold_dir / 'val.txt'
    
    base_data_path = Path(base_data_path)
    
    # Create filtered dataset structure if needed
    if filter_class0_only:
        # Create a filtered dataset in the fold directory
        filtered_root = fold_dir / 'filtered_dataset'
        filtered_images_dir = filtered_root / 'images'
        filtered_labels_dir = filtered_root / 'labels'
        filtered_images_dir.mkdir(parents=True, exist_ok=True)
        filtered_labels_dir.mkdir(parents=True, exist_ok=True)
    
    with open(train_txt, 'w') as f:
        for _, row in fold_data['train'].iterrows():
            img_path = base_data_path / row['img_path']
            
            if filter_class0_only:
                # Copy/symlink image and create filtered label
                img_name = Path(img_path).name
                target_img = filtered_images_dir / img_name
                
                # Create symlink to save space
                if not target_img.exists():
                    try:
                        target_img.symlink_to(img_path.absolute())
                    except:
                        # If symlink fails, copy
                        import shutil
                        shutil.copy2(img_path, target_img)
                
                # Filter and write label
                label_path = Path(str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt'))
                output_label = filtered_labels_dir / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
                filter_labels_keep_class0(label_path, output_label)
                
                # Write relative path for YOLO
                f.write(f"./images/{img_name}\n")
            else:
                f.write(f"{img_path.absolute()}\n")
    
    with open(val_txt, 'w') as f:
        for _, row in fold_data['val'].iterrows():
            img_path = base_data_path / row['img_path']
            
            if filter_class0_only:
                img_name = Path(img_path).name
                target_img = filtered_images_dir / img_name
                
                if not target_img.exists():
                    try:
                        target_img.symlink_to(img_path.absolute())
                    except:
                        import shutil
                        shutil.copy2(img_path, target_img)
                
                label_path = Path(str(img_path).replace('/images/', '/labels/').replace('.jpg', '.txt').replace('.png', '.txt'))
                output_label = filtered_labels_dir / img_name.replace('.jpg', '.txt').replace('.png', '.txt')
                filter_labels_keep_class0(label_path, output_label)
                
                f.write(f"./images/{img_name}\n")
            else:
                f.write(f"{img_path.absolute()}\n")
    
    # Set path based on filtering
    if filter_class0_only:
        dataset_path = str(filtered_root.absolute())
        # Copy train.txt and val.txt into filtered_dataset for YOLO to find
        filtered_train_txt = filtered_root / 'train.txt'
        filtered_val_txt = filtered_root / 'val.txt'
        import shutil
        shutil.copy2(train_txt, filtered_train_txt)
        shutil.copy2(val_txt, filtered_val_txt)
        
        yaml_content = {
            'path': dataset_path,
            'train': 'train.txt',  # Relative to path
            'val': 'val.txt',
            'nc': nc,
            'names': names if names else {0: 'ScafoideusTitanus'}
        }
    else:
        dataset_path = str(base_data_path.absolute())
        yaml_content = {
            'path': dataset_path,
            'train': str(train_txt.absolute()),
            'val': str(val_txt.absolute()),
            'nc': nc,
            'names': names if names else {0: 'ScafoideusTitanus'}
        }
    
    with open(output_yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    return output_yaml_path


def train_yolo_model(
    model_name: str,
    data_path: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 1024,
    project: str = 'runs/kfold',
    device: int = 1,
    patience: int = 20,
):
    """Train a single YOLO model and return results."""
    print(f"\n{'='*80}")
    print(f"Training {model_name} with image size {img_size}")
    print(f"{'='*80}\n")
    
    base_model = model_name.split('_fold')[0].split('_img')[0]
    model = YOLO(f"{base_model}.pt")
    
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=model_name,
        project=project,
        device=device,
        patience=patience,
        save=True,
        plots=True,
        verbose=True
    )
    
    return results


def extract_metrics_from_results(model_name: str, project: str = 'runs/kfold'):
    """Extract metrics from trained model results."""
    results_dir = Path(project) / model_name
    
    results_csv = results_dir / 'results.csv'
    if not results_csv.exists():
        print(f"Warning: results.csv not found for {model_name}")
        return None
    
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()
    
    last_epoch = df.iloc[-1]
    
    best_results = {
        'model': model_name,
        'epochs_trained': len(df),
        'final_mAP50': last_epoch.get('metrics/mAP50(B)', None),
        'final_mAP50-95': last_epoch.get('metrics/mAP50-95(B)', None),
        'best_mAP50': df['metrics/mAP50(B)'].max() if 'metrics/mAP50(B)' in df.columns else None,
        'best_mAP50-95': df['metrics/mAP50-95(B)'].max() if 'metrics/mAP50-95(B)' in df.columns else None,
        'final_precision': last_epoch.get('metrics/precision(B)', None),
        'final_recall': last_epoch.get('metrics/recall(B)', None),
        'best_precision': df['metrics/precision(B)'].max() if 'metrics/precision(B)' in df.columns else None,
        'best_recall': df['metrics/recall(B)'].max() if 'metrics/recall(B)' in df.columns else None,
        'final_train_loss': last_epoch.get('train/box_loss', None),
        'final_val_loss': last_epoch.get('val/box_loss', None),
    }
    
    args_file = results_dir / 'args.yaml'
    if args_file.exists():
        with open(args_file, 'r') as f:
            args = yaml.safe_load(f)
            best_results['batch_size'] = args.get('batch', None)
            best_results['img_size'] = args.get('imgsz', None)
    
    return best_results


def validate_model(model_path: str, data_path: str, device: int = 1):
    """Validate a trained model and return metrics."""
    model = YOLO(model_path)
    metrics = model.val(data=data_path, device=device)
    
    return {
        'val_mAP50': metrics.box.map50,
        'val_mAP50-95': metrics.box.map,
        'val_precision': metrics.box.mp,
        'val_recall': metrics.box.mr,
    }


def compute_fold_statistics(fold_results: list, base_model: str):
    """Compute mean and std across all folds for a model."""
    if not fold_results:
        return None
    
    df = pd.DataFrame(fold_results)
    
    metrics_to_aggregate = [
        'best_mAP50', 'best_mAP50-95', 'best_precision', 'best_recall',
        'val_mAP50', 'val_mAP50-95', 'val_precision', 'val_recall'
    ]
    
    stats = {
        'base_model': base_model,
        'n_folds': len(fold_results),
        'img_size': 1024,
    }
    
    for metric in metrics_to_aggregate:
        if metric in df.columns and df[metric].notna().any():
            stats[f'{metric}_mean'] = df[metric].mean()
            stats[f'{metric}_std'] = df[metric].std()
    
    return stats


def main(
    DATA_ROOT: str = "detector/data/hi_res",
    EPOCHS: int = 100,
    BATCH_SIZE: int = 16,
    IMG_SIZE: int = 1024,
    PROJECT: str = None,
    DEVICE: int = 1,
    N_FOLDS: int = 5,
    PATIENCE: int = 20,
    models: list = None,
):
    """Run k-fold experiments.

    DATA_ROOT can be a path or a dataset key (e.g. 'hi_res' or 'literature').
    PROJECT will be auto-generated (and uniquified) if None.
    """
    DATA_ROOT = str(get_dataset_root(DATA_ROOT))
    
    # Detect if this is literature dataset (multi-class) - filter to class 0 only
    is_literature = 'InsectDetectionDataset' in DATA_ROOT or 'literature' in DATA_ROOT.lower()
    filter_class0_only = is_literature
    
    if filter_class0_only:
        print(f"\n{'='*80}")
        print("LITERATURE DATASET DETECTED")
        print("Filtering annotations to class 0 only (keeping ALL images)")
        print("Class 1+ annotations will be removed from labels during training")
        print(f"{'='*80}\n")

    if PROJECT is None:
        dataset_tag = Path(DATA_ROOT).name
        PROJECT = f"runs/kfold_{dataset_tag}"
    # Don't create unique directories - reuse the same one for resume capability
    # PROJECT = get_unique_project_dir(PROJECT)

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
    print(f"# K-FOLD CROSS VALIDATION TRAINING")
    print(f"# Models: {models}")
    print(f"# Image Size: {IMG_SIZE}")
    print(f"# Folds: {N_FOLDS}")
    print(f"# Epochs: {EPOCHS}")
    print(f"# Batch Size: {BATCH_SIZE}")
    print(f"{'#'*100}\n")
    
    print("Loading dataset and creating K-fold splits...")
    # If filtering class 0 only, count only class 0 boxes for stratification
    filter_class = 0 if filter_class0_only else None
    df = load_dataset_info(DATA_ROOT, filter_class=filter_class)
    folds = create_kfold_splits(df, n_splits=N_FOLDS, seed=42)
    
    print(f"\nDataset info:")
    print(f"  Total images: {len(df)}")
    print(f"  Images with boxes: {df['has_boxes'].sum()} ({df['has_boxes'].mean()*100:.2f}%)")
    print(f"  Images without boxes: {(1-df['has_boxes']).sum()} ({(1-df['has_boxes']).mean()*100:.2f}%)")
    
    for fold_idx, fold_data in enumerate(folds):
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(fold_data['train'])} images ({fold_data['train']['has_boxes'].sum()} with boxes)")
        print(f"  Val: {len(fold_data['val'])} images ({fold_data['val']['has_boxes'].sum()} with boxes)")
    
    all_results = []
    model_fold_results = defaultdict(list)
    
    for model_name in models:
        print(f"\n{'#'*100}")
        print(f"# Processing Model: {model_name}")
        print(f"{'#'*100}\n")
        
        for fold_idx, fold_data in enumerate(folds):
            experiment_name = f"{model_name}_img{IMG_SIZE}_fold{fold_idx}"
            experiment_dir = Path(PROJECT) / experiment_name
            model_path = experiment_dir / 'weights' / 'best.pt'
            results_csv = experiment_dir / 'results.csv'
            
            fold_yaml_dir = Path(PROJECT) / 'folds' / f'fold_{fold_idx}'
            fold_yaml_path = fold_yaml_dir / 'fold.yaml'
            
            if results_csv.exists() and model_path.exists():
                print(f"\n{'='*80}")
                print(f"SKIPPING: {experiment_name} - Already trained")
                print(f"Loading existing results...")
                print(f"{'='*80}\n")
                
                metrics = extract_metrics_from_results(experiment_name, project=PROJECT)
                if metrics:
                    metrics['base_model'] = model_name
                    metrics['fold'] = fold_idx
                    metrics['img_size_used'] = IMG_SIZE
                    try:
                        if not fold_yaml_path.exists():
                            create_fold_yaml(
                                fold_data,
                                DATA_ROOT,
                                fold_yaml_path,
                                nc=1,
                                names={0: 'ScafoideusTitanus'},
                                filter_class0_only=filter_class0_only
                            )
                        val_metrics = validate_model(str(model_path), str(fold_yaml_path), device=DEVICE)
                        metrics.update(val_metrics)
                    except Exception as e:
                        print(f"Warning: Could not validate {experiment_name}: {e}")
                    all_results.append(metrics)
                    model_fold_results[model_name].append(metrics)
                continue
            
            try:
                print(f"\n{'='*80}")
                print(f"Training: {model_name} - Fold {fold_idx}/{N_FOLDS-1}")
                print(f"Experiment: {experiment_name}")
                print(f"{'='*80}\n")
                
                fold_yaml = create_fold_yaml(
                    fold_data,
                    DATA_ROOT,
                    fold_yaml_path,
                    nc=1,
                    names={0: 'ScafoideusTitanus'},
                    filter_class0_only=filter_class0_only
                )
                
                print(f"Created fold YAML: {fold_yaml}")
                
                train_yolo_model(
                    model_name=experiment_name,
                    data_path=str(fold_yaml),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    img_size=IMG_SIZE,
                    project=PROJECT,
                    device=DEVICE,
                    patience=PATIENCE,
                )
                
                metrics = extract_metrics_from_results(experiment_name, project=PROJECT)
                if metrics:
                    metrics['base_model'] = model_name
                    metrics['fold'] = fold_idx
                    metrics['img_size_used'] = IMG_SIZE
                    all_results.append(metrics)
                    model_fold_results[model_name].append(metrics)
                
                if model_path.exists():
                    val_metrics = validate_model(str(model_path), str(fold_yaml), device=DEVICE)
                    if metrics:
                        metrics.update(val_metrics)
                
            except Exception as e:
                print(f"\nERROR training {experiment_name}: {e}\n")
                print("Continuing with next fold...\n")
                continue
    
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)

        print("\n" + "="*100)
        print("ALL FOLD RESULTS")
        print("="*100 + "\n")
        # Print as a psql-style table if tabulate is available and save to text file
        project_path = Path(PROJECT)
        project_path.mkdir(parents=True, exist_ok=True)

        try:
            if tabulate is not None:
                psql_all = tabulate(results_df, headers='keys', tablefmt='psql', showindex=False)
            else:
                raise ImportError
        except Exception:
            psql_all = results_df.to_string(index=False)

        print(psql_all)

        # Save CSV and psql text
        output_csv = project_path / 'kfold_all_results.csv'
        results_df.to_csv(output_csv, index=False)
        output_txt = project_path / 'kfold_all_results.txt'
        with open(output_txt, 'w') as f:
            f.write(psql_all + "\n")

        print(f"\nAll results saved to: {output_csv}")
        print(f"PSQL-styled table saved to: {output_txt}")
        
        summary_stats = []
        for model_name in models:
            if model_name in model_fold_results and len(model_fold_results[model_name]) > 0:
                stats = compute_fold_statistics(model_fold_results[model_name], model_name)
                if stats:
                    summary_stats.append(stats)
        
        if summary_stats:
            summary_df = pd.DataFrame(summary_stats)
            summary_df = summary_df.sort_values('best_mAP50-95_mean', ascending=False)
            
            print("\n" + "="*100)
            print("K-FOLD CROSS VALIDATION SUMMARY - Mean ± Std across folds")
            print("="*100 + "\n")
            
            display_cols = ['base_model', 'n_folds']
            for metric in ['best_mAP50-95', 'best_mAP50', 'best_precision', 'best_recall', 
                          'val_mAP50-95', 'val_mAP50', 'val_precision', 'val_recall']:
                mean_col = f'{metric}_mean'
                std_col = f'{metric}_std'
                if mean_col in summary_df.columns:
                    display_cols.extend([mean_col, std_col])
            
            # Print the summary as a psql-style table if possible
            try:
                if tabulate is not None:
                    print(tabulate(summary_df[display_cols], headers=display_cols, tablefmt='psql', showindex=False))
                else:
                    raise ImportError
            except Exception:
                # Create psql-style summary string and save it
                try:
                    if tabulate is not None:
                        psql_summary = tabulate(summary_df[display_cols], headers=display_cols, tablefmt='psql', showindex=False)
                    else:
                        raise ImportError
                except Exception:
                    psql_summary = summary_df[display_cols].to_string(index=False)

                print(psql_summary)

                # Save summary CSV and psql text
                summary_file = Path(PROJECT) / 'kfold_summary.csv'
                summary_df.to_csv(summary_file, index=False)
                summary_txt = Path(PROJECT) / 'kfold_summary.txt'
                with open(summary_txt, 'w') as f:
                    f.write(psql_summary + "\n")
                print(f"\nSummary saved to: {summary_file}")
                print(f"PSQL-styled summary saved to: {summary_txt}")
            
            summary_file = Path(PROJECT) / 'kfold_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            print(f"\nSummary saved to: {summary_file}")
            
            excel_file = Path(PROJECT) / 'kfold_summary.xlsx'
            with pd.ExcelWriter(excel_file) as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                results_df.to_excel(writer, sheet_name='All Folds', index=False)
            print(f"Excel report saved to: {excel_file}")
            
            print("\n" + "="*100)
            print("BEST MODEL (by mAP50-95)")
            print("="*100 + "\n")
            best = summary_df.iloc[0]
            print(f"Model: {best['base_model']}")
            print(f"mAP50-95: {best['best_mAP50-95_mean']:.4f} ± {best['best_mAP50-95_std']:.4f}")
            print(f"mAP50: {best['best_mAP50_mean']:.4f} ± {best['best_mAP50_std']:.4f}")
            print(f"Precision: {best['best_precision_mean']:.4f} ± {best['best_precision_std']:.4f}")
            print(f"Recall: {best['best_recall_mean']:.4f} ± {best['best_recall_std']:.4f}")
            print()
            
            print("\n" + "="*100)
            print("MODEL COMPARISON")
            print("="*100 + "\n")
            # Also print a compact model comparison table and save it
            comp_cols = [c for c in summary_df.columns if any(s in c for s in ['best_mAP50-95_mean','best_mAP50_mean','best_precision_mean','best_recall_mean'])]
            comp_display = ['base_model'] + comp_cols
            try:
                if tabulate is not None:
                    psql_comp = tabulate(summary_df[comp_display], headers=comp_display, tablefmt='psql', showindex=False)
                else:
                    raise ImportError
            except Exception:
                # Fallback to a simple formatted block
                lines = []
                header = ' | '.join(comp_display)
                lines.append(header)
                for _, row in summary_df.iterrows():
                    vals = [str(row.get(c, '')) for c in comp_display]
                    lines.append(' | '.join(vals))
                psql_comp = "\n".join(lines)

            print(psql_comp)

            comp_txt = Path(PROJECT) / 'kfold_model_comparison.txt'
            with open(comp_txt, 'w') as f:
                f.write(psql_comp + "\n")
            print(f"PSQL-styled model comparison saved to: {comp_txt}")

            # Build the final concise psql-style table with mean ± std for requested metrics
            final_rows = []
            for _, row in summary_df.iterrows():
                model = row.get('base_model', '')
                def fmt(metric):
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"
                    mean = row.get(mean_col, None)
                    std = row.get(std_col, None)
                    if pd.isna(mean) or mean is None:
                        return 'N/A'
                    if pd.isna(std) or std is None:
                        return f"{mean:.4f} ± N/A"
                    return f"{mean:.4f} ± {std:.4f}"

                map50 = fmt('best_mAP50')
                map50_95 = fmt('best_mAP50-95')
                mean_prec = fmt('best_precision')
                mean_recall = fmt('best_recall')

                final_rows.append({
                    'model': model,
                    'mAP@0.5': map50,
                    'mAP@0.5-0.95': map50_95,
                    'mean_precision': mean_prec,
                    'mean_recall': mean_recall,
                })

            final_df = pd.DataFrame(final_rows)

            # Print and save final psql table
            try:
                if tabulate is not None:
                    final_psql = tabulate(final_df, headers='keys', tablefmt='psql', showindex=False)
                else:
                    raise ImportError
            except Exception:
                final_psql = final_df.to_string(index=False)

            print("\n" + "="*100)
            print("FINAL SUMMARY TABLE (mean ± std)")
            print("="*100 + "\n")
            print(final_psql)

            final_txt = Path(PROJECT) / 'kfold_final_summary.txt'
            with open(final_txt, 'w') as f:
                f.write(final_psql + "\n")
            print(f"PSQL-styled final summary saved to: {final_txt}")
            
    else:
        print("\nNo results collected\n")


def _parse_args_and_run():
    parser = argparse.ArgumentParser(description='K-Fold training for YOLO models')
    parser.add_argument('--dataset', type=str, default='hi_res', help="Dataset key or path (hi_res or literature)")
    parser.add_argument('--project', type=str, default=None, help='Project base folder (runs/...)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=1024)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--models', type=str, default=None, help='Comma-separated model list (e.g. yolov8n,yolov8s)')

    args = parser.parse_args()

    data_root = args.dataset
    if args.dataset.lower() in ('hi_res', 'hires', 'literature', 'insectdetectiondataset'):
        data_root = get_dataset_root(args.dataset)

    models_list = None
    if args.models:
        models_list = [m.strip() for m in args.models.split(',') if m.strip()]

    main(
        DATA_ROOT=str(data_root),
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch,
        IMG_SIZE=args.imgsz,
        PROJECT=args.project,
        DEVICE=args.device,
        N_FOLDS=args.folds,
        PATIENCE=args.patience,
        models=models_list,
    )


if __name__ == "__main__":
    _parse_args_and_run()
