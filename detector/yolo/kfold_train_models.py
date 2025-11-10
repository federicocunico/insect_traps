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


def load_dataset_info(data_root: str):
    """Load dataset information from existing YOLO train/val/test splits."""
    train_txt = Path(data_root) / 'train.txt'
    val_txt = Path(data_root) / 'val.txt'
    test_txt = Path(data_root) / 'test.txt'
    
    all_images = []
    
    for split_file in [train_txt, val_txt, test_txt]:
        if split_file.exists():
            with open(split_file, 'r') as f:
                for line in f:
                    img_path = line.strip()
                    if img_path:
                        all_images.append(img_path)
    
    data_root_path = Path(data_root)
    image_data = []
    
    for img_path in all_images:
        full_img_path = data_root_path / img_path
        label_path = Path(str(full_img_path).replace('/images/', '/labels/').replace('.jpg', '.txt'))
        
        num_boxes = 0
        if label_path.exists():
            with open(label_path, 'r') as f:
                num_boxes = len([line for line in f if line.strip()])
        
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


def create_fold_yaml(
    fold_data: dict,
    base_data_path: str,
    output_yaml_path: str,
    nc: int = 1,
    names: list = None
):
    """Create a YAML file for a specific fold."""
    fold_dir = Path(output_yaml_path).parent
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    train_txt = fold_dir / 'train.txt'
    val_txt = fold_dir / 'val.txt'
    
    base_data_path = Path(base_data_path)
    
    with open(train_txt, 'w') as f:
        for _, row in fold_data['train'].iterrows():
            img_path = base_data_path / row['img_path']
            f.write(f"{img_path.absolute()}\n")
    
    with open(val_txt, 'w') as f:
        for _, row in fold_data['val'].iterrows():
            img_path = base_data_path / row['img_path']
            f.write(f"{img_path.absolute()}\n")
    
    yaml_content = {
        'path': str(base_data_path.absolute()),
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


def main():
    DATA_ROOT = "detector/data/hi_res"
    EPOCHS = 100
    BATCH_SIZE = 16
    IMG_SIZE = 1024
    PROJECT = 'runs/kfold'
    DEVICE = 1
    N_FOLDS = 5
    PATIENCE = 20
    
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
    df = load_dataset_info(DATA_ROOT)
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
                                names={0: 'ScafoideusTitanus'}
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
                    names={0: 'ScafoideusTitanus'}
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


if __name__ == "__main__":
    main()
