import os
import json
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import yaml


def train_yolo_model(
    model_name: str,
    data_path: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    project: str = 'runs/detect',
    device: int = 1,
):
    """Train a single YOLO model and return results."""
    print(f"\n{'='*80}")
    print(f"Training {model_name} with image size {img_size}")
    print(f"{'='*80}\n")
    
    # Extract base model name (e.g., yolov8n from yolov8n_img640)
    base_model = model_name.split('_img')[0] if '_img' in model_name else model_name
    model = YOLO(f"{base_model}.pt")
    
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=model_name,
        project=project,
        device=device,
        patience=20,
        save=True,
        plots=True,
        verbose=True
    )
    
    return results


def extract_metrics_from_results(model_name: str, project: str = 'runs/detect'):
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


def main():
    DATA_PATH = "detector/data/hi_res/hi_res.yaml"
    EPOCHS = 100
    BATCH_SIZE = 16
    PROJECT = 'runs/detect_comparison'
    DEVICE = 1
    
    models = [
        'yolov5n',
        'yolov5s',
        'yolov5m',
        'yolov8n',
        'yolov8s',
        'yolov8m',
        'yolov11n',
        'yolov11s',
        'yolov11m',
    ]
    
    img_sizes = [512, 640, 768, 896, 1024]
    
    all_results = []
    
    for img_size in img_sizes:
        for model_name in models:
            experiment_name = f"{model_name}_img{img_size}"
            
            try:
                print(f"\n{'#'*100}")
                print(f"# Training: {model_name} with image size {img_size}")
                print(f"# Experiment: {experiment_name}")
                print(f"{'#'*100}\n")
                
                train_yolo_model(
                    model_name=experiment_name,
                    data_path=DATA_PATH,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    img_size=img_size,
                    project=PROJECT,
                    device=DEVICE,
                )
                
                metrics = extract_metrics_from_results(experiment_name, project=PROJECT)
                if metrics:
                    metrics['base_model'] = model_name
                    metrics['img_size_used'] = img_size
                    all_results.append(metrics)
                
                model_path = Path(PROJECT) / experiment_name / 'weights' / 'best.pt'
                if model_path.exists():
                    val_metrics = validate_model(str(model_path), DATA_PATH, device=DEVICE)
                    if metrics:
                        metrics.update(val_metrics)
                
            except Exception as e:
                print(f"\nError training {experiment_name}: {e}\n")
                continue
    
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) > 0:
        results_df = results_df.sort_values('best_mAP50-95', ascending=False)
        
        print("\n" + "="*100)
        print("RESULTS SUMMARY - Sorted by Best mAP50-95")
        print("="*100 + "\n")
        print(results_df.to_string(index=False))
        print("\n" + "="*100 + "\n")
        
        output_file = Path(PROJECT) / 'model_comparison_results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
        excel_file = Path(PROJECT) / 'model_comparison_results.xlsx'
        results_df.to_excel(excel_file, index=False)
        print(f"Results saved to: {excel_file}")
        
        print("\n" + "="*100)
        print("TOP 5 MODELS BY mAP50-95")
        print("="*100 + "\n")
        top5 = results_df.nlargest(5, 'best_mAP50-95')[['model', 'base_model', 'img_size_used', 'best_mAP50', 'best_mAP50-95', 'best_precision', 'best_recall']]
        print(top5.to_string(index=False))
        print("\n" + "="*100 + "\n")
        
        print("\n" + "="*100)
        print("SUMMARY BY YOLO VERSION")
        print("="*100 + "\n")
        summary_stats = results_df.groupby('base_model').agg({
            'best_mAP50-95': ['mean', 'max'],
            'best_mAP50': ['mean', 'max'],
            'best_precision': ['mean', 'max'],
            'best_recall': ['mean', 'max']
        }).round(4)
        print(summary_stats)
        print("\n" + "="*100 + "\n")
        
        print("\n" + "="*100)
        print("SUMMARY BY IMAGE SIZE")
        print("="*100 + "\n")
        img_size_stats = results_df.groupby('img_size_used').agg({
            'best_mAP50-95': ['mean', 'max'],
            'best_mAP50': ['mean', 'max'],
            'best_precision': ['mean', 'max'],
            'best_recall': ['mean', 'max']
        }).round(4)
        print(img_size_stats)
        print("\n" + "="*100 + "\n")
        
        for img_size in img_sizes:
            print(f"\n{'='*100}")
            print(f"BEST MODEL FOR IMAGE SIZE {img_size}")
            print(f"{'='*100}\n")
            size_df = results_df[results_df['img_size_used'] == img_size].nlargest(1, 'best_mAP50-95')
            if len(size_df) > 0:
                print(size_df[['model', 'base_model', 'best_mAP50', 'best_mAP50-95', 'best_precision', 'best_recall']].to_string(index=False))
            print()
        
    else:
        print("\nNo results collected\n")


if __name__ == "__main__":
    main()
