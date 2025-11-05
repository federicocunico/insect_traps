import pandas as pd
from pathlib import Path
import yaml
import json


def analyze_existing_results(project_dir: str = 'runs/detect_comparison'):
    """Analyze existing training results without retraining."""
    project_path = Path(project_dir)
    
    if not project_path.exists():
        print(f"Project directory {project_dir} does not exist")
        return None
    
    all_results = []
    
    for model_dir in sorted(project_path.iterdir()):
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        results_csv = model_dir / 'results.csv'
        
        if not results_csv.exists():
            print(f"Skipping {model_name} - no results.csv found")
            continue
        
        print(f"Analyzing {model_name}...")
        
        try:
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            
            last_epoch = df.iloc[-1]
            
            metrics = {
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
            
            args_file = model_dir / 'args.yaml'
            if args_file.exists():
                with open(args_file, 'r') as f:
                    args = yaml.safe_load(f)
                    metrics['batch_size'] = args.get('batch', None)
                    metrics['img_size'] = args.get('imgsz', None)
            
            all_results.append(metrics)
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            continue
    
    if len(all_results) == 0:
        print("No valid results found")
        return None
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('best_mAP50-95', ascending=False)
    
    print("\n" + "="*100)
    print("RESULTS SUMMARY - Sorted by Best mAP50-95")
    print("="*100 + "\n")
    print(results_df.to_string(index=False))
    print("\n" + "="*100 + "\n")
    
    output_file = Path(project_dir) / 'model_comparison_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"✓ Results saved to: {output_file}")
    
    try:
        excel_file = Path(project_dir) / 'model_comparison_results.xlsx'
        results_df.to_excel(excel_file, index=False)
        print(f"✓ Results saved to: {excel_file}")
    except Exception as e:
        print(f"Could not save Excel file: {e}")
    
    print("\n" + "="*100)
    print("TOP 3 MODELS BY mAP50-95")
    print("="*100 + "\n")
    top3 = results_df.nlargest(3, 'best_mAP50-95')[['model', 'best_mAP50', 'best_mAP50-95', 'best_precision', 'best_recall']]
    print(top3.to_string(index=False))
    print("\n" + "="*100 + "\n")
    
    results_df['version'] = results_df['model'].str.extract(r'(yolov\d+)', expand=False)
    summary_stats = results_df.groupby('version').agg({
        'best_mAP50-95': ['mean', 'max'],
        'best_mAP50': ['mean', 'max'],
        'best_precision': ['mean', 'max'],
        'best_recall': ['mean', 'max']
    }).round(4)
    
    print("\n" + "="*100)
    print("SUMMARY BY YOLO VERSION")
    print("="*100 + "\n")
    print(summary_stats)
    print("\n" + "="*100 + "\n")
    
    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze YOLO training results")
    parser.add_argument("--project_dir", type=str, default="runs/detect_comparison", help="Path to the project directory")
    args = parser.parse_args()
