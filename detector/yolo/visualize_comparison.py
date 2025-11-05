import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def plot_model_comparison_with_imgsize(results_csv: str, output_dir: str = None):
    """Create comprehensive comparison plots including image size analysis."""
    df = pd.read_csv(results_csv)
    
    if output_dir is None:
        output_dir = Path(results_csv).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract model information
    if 'base_model' not in df.columns:
        df['base_model'] = df['model'].str.extract(r'(yolov\d+[nsmxl])', expand=False)
    
    if 'img_size_used' not in df.columns:
        df['img_size_used'] = df['img_size'].fillna(640)
    
    df['version'] = df['base_model'].str.extract(r'(yolov\d+)', expand=False)
    df['size'] = df['base_model'].str.extract(r'yolov\d+([nsmxl])', expand=False)
    
    sns.set_style("whitegrid")
    
    # Plot 1: Overall comparison - Top models
    fig, ax = plt.subplots(figsize=(16, 10))
    top_models = df.nlargest(15, 'best_mAP50-95')
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_models)))
    
    bars = ax.barh(range(len(top_models)), top_models['best_mAP50-95'], color=colors)
    ax.set_yticks(range(len(top_models)))
    ax.set_yticklabels([f"{row['model']}" for _, row in top_models.iterrows()])
    ax.set_xlabel('mAP50-95', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Model Configurations by mAP50-95', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(top_models.iterrows()):
        value = row['best_mAP50-95']
        if pd.notna(value):
            ax.text(value, i, f' {value:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    output_file = output_dir / 'top_models_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Top models chart saved to: {output_file}")
    plt.close()
    
    # Plot 2: mAP by image size for each model
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for model in df['base_model'].unique():
        model_data = df[df['base_model'] == model].sort_values('img_size_used')
        if len(model_data) > 1:
            ax.plot(model_data['img_size_used'], model_data['best_mAP50-95'], 
                    marker='o', linewidth=2, markersize=8, label=model, alpha=0.7)
    
    ax.set_xlabel('Image Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('mAP50-95', fontsize=12, fontweight='bold')
    ax.set_title('mAP50-95 vs Image Size by Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, ncol=3, loc='best')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'map_vs_imgsize.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ mAP vs image size plot saved to: {output_file}")
    plt.close()
    
    # Plot 3: Heatmap - Model vs Image Size
    if 'img_size_used' in df.columns and 'base_model' in df.columns:
        pivot_map = df.pivot_table(
            values='best_mAP50-95', 
            index='base_model', 
            columns='img_size_used',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(pivot_map, annot=True, fmt='.4f', cmap='YlGnBu', 
                    linewidths=0.5, ax=ax, cbar_kws={'label': 'mAP50-95'})
        ax.set_title('mAP50-95 Heatmap: Model vs Image Size', fontsize=14, fontweight='bold')
        ax.set_xlabel('Image Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        output_file = output_dir / 'model_imgsize_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Model vs Image Size heatmap saved to: {output_file}")
        plt.close()
    
    # Plot 4: Best performance by YOLO version across image sizes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, version in enumerate(sorted(df['version'].unique())):
        ax = axes[idx]
        version_data = df[df['version'] == version]
        
        for size in sorted(version_data['size'].unique()):
            size_data = version_data[version_data['size'] == size].sort_values('img_size_used')
            if len(size_data) > 0:
                ax.plot(size_data['img_size_used'], size_data['best_mAP50-95'], 
                        marker='o', linewidth=2, markersize=8, label=f'{version}{size}')
        
        ax.set_xlabel('Image Size', fontsize=10, fontweight='bold')
        ax.set_ylabel('mAP50-95', fontsize=10, fontweight='bold')
        ax.set_title(f'{version.upper()} Performance', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'version_comparison_by_imgsize.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Version comparison plot saved to: {output_file}")
    plt.close()
    
    # Plot 5: Precision vs Recall colored by image size
    fig, ax = plt.subplots(figsize=(12, 10))
    
    scatter = ax.scatter(df['best_precision'], df['best_recall'], 
                        s=df['best_mAP50-95']*500, 
                        c=df['img_size_used'], 
                        cmap='viridis', 
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Image Size', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Precision', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax.set_title('Precision vs Recall (bubble size = mAP50-95, color = image size)', 
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'precision_recall_scatter.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Precision-Recall scatter plot saved to: {output_file}")
    plt.close()
    
    # Plot 6: Summary statistics by image size
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    img_sizes = sorted(df['img_size_used'].unique())
    metrics = [
        ('best_mAP50-95', 'mAP50-95', axes[0, 0]),
        ('best_mAP50', 'mAP50', axes[0, 1]),
        ('best_precision', 'Precision', axes[1, 0]),
        ('best_recall', 'Recall', axes[1, 1])
    ]
    
    for metric, label, ax in metrics:
        stats_by_size = df.groupby('img_size_used')[metric].agg(['mean', 'max', 'min'])
        
        x_pos = np.arange(len(img_sizes))
        width = 0.25
        
        ax.bar(x_pos - width, stats_by_size['min'], width, label='Min', alpha=0.7)
        ax.bar(x_pos, stats_by_size['mean'], width, label='Mean', alpha=0.7)
        ax.bar(x_pos + width, stats_by_size['max'], width, label='Max', alpha=0.7)
        
        ax.set_xlabel('Image Size', fontsize=12, fontweight='bold')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Statistics by Image Size', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([int(s) for s in img_sizes])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'imgsize_statistics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Image size statistics plot saved to: {output_file}")
    plt.close()
    
    print("\n✓ All visualizations created successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize model comparison results with image size analysis.")
    parser.add_argument("results_csv", type=str, help="Path to the model comparison results CSV file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the output plots")
    args = parser.parse_args()
    results_csv = args.results_csv
    output_dir = args.output_dir

    plot_model_comparison_with_imgsize(results_csv, output_dir)
