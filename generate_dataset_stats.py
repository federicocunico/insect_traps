#!/usr/bin/env python
"""
Generate dataset statistics, comparison tables, and visualizations for the research paper.

This script analyzes the datasets used in the study and generates:
- Comparison table with dataset specifications (exportable to PostgreSQL format)
- Statistical summaries for the paper
- Visualization plots for ours_final dataset
- Example images from each dataset
- Distribution comparisons between hi_res and low_res

Usage:
    python generate_dataset_stats.py
    python generate_dataset_stats.py --output-dir stats_output
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from detector.datasets.data_loader import DatasetManager, CVATAnnotationParser

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300


class DatasetAnalyzer:
    """Analyze datasets and generate comprehensive statistics."""
    
    def __init__(self, data_root: Path, output_dir: Path):
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.manager = DatasetManager(self.data_root)
        self.stats = {}
        
        print(f"Dataset root: {self.data_root}")
        print(f"Output directory: {self.output_dir}")
    
    def analyze_all_datasets(self):
        """Analyze all datasets and generate statistics."""
        print("\n" + "="*80)
        print("DATASET ANALYSIS")
        print("="*80)
        
        # Analyze each dataset
        print("\n1. Analyzing literature dataset...")
        self.stats['literature'] = self.analyze_literature_dataset()
        
        print("\n2. Analyzing hi_res dataset...")
        self.stats['hi_res'] = self.analyze_hires_dataset()
        
        print("\n3. Analyzing low_res dataset...")
        self.stats['low_res'] = self.analyze_lowres_dataset()
        
        print("\n4. Creating combined dataset stats...")
        self.stats['hi_res_low_res'] = self.compute_combined_stats()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
    
    def analyze_literature_dataset(self) -> Dict:
        """Analyze the literature dataset (Checola et al., 2024)."""
        lit_path = self.data_root / 'InsectDetectionDataset'
        
        if not lit_path.exists():
            print(f"  Warning: Literature dataset not found at {lit_path}")
            return self._empty_stats()
        
        stats = {
            'name': 'Literature',
            'reference': 'Checola et al., 2024',
            'alias': 'LIT',
            'path': str(lit_path)
        }
        
        # Count images and annotations
        images_dir = lit_path / 'images'
        labels_dir = lit_path / 'labels'
        
        if images_dir.exists():
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            stats['num_images'] = len(image_files)
            
            # Analyze annotations
            annotations = []
            resolutions = []
            
            for img_file in image_files:
                label_file = labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    with open(label_file) as f:
                        lines = f.readlines()
                        annotations.extend(lines)
                
                # Get image resolution
                img = Image.open(img_file)
                resolutions.append(img.size)
            
            stats['num_annotations'] = len(annotations)
            stats['avg_annotations_per_image'] = len(annotations) / len(image_files) if image_files else 0
            stats['resolutions'] = resolutions
            stats['avg_width'] = np.mean([r[0] for r in resolutions]) if resolutions else 0
            stats['avg_height'] = np.mean([r[1] for r in resolutions]) if resolutions else 0
            stats['acquisition_method'] = 'Mixed (field, laboratory, scanned traps, smart-trap)'
            stats['controlled_conditions'] = 'No'
            stats['resolution_quality'] = 'Variable'
        else:
            stats.update(self._empty_stats())
        
        print(f"  Images: {stats.get('num_images', 0)}")
        print(f"  Annotations: {stats.get('num_annotations', 0)}")
        
        return stats
    
    def analyze_hires_dataset(self) -> Dict:
        """Analyze the hi_res dataset."""
        hires_path = self.data_root / 'hi_res'
        
        if not hires_path.exists():
            print(f"  Warning: Hi-res dataset not found at {hires_path}")
            return self._empty_stats()
        
        stats = {
            'name': 'High-Resolution',
            'reference': 'Current work',
            'alias': 'OURS_1',
            'path': str(hires_path)
        }
        
        # Count images and annotations
        images_dir = hires_path / 'images'
        labels_dir = hires_path / 'labels'
        
        image_files = []
        if images_dir.exists():
            # Check for train/val/test split
            if (images_dir / 'train').exists():
                for split in ['train', 'val', 'test']:
                    split_dir = images_dir / split
                    if split_dir.exists():
                        image_files.extend(list(split_dir.glob('*.jpg')) + list(split_dir.glob('*.png')))
            else:
                image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        stats['num_images'] = len(image_files)
        
        # Analyze annotations
        annotations = []
        box_sizes = []
        resolutions = []
        
        for img_file in image_files:
            # Find corresponding label file
            if (images_dir / 'train').exists():
                split_name = img_file.parent.name
                label_file = labels_dir / split_name / (img_file.stem + '.txt')
            else:
                label_file = labels_dir / (img_file.stem + '.txt')
            
            if label_file.exists():
                with open(label_file) as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                    annotations.extend(lines)
                    
                    # Get box sizes (width, height in normalized coordinates)
                    img = Image.open(img_file)
                    img_w, img_h = img.size
                    resolutions.append((img_w, img_h))
                    
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 5:
                            _, x, y, w, h = parts[:5]
                            # Convert normalized to pixel coordinates
                            w_px = float(w) * img_w
                            h_px = float(h) * img_h
                            box_sizes.append((w_px, h_px))
        
        stats['num_annotations'] = len(annotations)
        stats['avg_annotations_per_image'] = len(annotations) / len(image_files) if image_files else 0
        stats['box_sizes'] = box_sizes
        stats['resolutions'] = resolutions
        stats['avg_width'] = np.mean([r[0] for r in resolutions]) if resolutions else 0
        stats['avg_height'] = np.mean([r[1] for r in resolutions]) if resolutions else 0
        stats['avg_box_width'] = np.mean([b[0] for b in box_sizes]) if box_sizes else 0
        stats['avg_box_height'] = np.mean([b[1] for b in box_sizes]) if box_sizes else 0
        stats['acquisition_method'] = 'Controlled imaging box with fixed geometry'
        stats['controlled_conditions'] = 'Yes'
        stats['resolution_quality'] = 'High (~26.6 µm/pixel, 4056×3040)'
        
        print(f"  Images: {stats.get('num_images', 0)}")
        print(f"  Annotations: {stats.get('num_annotations', 0)}")
        
        return stats
    
    def analyze_lowres_dataset(self) -> Dict:
        """Analyze the low_res dataset (field-acquired)."""
        lowres_path = self.data_root / 'low_res'
        
        if not lowres_path.exists():
            print(f"  Warning: Low-res dataset not found at {lowres_path}")
            return self._empty_stats()
        
        stats = {
            'name': 'Field Low-Resolution',
            'reference': 'Current work',
            'alias': 'OURS_2',
            'path': str(lowres_path)
        }
        
        # Check for merged dataset first
        merged_path = lowres_path / 'merged'
        if merged_path.exists() and (merged_path / 'images').exists():
            print("  Using pre-merged low_res dataset...")
            images_dir = merged_path / 'images'
            labels_dir = merged_path / 'labels'
            
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            stats['num_images'] = len(image_files)
            
            annotations = []
            box_sizes = []
            resolutions = []
            
            for img_file in image_files:
                label_file = labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    with open(label_file) as f:
                        lines = [l.strip() for l in f.readlines() if l.strip()]
                        annotations.extend(lines)
                        
                        # Get box sizes
                        img = Image.open(img_file)
                        img_w, img_h = img.size
                        resolutions.append((img_w, img_h))
                        
                        for line in lines:
                            parts = line.split()
                            if len(parts) >= 5:
                                _, x, y, w, h = parts[:5]
                                w_px = float(w) * img_w
                                h_px = float(h) * img_h
                                box_sizes.append((w_px, h_px))
            
            stats['num_annotations'] = len(annotations)
            stats['box_sizes'] = box_sizes
            stats['resolutions'] = resolutions
            
        else:
            # Parse from individual parts
            print("  Merging low_res parts...")
            merge_output, merge_stats = self.manager.merge_low_res_parts()
            
            stats['num_images'] = merge_stats['total_images']
            stats['num_annotations'] = merge_stats['total_annotations']
            stats['parts_info'] = merge_stats['parts']
            
            # Analyze merged dataset
            images_dir = merge_output / 'images'
            labels_dir = merge_output / 'labels'
            
            box_sizes = []
            resolutions = []
            
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            for img_file in image_files[:100]:  # Sample for efficiency
                img = Image.open(img_file)
                resolutions.append(img.size)
                
                label_file = labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    img_w, img_h = img.size
                    with open(label_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                _, x, y, w, h = parts[:5]
                                w_px = float(w) * img_w
                                h_px = float(h) * img_h
                                box_sizes.append((w_px, h_px))
            
            stats['box_sizes'] = box_sizes
            stats['resolutions'] = resolutions
        
        stats['avg_annotations_per_image'] = stats['num_annotations'] / stats['num_images'] if stats['num_images'] > 0 else 0
        stats['avg_width'] = np.mean([r[0] for r in stats['resolutions']]) if stats['resolutions'] else 0
        stats['avg_height'] = np.mean([r[1] for r in stats['resolutions']]) if stats['resolutions'] else 0
        stats['avg_box_width'] = np.mean([b[0] for b in stats['box_sizes']]) if stats['box_sizes'] else 0
        stats['avg_box_height'] = np.mean([b[1] for b in stats['box_sizes']]) if stats['box_sizes'] else 0
        stats['acquisition_method'] = 'Mobile phone (field conditions)'
        stats['controlled_conditions'] = 'No'
        stats['resolution_quality'] = 'Variable (smartphone-dependent)'
        
        print(f"  Images: {stats.get('num_images', 0)}")
        print(f"  Annotations: {stats.get('num_annotations', 0)}")
        
        return stats
    
    def compute_combined_stats(self) -> Dict:
        """Compute statistics for the combined hi_res + low_res dataset."""
        hi_res = self.stats.get('hi_res', {})
        low_res = self.stats.get('low_res', {})
        
        stats = {
            'name': 'Combined (Hi-Res + Low-Res)',
            'reference': 'Current work',
            'alias': 'OURS_FINAL',
            'path': 'Combined dataset',
            'num_images': hi_res.get('num_images', 0) + low_res.get('num_images', 0),
            'num_annotations': hi_res.get('num_annotations', 0) + low_res.get('num_annotations', 0),
            'acquisition_method': 'Combined (controlled + field)',
            'controlled_conditions': 'Mixed',
            'resolution_quality': 'Mixed (high + variable)',
        }
        
        stats['avg_annotations_per_image'] = stats['num_annotations'] / stats['num_images'] if stats['num_images'] > 0 else 0
        
        # Combine box sizes and resolutions if available
        box_sizes = []
        resolutions = []
        
        if 'box_sizes' in hi_res:
            box_sizes.extend(hi_res['box_sizes'])
        if 'box_sizes' in low_res:
            box_sizes.extend(low_res['box_sizes'])
        if 'resolutions' in hi_res:
            resolutions.extend(hi_res['resolutions'])
        if 'resolutions' in low_res:
            resolutions.extend(low_res['resolutions'])
        
        stats['box_sizes'] = box_sizes
        stats['resolutions'] = resolutions
        
        if resolutions:
            stats['avg_width'] = np.mean([r[0] for r in resolutions])
            stats['avg_height'] = np.mean([r[1] for r in resolutions])
        
        if box_sizes:
            stats['avg_box_width'] = np.mean([b[0] for b in box_sizes])
            stats['avg_box_height'] = np.mean([b[1] for b in box_sizes])
        
        print(f"  Images: {stats.get('num_images', 0)}")
        print(f"  Annotations: {stats.get('num_annotations', 0)}")
        
        return stats
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table for all datasets."""
        print("\n" + "="*80)
        print("CREATING COMPARISON TABLE")
        print("="*80)
        
        data = []
        for dataset_key in ['literature', 'hi_res', 'low_res', 'hi_res_low_res']:
            stats = self.stats.get(dataset_key, {})
            
            row = {
                'Dataset': stats.get('alias', dataset_key.upper()),
                'Name': stats.get('name', 'N/A'),
                'Reference': stats.get('reference', 'N/A'),
                'Images': stats.get('num_images', 0),
                'Annotations': stats.get('num_annotations', 0),
                'Avg_Annotations_Per_Image': round(stats.get('avg_annotations_per_image', 0), 2),
                'Avg_Image_Width': int(stats.get('avg_width', 0)),
                'Avg_Image_Height': int(stats.get('avg_height', 0)),
                'Avg_Box_Width_px': int(stats.get('avg_box_width', 0)),
                'Avg_Box_Height_px': int(stats.get('avg_box_height', 0)),
                'Acquisition_Method': stats.get('acquisition_method', 'N/A'),
                'Controlled_Conditions': stats.get('controlled_conditions', 'N/A'),
                'Resolution_Quality': stats.get('resolution_quality', 'N/A'),
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save to various formats
        csv_path = self.output_dir / 'dataset_comparison.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved CSV: {csv_path}")
        
        # Save as LaTeX table
        latex_path = self.output_dir / 'dataset_comparison.tex'
        df.to_latex(latex_path, index=False, escape=False)
        print(f"✓ Saved LaTeX: {latex_path}")
        
        # Save as PostgreSQL INSERT statements
        psql_path = self.output_dir / 'dataset_comparison.sql'
        self._export_to_psql(df, psql_path)
        print(f"✓ Saved PostgreSQL: {psql_path}")
        
        # Save as Markdown
        md_path = self.output_dir / 'dataset_comparison.md'
        df.to_markdown(md_path, index=False)
        print(f"✓ Saved Markdown: {md_path}")
        
        # Print to console
        print("\n" + "="*80)
        print("DATASET COMPARISON TABLE")
        print("="*80)
        print(df.to_string(index=False))
        
        return df
    
    def _export_to_psql(self, df: pd.DataFrame, output_path: Path):
        """Export DataFrame to PostgreSQL INSERT statements."""
        with open(output_path, 'w') as f:
            # Create table
            f.write("-- Dataset Comparison Table\n")
            f.write("DROP TABLE IF EXISTS dataset_comparison;\n\n")
            f.write("CREATE TABLE dataset_comparison (\n")
            f.write("    dataset VARCHAR(50) PRIMARY KEY,\n")
            f.write("    name VARCHAR(100),\n")
            f.write("    reference VARCHAR(100),\n")
            f.write("    images INTEGER,\n")
            f.write("    annotations INTEGER,\n")
            f.write("    avg_annotations_per_image FLOAT,\n")
            f.write("    avg_image_width INTEGER,\n")
            f.write("    avg_image_height INTEGER,\n")
            f.write("    avg_box_width_px INTEGER,\n")
            f.write("    avg_box_height_px INTEGER,\n")
            f.write("    acquisition_method TEXT,\n")
            f.write("    controlled_conditions VARCHAR(10),\n")
            f.write("    resolution_quality TEXT\n")
            f.write(");\n\n")
            
            # Insert data
            for _, row in df.iterrows():
                values = [
                    f"'{row['Dataset']}'",
                    f"'{row['Name']}'",
                    f"'{row['Reference']}'",
                    str(row['Images']),
                    str(row['Annotations']),
                    str(row['Avg_Annotations_Per_Image']),
                    str(row['Avg_Image_Width']),
                    str(row['Avg_Image_Height']),
                    str(row['Avg_Box_Width_px']),
                    str(row['Avg_Box_Height_px']),
                    f"'{row['Acquisition_Method']}'",
                    f"'{row['Controlled_Conditions']}'",
                    f"'{row['Resolution_Quality']}'"
                ]
                f.write(f"INSERT INTO dataset_comparison VALUES ({', '.join(values)});\n")
    
    def generate_visualizations(self):
        """Generate all visualizations for the combined dataset."""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Example images from each dataset
        print("\n1. Generating example images...")
        self._plot_example_images(viz_dir)
        
        # 2. Box size distributions
        print("2. Generating box size distributions...")
        self._plot_box_distributions(viz_dir)
        
        # 3. Resolution distributions
        print("3. Generating resolution distributions...")
        self._plot_resolution_distributions(viz_dir)
        
        # 4. Comparison between hi_res and low_res
        print("4. Generating hi_res vs low_res comparison...")
        self._plot_hires_vs_lowres_comparison(viz_dir)
        
        # 5. Annotation density
        print("5. Generating annotation density plots...")
        self._plot_annotation_density(viz_dir)
        
        # 6. Dataset composition
        print("6. Generating dataset composition plots...")
        self._plot_dataset_composition(viz_dir)
        
        print(f"\n✓ All visualizations saved to: {viz_dir}")
    
    def _plot_example_images(self, output_dir: Path):
        """Plot example images with annotations from each dataset."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Example Images from Each Dataset', fontsize=16, fontweight='bold')
        
        datasets = [
            ('literature', 'Literature (Checola et al.)'),
            ('hi_res', 'Hi-Res (Controlled)'),
            ('low_res', 'Low-Res (Field)'),
        ]
        
        for idx, (dataset_key, title) in enumerate(datasets):
            stats = self.stats.get(dataset_key, {})
            dataset_path = Path(stats.get('path', ''))
            
            if not dataset_path.exists():
                continue
            
            # Find image directory
            if dataset_key == 'literature':
                images_dir = dataset_path / 'images'
                labels_dir = dataset_path / 'labels'
            elif dataset_key == 'hi_res':
                images_dir = dataset_path / 'images'
                labels_dir = dataset_path / 'labels'
                if (images_dir / 'train').exists():
                    images_dir = images_dir / 'train'
                    labels_dir = labels_dir / 'train'
            else:  # low_res
                merged_path = dataset_path / 'merged'
                if merged_path.exists():
                    images_dir = merged_path / 'images'
                    labels_dir = merged_path / 'labels'
                else:
                    continue
            
            if not images_dir.exists():
                continue
            
            # Get sample images
            image_files = list(images_dir.glob('*.jpg'))[:2]
            if not image_files:
                image_files = list(images_dir.glob('*.png'))[:2]
            
            for img_idx, img_file in enumerate(image_files[:2]):
                ax = axes[img_idx, idx]
                
                # Load image
                img = cv2.imread(str(img_file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                
                # Load annotations
                label_file = labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    with open(label_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                _, x, y, bw, bh = map(float, parts[:5])
                                # Convert normalized to pixel coordinates
                                x1 = int((x - bw/2) * w)
                                y1 = int((y - bh/2) * h)
                                x2 = int((x + bw/2) * w)
                                y2 = int((y + bh/2) * h)
                                
                                # Draw bounding box
                                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                ax.imshow(img)
                ax.set_title(f'{title} - Sample {img_idx + 1}')
                ax.axis('off')
        
        # Hide unused subplots
        for i in range(2):
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'example_images.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_box_distributions(self, output_dir: Path):
        """Plot bounding box size distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Bounding Box Size Distributions', fontsize=16, fontweight='bold')
        
        datasets = ['hi_res', 'low_res']
        colors = ['#3498db', '#e74c3c']
        
        # Box width distribution
        ax = axes[0, 0]
        for dataset_key, color in zip(datasets, colors):
            stats = self.stats.get(dataset_key, {})
            if 'box_sizes' in stats and stats['box_sizes']:
                widths = [b[0] for b in stats['box_sizes']]
                ax.hist(widths, bins=50, alpha=0.6, color=color, 
                       label=stats.get('alias', dataset_key))
        ax.set_xlabel('Box Width (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Box Width Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box height distribution
        ax = axes[0, 1]
        for dataset_key, color in zip(datasets, colors):
            stats = self.stats.get(dataset_key, {})
            if 'box_sizes' in stats and stats['box_sizes']:
                heights = [b[1] for b in stats['box_sizes']]
                ax.hist(heights, bins=50, alpha=0.6, color=color,
                       label=stats.get('alias', dataset_key))
        ax.set_xlabel('Box Height (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Box Height Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box area distribution
        ax = axes[1, 0]
        for dataset_key, color in zip(datasets, colors):
            stats = self.stats.get(dataset_key, {})
            if 'box_sizes' in stats and stats['box_sizes']:
                areas = [b[0] * b[1] for b in stats['box_sizes']]
                ax.hist(areas, bins=50, alpha=0.6, color=color,
                       label=stats.get('alias', dataset_key))
        ax.set_xlabel('Box Area (pixels²)')
        ax.set_ylabel('Frequency')
        ax.set_title('Box Area Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box aspect ratio
        ax = axes[1, 1]
        for dataset_key, color in zip(datasets, colors):
            stats = self.stats.get(dataset_key, {})
            if 'box_sizes' in stats and stats['box_sizes']:
                ratios = [b[0] / b[1] if b[1] > 0 else 0 for b in stats['box_sizes']]
                ax.hist(ratios, bins=50, alpha=0.6, color=color,
                       label=stats.get('alias', dataset_key))
        ax.set_xlabel('Aspect Ratio (Width/Height)')
        ax.set_ylabel('Frequency')
        ax.set_title('Box Aspect Ratio Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'box_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_resolution_distributions(self, output_dir: Path):
        """Plot image resolution distributions."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Image Resolution Distributions', fontsize=16, fontweight='bold')
        
        datasets = ['literature', 'hi_res', 'low_res']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        # Width distribution
        ax = axes[0]
        for dataset_key, color in zip(datasets, colors):
            stats = self.stats.get(dataset_key, {})
            if 'resolutions' in stats and stats['resolutions']:
                widths = [r[0] for r in stats['resolutions']]
                ax.hist(widths, bins=30, alpha=0.6, color=color,
                       label=stats.get('alias', dataset_key))
        ax.set_xlabel('Image Width (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Image Width Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Height distribution
        ax = axes[1]
        for dataset_key, color in zip(datasets, colors):
            stats = self.stats.get(dataset_key, {})
            if 'resolutions' in stats and stats['resolutions']:
                heights = [r[1] for r in stats['resolutions']]
                ax.hist(heights, bins=30, alpha=0.6, color=color,
                       label=stats.get('alias', dataset_key))
        ax.set_xlabel('Image Height (pixels)')
        ax.set_ylabel('Frequency')
        ax.set_title('Image Height Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Megapixels
        ax = axes[2]
        for dataset_key, color in zip(datasets, colors):
            stats = self.stats.get(dataset_key, {})
            if 'resolutions' in stats and stats['resolutions']:
                megapixels = [(r[0] * r[1]) / 1e6 for r in stats['resolutions']]
                ax.hist(megapixels, bins=30, alpha=0.6, color=color,
                       label=stats.get('alias', dataset_key))
        ax.set_xlabel('Megapixels')
        ax.set_ylabel('Frequency')
        ax.set_title('Image Size (Megapixels)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'resolution_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_hires_vs_lowres_comparison(self, output_dir: Path):
        """Generate comparison plots between hi_res and low_res datasets."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hi-Res vs Low-Res Comparison', fontsize=16, fontweight='bold')
        
        hi_res = self.stats.get('hi_res', {})
        low_res = self.stats.get('low_res', {})
        
        # 1. Box size comparison
        ax = axes[0, 0]
        data_to_plot = []
        labels = []
        
        if 'box_sizes' in hi_res and hi_res['box_sizes']:
            hi_widths = [b[0] for b in hi_res['box_sizes']]
            data_to_plot.append(hi_widths)
            labels.append('Hi-Res')
        
        if 'box_sizes' in low_res and low_res['box_sizes']:
            low_widths = [b[0] for b in low_res['box_sizes']]
            data_to_plot.append(low_widths)
            labels.append('Low-Res')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            bp['boxes'][0].set_facecolor('#3498db')
            if len(bp['boxes']) > 1:
                bp['boxes'][1].set_facecolor('#e74c3c')
        ax.set_ylabel('Box Width (pixels)')
        ax.set_title('Box Width Comparison')
        ax.grid(True, alpha=0.3)
        
        # 2. Annotations per image
        ax = axes[0, 1]
        categories = []
        values = []
        if hi_res.get('avg_annotations_per_image'):
            categories.append('Hi-Res')
            values.append(hi_res['avg_annotations_per_image'])
        if low_res.get('avg_annotations_per_image'):
            categories.append('Low-Res')
            values.append(low_res['avg_annotations_per_image'])
        
        if values:
            bars = ax.bar(categories, values, color=['#3498db', '#e74c3c'])
            ax.set_ylabel('Avg Annotations per Image')
            ax.set_title('Annotation Density')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom')
        
        # 3. Dataset size comparison
        ax = axes[1, 0]
        categories = ['Images', 'Annotations']
        hi_vals = [hi_res.get('num_images', 0), hi_res.get('num_annotations', 0)]
        low_vals = [low_res.get('num_images', 0), low_res.get('num_annotations', 0)]
        
        x = np.arange(len(categories))
        width = 0.35
        ax.bar(x - width/2, hi_vals, width, label='Hi-Res', color='#3498db')
        ax.bar(x + width/2, low_vals, width, label='Low-Res', color='#e74c3c')
        ax.set_ylabel('Count')
        ax.set_title('Dataset Size Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. Average image resolution
        ax = axes[1, 1]
        if hi_res.get('avg_width') and low_res.get('avg_width'):
            categories = ['Width', 'Height']
            hi_vals = [hi_res.get('avg_width', 0), hi_res.get('avg_height', 0)]
            low_vals = [low_res.get('avg_width', 0), low_res.get('avg_height', 0)]
            
            x = np.arange(len(categories))
            width = 0.35
            ax.bar(x - width/2, hi_vals, width, label='Hi-Res', color='#3498db')
            ax.bar(x + width/2, low_vals, width, label='Low-Res', color='#e74c3c')
            ax.set_ylabel('Pixels')
            ax.set_title('Average Image Resolution')
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'hires_vs_lowres_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_annotation_density(self, output_dir: Path):
        """Plot annotation density statistics."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        datasets = ['literature', 'hi_res', 'low_res', 'hi_res_low_res']
        labels = []
        values = []
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
        
        for dataset_key, color in zip(datasets, colors):
            stats = self.stats.get(dataset_key, {})
            if stats.get('avg_annotations_per_image'):
                labels.append(stats.get('alias', dataset_key))
                values.append(stats['avg_annotations_per_image'])
        
        if values:
            bars = ax.bar(labels, values, color=colors[:len(values)])
            ax.set_ylabel('Average Annotations per Image')
            ax.set_title('Annotation Density Across Datasets', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'annotation_density.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dataset_composition(self, output_dir: Path):
        """Plot dataset composition for the combined dataset."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Combined Dataset (OURS_FINAL) Composition', fontsize=16, fontweight='bold')
        
        hi_res = self.stats.get('hi_res', {})
        low_res = self.stats.get('low_res', {})
        
        # Image composition
        ax = axes[0]
        sizes = [hi_res.get('num_images', 0), low_res.get('num_images', 0)]
        labels = ['Hi-Res', 'Low-Res']
        colors = ['#3498db', '#e74c3c']
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax.set_title('Image Distribution')
        
        # Annotation composition
        ax = axes[1]
        sizes = [hi_res.get('num_annotations', 0), low_res.get('num_annotations', 0)]
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax.set_title('Annotation Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dataset_composition.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_paper_statistics(self):
        """Generate formatted statistics for the paper."""
        print("\n" + "="*80)
        print("GENERATING PAPER STATISTICS")
        print("="*80)
        
        stats_path = self.output_dir / 'paper_statistics.txt'
        
        with open(stats_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATASET STATISTICS FOR RESEARCH PAPER\n")
            f.write("=" * 80 + "\n\n")
            
            for dataset_key in ['literature', 'hi_res', 'low_res', 'hi_res_low_res']:
                stats = self.stats.get(dataset_key, {})
                
                f.write(f"\n{stats.get('name', dataset_key)}:\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Alias: {stats.get('alias', 'N/A')}\n")
                f.write(f"  Reference: {stats.get('reference', 'N/A')}\n")
                f.write(f"  Total images: {stats.get('num_images', 0):,}\n")
                f.write(f"  Total annotations: {stats.get('num_annotations', 0):,}\n")
                f.write(f"  Avg annotations per image: {stats.get('avg_annotations_per_image', 0):.2f}\n")
                
                if stats.get('avg_width'):
                    f.write(f"  Avg image resolution: {int(stats['avg_width'])} × {int(stats['avg_height'])} pixels\n")
                
                if stats.get('avg_box_width'):
                    f.write(f"  Avg box size: {int(stats['avg_box_width'])} × {int(stats['avg_box_height'])} pixels\n")
                
                f.write(f"  Acquisition method: {stats.get('acquisition_method', 'N/A')}\n")
                f.write(f"  Controlled conditions: {stats.get('controlled_conditions', 'N/A')}\n")
                f.write(f"  Resolution quality: {stats.get('resolution_quality', 'N/A')}\n")
                
                # Additional statistics
                if 'box_sizes' in stats and stats['box_sizes']:
                    areas = [b[0] * b[1] for b in stats['box_sizes']]
                    f.write(f"  Box area (mean ± std): {np.mean(areas):.1f} ± {np.std(areas):.1f} px²\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("KEY COMPARISONS\n")
            f.write("=" * 80 + "\n\n")
            
            # Hi-res vs literature
            hi_res = self.stats.get('hi_res', {})
            lit = self.stats.get('literature', {})
            
            if hi_res and lit:
                img_ratio = hi_res.get('num_images', 0) / lit.get('num_images', 1)
                ann_ratio = hi_res.get('num_annotations', 0) / lit.get('num_annotations', 1)
                f.write(f"Hi-Res vs Literature:\n")
                f.write(f"  Image ratio: {img_ratio:.2f}x\n")
                f.write(f"  Annotation ratio: {ann_ratio:.2f}x\n\n")
            
            # Low-res vs literature
            low_res = self.stats.get('low_res', {})
            if low_res and lit:
                img_ratio = low_res.get('num_images', 0) / lit.get('num_images', 1)
                ann_ratio = low_res.get('num_annotations', 0) / lit.get('num_annotations', 1)
                f.write(f"Low-Res vs Literature:\n")
                f.write(f"  Image ratio: {img_ratio:.2f}x\n")
                f.write(f"  Annotation ratio: {ann_ratio:.2f}x\n\n")
            
            # Combined vs literature
            combined = self.stats.get('hi_res_low_res', {})
            if combined and lit:
                img_ratio = combined.get('num_images', 0) / lit.get('num_images', 1)
                ann_ratio = combined.get('num_annotations', 0) / lit.get('num_annotations', 1)
                f.write(f"Combined (OURS_FINAL) vs Literature:\n")
                f.write(f"  Image ratio: {img_ratio:.2f}x\n")
                f.write(f"  Annotation ratio: {ann_ratio:.2f}x\n")
        
        print(f"\n✓ Paper statistics saved to: {stats_path}")
        
        # Also save as JSON for programmatic access
        json_path = self.output_dir / 'dataset_stats.json'
        with open(json_path, 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            stats_copy = {}
            for key, value in self.stats.items():
                stats_copy[key] = {}
                for k, v in value.items():
                    if k in ['box_sizes', 'resolutions']:
                        continue  # Skip large arrays
                    if isinstance(v, (np.integer, np.floating)):
                        stats_copy[key][k] = float(v)
                    else:
                        stats_copy[key][k] = v
            
            json.dump(stats_copy, f, indent=2)
        
        print(f"✓ JSON statistics saved to: {json_path}")
    
    def _empty_stats(self) -> Dict:
        """Return empty statistics dictionary."""
        return {
            'num_images': 0,
            'num_annotations': 0,
            'avg_annotations_per_image': 0,
            'box_sizes': [],
            'resolutions': [],
        }


def main():
    parser = argparse.ArgumentParser(
        description='Generate dataset statistics and comparison tables',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-root', type=str, default='detector/data',
                       help='Root directory containing datasets')
    parser.add_argument('--output-dir', type=str, default='dataset_statistics',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DatasetAnalyzer(
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir)
    )
    
    # Run analysis
    analyzer.analyze_all_datasets()
    
    # Create comparison table
    df = analyzer.create_comparison_table()
    
    # Generate visualizations
    analyzer.generate_visualizations()
    
    # Generate paper statistics
    analyzer.generate_paper_statistics()
    
    print("\n" + "="*80)
    print("ALL ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {analyzer.output_dir}")
    print("\nGenerated files:")
    print("  - dataset_comparison.csv (CSV format)")
    print("  - dataset_comparison.tex (LaTeX table)")
    print("  - dataset_comparison.sql (PostgreSQL)")
    print("  - dataset_comparison.md (Markdown)")
    print("  - dataset_stats.json (JSON format)")
    print("  - paper_statistics.txt (formatted for paper)")
    print("  - visualizations/ (all plots)")


if __name__ == '__main__':
    main()
