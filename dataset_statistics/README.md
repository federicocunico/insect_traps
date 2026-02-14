# Dataset Statistics and Comparison

This directory contains comprehensive statistics and visualizations for the insect detection datasets used in the research paper.

## Generated Files

### Comparison Tables
- **dataset_comparison.csv** - CSV format for spreadsheet analysis
- **dataset_comparison.md** - Markdown table for documentation
- **dataset_comparison.tex** - LaTeX table for paper inclusion
- **dataset_comparison.sql** - PostgreSQL INSERT statements for database import
- **dataset_stats.json** - JSON format for programmatic access

### Statistics
- **paper_statistics.txt** - Formatted statistics ready for paper inclusion
  - Complete dataset specifications
  - Key comparisons between datasets
  - Mean ± std values for box areas

### Visualizations (visualizations/)
- **example_images.png** - Sample annotated images from each dataset
- **box_distributions.png** - Bounding box size distributions (width, height, area, aspect ratio)
- **resolution_distributions.png** - Image resolution distributions across datasets
- **hires_vs_lowres_comparison.png** - Direct comparison between OURS_1 and OURS_2
- **annotation_density.png** - Average annotations per image across all datasets
- **dataset_composition.png** - Pie charts showing OURS_FINAL composition

## Dataset Summary

| Dataset | Alias | Images | Annotations | Avg Ann/Img | Resolution | Acquisition |
|---------|-------|--------|-------------|-------------|------------|-------------|
| Literature | LIT | 615 | 2,835 | 4.61 | 3218×4939 | Mixed sources |
| Hi-Res | OURS_1 | 1,760 | 2,782 | 1.58 | 4056×3040 | Controlled box |
| Low-Res | OURS_2 | 3,800 | 13,142 | 3.46 | 4028×3027 | Mobile phone |
| Combined | OURS_FINAL | 5,560 | 15,924 | 2.86 | 4036×3031 | Mixed |

## Key Findings

### Dataset Size Comparison
- **OURS_FINAL** is **9.04x larger** than Literature in images
- **OURS_FINAL** has **5.62x more** annotations than Literature
- **OURS_2** alone is **6.18x larger** than Literature

### Bounding Box Analysis
- **OURS_1** (Hi-Res) boxes: 184×179 px (33,596 px² mean area)
- **OURS_2** (Low-Res) boxes: 109×106 px (12,200 px² mean area)
- Hi-Res boxes are **2.75x larger** than Low-Res boxes (controlled acquisition benefit)

### Annotation Density
- Literature has highest density: 4.61 annotations/image
- OURS_2 has 3.46 annotations/image (field conditions)
- OURS_1 has 1.58 annotations/image (fewer insects per sticky trap)
- OURS_FINAL balanced at 2.86 annotations/image

## Using the Data

### PostgreSQL Import
```bash
psql -U username -d database -f dataset_comparison.sql
```

### Python Analysis
```python
import pandas as pd
import json

# Load CSV
df = pd.read_csv('dataset_comparison.csv')

# Load JSON
with open('dataset_stats.json') as f:
    stats = json.load(f)
```

### LaTeX Paper Inclusion
```latex
\input{dataset_comparison.tex}
```

## Regenerating Statistics

To regenerate all statistics and visualizations:

```bash
python generate_dataset_stats.py
```

To use a different output directory:

```bash
python generate_dataset_stats.py --output-dir my_stats_output
```

## Citation Format

For the paper, you can cite the datasets as:

- **LIT**: Checola et al., 2024 (615 images, 2,835 annotations)
- **OURS_1**: Current work - High-resolution controlled acquisition (1,760 images, 2,782 annotations)
- **OURS_2**: Current work - Field-acquired smartphone images (3,800 images, 13,142 annotations)
- **OURS_FINAL**: Combined novel dataset (5,560 images, 15,924 annotations)

## Notes

- All images are analyzed at their original resolution
- Box sizes are reported in pixel dimensions
- Annotation counts include only *Scaphoideus titanus* (ST) class
- Literature dataset (LIT) includes mixed acquisition methods from Checola et al., 2024
- OURS_1 uses controlled imaging box with fixed 26.6 µm/pixel resolution
- OURS_2 uses variable smartphone cameras in field conditions
- OURS_FINAL combines both novel datasets for maximum diversity and size
