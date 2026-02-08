"""
Unified data loader for insect detection experiments.

Handles merging hi_res and low_res (field-lr) datasets, 
converting CVAT XML annotations to YOLO format, and managing dataset splits.
"""

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold


@dataclass
class BoundingBox:
    """Bounding box annotation."""
    label: str
    x_center: float  # normalized 0-1
    y_center: float  # normalized 0-1
    width: float     # normalized 0-1
    height: float    # normalized 0-1
    
    def to_yolo_line(self, class_id: int = 0) -> str:
        return f"{class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


@dataclass
class ImageAnnotation:
    """Annotation for a single image."""
    image_name: str
    image_width: int
    image_height: int
    boxes: List[BoundingBox] = field(default_factory=list)
    source_path: Optional[Path] = None


class CVATAnnotationParser:
    """Parser for CVAT XML annotation format."""
    
    def __init__(self, xml_path: Path, images_dir: Path):
        self.xml_path = xml_path
        self.images_dir = images_dir
        
    def parse(self) -> List[ImageAnnotation]:
        """Parse CVAT XML and return list of image annotations."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        annotations = []
        for image_elem in root.findall('image'):
            img_name = image_elem.get('name')
            img_width = int(image_elem.get('width'))
            img_height = int(image_elem.get('height'))
            
            boxes = []
            for box_elem in image_elem.findall('box'):
                label = box_elem.get('label')
                xtl = float(box_elem.get('xtl'))
                ytl = float(box_elem.get('ytl'))
                xbr = float(box_elem.get('xbr'))
                ybr = float(box_elem.get('ybr'))
                
                # Convert to YOLO format (normalized x_center, y_center, width, height)
                x_center = (xtl + xbr) / 2 / img_width
                y_center = (ytl + ybr) / 2 / img_height
                width = (xbr - xtl) / img_width
                height = (ybr - ytl) / img_height
                
                boxes.append(BoundingBox(
                    label=label,
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height
                ))
            
            # Build source path
            source_path = self.images_dir / img_name
            if not source_path.exists():
                # Try looking without path
                source_path = self.images_dir / Path(img_name).name
                
            annotations.append(ImageAnnotation(
                image_name=img_name,
                image_width=img_width,
                image_height=img_height,
                boxes=boxes,
                source_path=source_path if source_path.exists() else None
            ))
        
        return annotations


@dataclass  
class DatasetSplit:
    """Represents a train/val/test split."""
    train: List[str]
    val: List[str]
    test: List[str]
    

@dataclass
class YOLODatasetConfig:
    """Configuration for a YOLO dataset."""
    name: str
    path: Path
    nc: int = 1
    names: Dict[int, str] = field(default_factory=lambda: {0: 'ScafoideusTitanus'})
    
    def to_yaml(self, output_path: Path, train_file: str = 'train.txt', 
                val_file: str = 'val.txt', test_file: Optional[str] = 'test.txt') -> Path:
        """Write YOLO data.yaml configuration."""
        config = {
            'path': str(self.path.absolute()),
            'train': train_file,
            'val': val_file,
            'nc': self.nc,
            'names': self.names
        }
        if test_file:
            config['test'] = test_file
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return output_path


class DatasetManager:
    """Manages dataset creation, merging, and caching."""
    
    # Class labels
    CLASS_NAMES = {0: 'ScafoideusTitanus'}
    
    def __init__(self, base_dir: Path, cache_dir: Optional[Path] = None):
        """
        Initialize dataset manager.
        
        Args:
            base_dir: Base directory for detector data (e.g., detector/data)
            cache_dir: Directory for caching processed datasets
        """
        self.base_dir = Path(base_dir)
        self.cache_dir = cache_dir or (self.base_dir / '.cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Known dataset paths
        self.hi_res_dir = self.base_dir / 'hi_res'
        self.low_res_dir = self.base_dir / 'low_res'
        self.literature_dir = self.base_dir / 'InsectDetectionDataset'
        
    def _compute_hash(self, *paths: Path) -> str:
        """Compute hash of file contents for cache invalidation."""
        hasher = hashlib.md5()
        for path in sorted(paths):
            if path.is_file():
                hasher.update(path.read_bytes())
            elif path.is_dir():
                for f in sorted(path.rglob('*')):
                    if f.is_file():
                        hasher.update(str(f).encode())
                        hasher.update(str(f.stat().st_mtime).encode())
        return hasher.hexdigest()[:12]
    
    def _is_cached(self, cache_key: str, output_dir: Path) -> bool:
        """Check if a cached version exists and is valid."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return False
        
        with open(cache_file) as f:
            cache_info = json.load(f)
        
        # Check if output directory exists and hash matches
        if output_dir.exists() and cache_info.get('hash') == cache_info.get('expected_hash'):
            return True
        return False
    
    def _save_cache_info(self, cache_key: str, data_hash: str, output_dir: Path, metadata: dict = None):
        """Save cache metadata."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_info = {
            'hash': data_hash,
            'expected_hash': data_hash,
            'output_dir': str(output_dir),
            'metadata': metadata or {}
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_info, f, indent=2)
    
    def merge_low_res_parts(self, force: bool = False) -> Tuple[Path, Dict]:
        """
        Merge all low_res annotation parts into a unified YOLO dataset.
        
        Returns:
            Tuple of (output_directory, statistics)
        """
        output_dir = self.low_res_dir / 'merged'
        cache_key = 'low_res_merged'
        
        # Find all parts
        parts = sorted(self.low_res_dir.glob('part*'))
        if not parts:
            raise ValueError(f"No parts found in {self.low_res_dir}")
        
        # Compute hash
        data_hash = self._compute_hash(*parts)
        
        # Check cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not force and cache_file.exists() and output_dir.exists():
            with open(cache_file) as f:
                cache_info = json.load(f)
            if cache_info.get('hash') == data_hash:
                print(f"[CACHED] Low-res merged dataset: {output_dir}")
                return output_dir, cache_info.get('metadata', {})
        
        print(f"Merging {len(parts)} low_res annotation batches...")
        
        # Create output structure
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        shutil.rmtree(output_dir, ignore_errors=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        all_annotations = []
        stats = {'parts': {}, 'total_images': 0, 'total_boxes': 0}
        
        for part_dir in parts:
            xml_file = part_dir / 'annotations.xml'
            part_images_dir = part_dir / 'images'
            
            if not xml_file.exists():
                print(f"  Warning: No annotations.xml in {part_dir}")
                continue
            
            parser = CVATAnnotationParser(xml_file, part_images_dir)
            annotations = parser.parse()
            
            part_stats = {'images': len(annotations), 'boxes': sum(len(a.boxes) for a in annotations)}
            stats['parts'][part_dir.name] = part_stats
            
            print(f"  {part_dir.name}: {part_stats['images']} images, {part_stats['boxes']} boxes")
            
            for ann in annotations:
                if ann.source_path and ann.source_path.exists():
                    # Copy image
                    dst_img = images_dir / ann.image_name
                    if not dst_img.exists():
                        shutil.copy2(ann.source_path, dst_img)
                    
                    # Write YOLO label
                    label_name = Path(ann.image_name).stem + '.txt'
                    label_path = labels_dir / label_name
                    with open(label_path, 'w') as f:
                        for box in ann.boxes:
                            f.write(box.to_yolo_line(class_id=0) + '\n')
                    
                    all_annotations.append(ann)
        
        stats['total_images'] = len(all_annotations)
        stats['total_boxes'] = sum(len(a.boxes) for a in all_annotations)
        
        print(f"Total: {stats['total_images']} images, {stats['total_boxes']} boxes")
        
        # Save cache info
        self._save_cache_info(cache_key, data_hash, output_dir, stats)
        
        return output_dir, stats
    
    def prepare_dataset(
        self,
        name: str,
        n_folds: int = 5,
        test_ratio: float = 0.15,
        seed: int = 42,
        force: bool = False
    ) -> Tuple[Path, List[DatasetSplit]]:
        """
        Prepare a dataset with stratified k-fold splits.
        
        Args:
            name: Dataset name ('hi_res', 'low_res', 'literature', 'combined')
            n_folds: Number of cross-validation folds
            test_ratio: Ratio of data for holdout test set
            seed: Random seed for reproducibility
            force: Force regeneration even if cached
            
        Returns:
            Tuple of (dataset_path, list_of_splits)
        """
        cache_key = f'{name}_prepared_{n_folds}fold_{seed}'
        
        # Get dataset directory
        if name == 'hi_res':
            dataset_dir = self.hi_res_dir
        elif name == 'low_res':
            dataset_dir, _ = self.merge_low_res_parts(force=force)
        elif name == 'literature':
            dataset_dir = self.literature_dir
        elif name == 'combined':
            return self.prepare_combined_dataset(n_folds=n_folds, test_ratio=test_ratio, 
                                                  seed=seed, force=force)
        elif name == 'hi_res_low_res':
            return self.prepare_combined_dataset(
                datasets=['hi_res', 'low_res'],
                n_folds=n_folds, test_ratio=test_ratio,
                seed=seed, force=force
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")
        
        # Check cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        splits_dir = dataset_dir / 'splits'
        
        if not force and cache_file.exists() and splits_dir.exists():
            with open(cache_file) as f:
                cache_info = json.load(f)
            # Load splits from cache
            splits = self._load_splits(splits_dir, n_folds)
            print(f"[CACHED] Dataset {name} with {n_folds}-fold splits")
            return dataset_dir, splits
        
        # Create splits
        splits = self._create_stratified_splits(dataset_dir, n_folds, test_ratio, seed)
        self._save_splits(dataset_dir / 'splits', splits)
        
        # Save cache
        self._save_cache_info(cache_key, '', dataset_dir, {'n_folds': n_folds})
        
        return dataset_dir, splits
    
    def _get_image_paths(self, dataset_dir: Path) -> List[Path]:
        """Get all image paths from a dataset directory."""
        images_dir = dataset_dir / 'images'
        images = []
        
        if images_dir.exists():
            # Look for images in subdirectories or directly
            for ext in ['*.jpg', '*.png', '*.jpeg']:
                images.extend(images_dir.glob(ext))
                images.extend(images_dir.glob(f'**/{ext}'))
            return list(set(images))
        
        # Try reading from txt files
        for txt_file in ['train.txt', 'val.txt', 'test.txt']:
            txt_path = dataset_dir / txt_file
            if txt_path.exists():
                with open(txt_path) as f:
                    for line in f:
                        img_path = Path(line.strip())
                        if not img_path.is_absolute():
                            img_path = dataset_dir / img_path
                        if img_path.exists():
                            images.append(img_path)
        
        return images
    
    def _get_labels_for_image(self, image_path: Path, dataset_dir: Path) -> int:
        """Get number of boxes for an image (for stratification)."""
        # Try to find label file
        image_path = Path(image_path)
        try:
            rel_path = image_path.relative_to(dataset_dir)
        except ValueError:
            rel_path = Path(image_path.name)
        
        label_path = None
        for labels_base in [dataset_dir / 'labels', dataset_dir]:
            # Try various label path patterns
            candidates = [
                labels_base / rel_path.with_suffix('.txt'),
                labels_base / rel_path.parent / (rel_path.stem + '.txt'),
                labels_base / (rel_path.stem + '.txt'),
            ]
            for cand in candidates:
                if cand.exists():
                    label_path = cand
                    break
            if label_path:
                break
        
        if label_path and label_path.exists():
            with open(label_path) as f:
                return sum(1 for line in f if line.strip())
        return 0
    
    def _create_stratified_splits(
        self, 
        dataset_dir: Path, 
        n_folds: int,
        test_ratio: float,
        seed: int
    ) -> List[DatasetSplit]:
        """Create stratified k-fold splits with holdout test set."""
        images = self._get_image_paths(dataset_dir)
        
        if not images:
            raise ValueError(f"No images found in {dataset_dir}")
        
        # Get stratification labels (has boxes or not)
        strat_labels = [1 if self._get_labels_for_image(img, dataset_dir) > 0 else 0 for img in images]
        
        np.random.seed(seed)
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        
        # Create holdout test set
        n_test = int(len(images) * test_ratio)
        test_indices = indices[:n_test]
        train_val_indices = indices[n_test:]
        
        test_images = [str(images[i]) for i in test_indices]
        
        # Create k-fold splits on remaining data
        train_val_images = [images[i] for i in train_val_indices]
        train_val_labels = [strat_labels[i] for i in train_val_indices]
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_images, train_val_labels)):
            train_imgs = [str(train_val_images[i]) for i in train_idx]
            val_imgs = [str(train_val_images[i]) for i in val_idx]
            splits.append(DatasetSplit(train=train_imgs, val=val_imgs, test=test_images))
        
        return splits
    
    def _save_splits(self, splits_dir: Path, splits: List[DatasetSplit]):
        """Save splits to disk."""
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        for fold_idx, split in enumerate(splits):
            fold_dir = splits_dir / f'fold_{fold_idx}'
            fold_dir.mkdir(exist_ok=True)
            
            for name, paths in [('train', split.train), ('val', split.val), ('test', split.test)]:
                with open(fold_dir / f'{name}.txt', 'w') as f:
                    for p in paths:
                        f.write(f"{p}\n")
    
    def _load_splits(self, splits_dir: Path, n_folds: int) -> List[DatasetSplit]:
        """Load splits from disk."""
        splits = []
        for fold_idx in range(n_folds):
            fold_dir = splits_dir / f'fold_{fold_idx}'
            
            def read_list(filename):
                with open(fold_dir / filename) as f:
                    return [line.strip() for line in f if line.strip()]
            
            splits.append(DatasetSplit(
                train=read_list('train.txt'),
                val=read_list('val.txt'),
                test=read_list('test.txt')
            ))
        return splits
    
    def prepare_combined_dataset(
        self,
        datasets: List[str] = None,
        n_folds: int = 5,
        test_ratio: float = 0.15,
        seed: int = 42,
        force: bool = False
    ) -> Tuple[Path, List[DatasetSplit]]:
        """
        Prepare a combined dataset from multiple sources.
        
        Args:
            datasets: List of dataset names to combine (default: all)
            n_folds: Number of cross-validation folds
            test_ratio: Ratio for holdout test set
            seed: Random seed
            force: Force regeneration
        """
        if datasets is None:
            datasets = ['hi_res', 'low_res', 'literature']
        
        combined_name = '_'.join(sorted(datasets))
        cache_key = f'combined_{combined_name}_{n_folds}fold_{seed}'
        output_dir = self.base_dir / f'combined_{combined_name}'
        
        # Check cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        splits_dir = output_dir / 'splits'
        
        if not force and cache_file.exists() and splits_dir.exists():
            splits = self._load_splits(splits_dir, n_folds)
            print(f"[CACHED] Combined dataset: {combined_name}")
            return output_dir, splits
        
        print(f"Creating combined dataset: {combined_name}")
        
        # Create output structure
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        shutil.rmtree(output_dir, ignore_errors=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        all_images = []
        
        for ds_name in datasets:
            print(f"  Adding {ds_name}...")
            
            if ds_name == 'hi_res':
                ds_dir = self.hi_res_dir
            elif ds_name == 'low_res':
                ds_dir, _ = self.merge_low_res_parts(force=force)
            elif ds_name == 'literature':
                ds_dir = self.literature_dir
            else:
                raise ValueError(f"Unknown dataset: {ds_name}")
            
            # Copy images and labels with prefix
            src_images = self._get_image_paths(ds_dir)
            prefix = ds_name.replace('_', '')
            
            for src_img in src_images:
                dst_name = f"{prefix}_{src_img.name}"
                dst_img = images_dir / dst_name
                
                if not dst_img.exists():
                    # Use symlink for efficiency
                    try:
                        dst_img.symlink_to(src_img.absolute())
                    except:
                        shutil.copy2(src_img, dst_img)
                
                # Find and copy label
                label_name = src_img.stem + '.txt'
                src_label = None
                for candidate in [
                    ds_dir / 'labels' / label_name,
                    ds_dir / 'labels' / src_img.parent.name / label_name,
                ]:
                    if candidate.exists():
                        src_label = candidate
                        break
                
                dst_label = labels_dir / (prefix + '_' + label_name)
                if src_label and src_label.exists():
                    # Filter to class 0 only for literature dataset
                    if ds_name == 'literature':
                        self._filter_label_class0(src_label, dst_label)
                    else:
                        shutil.copy2(src_label, dst_label)
                else:
                    dst_label.touch()
                
                all_images.append(dst_img)
        
        print(f"  Total: {len(all_images)} images")
        
        # Create splits
        strat_labels = [1 if (labels_dir / (img.stem + '.txt')).stat().st_size > 0 else 0 
                       for img in all_images]
        
        np.random.seed(seed)
        indices = np.arange(len(all_images))
        np.random.shuffle(indices)
        
        n_test = int(len(all_images) * test_ratio)
        test_indices = indices[:n_test]
        train_val_indices = indices[n_test:]
        
        test_images = [str(all_images[i]) for i in test_indices]
        train_val_images = [all_images[i] for i in train_val_indices]
        train_val_labels = [strat_labels[i] for i in train_val_indices]
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_images, train_val_labels)):
            train_imgs = [str(train_val_images[i]) for i in train_idx]
            val_imgs = [str(train_val_images[i]) for i in val_idx]
            splits.append(DatasetSplit(train=train_imgs, val=val_imgs, test=test_images))
        
        self._save_splits(splits_dir, splits)
        self._save_cache_info(cache_key, '', output_dir, {'datasets': datasets, 'n_folds': n_folds})
        
        return output_dir, splits
    
    def _filter_label_class0(self, src_label: Path, dst_label: Path):
        """Filter label file to keep only class 0."""
        with open(src_label) as f:
            lines = [l.strip() for l in f if l.strip() and l.strip().split()[0] == '0']
        with open(dst_label, 'w') as f:
            f.write('\n'.join(lines) + '\n' if lines else '')
    
    def create_fold_yaml(
        self,
        dataset_dir: Path,
        split: DatasetSplit,
        output_path: Path,
        relative_paths: bool = True
    ) -> Path:
        """Create a YOLO yaml config for a specific fold."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write split files
        for name, paths in [('train', split.train), ('val', split.val), ('test', split.test)]:
            txt_path = output_path.parent / f'{name}.txt'
            with open(txt_path, 'w') as f:
                for p in paths:
                    if relative_paths:
                        try:
                            p = str(Path(p).relative_to(dataset_dir))
                        except ValueError:
                            pass
                    f.write(f"{p}\n")
        
        config = YOLODatasetConfig(
            name=dataset_dir.name,
            path=dataset_dir,
            nc=1,
            names=self.CLASS_NAMES
        )
        config.to_yaml(output_path)
        return output_path


# Registry of available datasets
DATASET_REGISTRY = {
    'hi_res': 'High-resolution controlled acquisition dataset',
    'low_res': 'Field-acquired low-resolution dataset (FIELD-LR)',
    'literature': 'Literature dataset from Checola et al. (2024)',
    'combined': 'All datasets combined (HI-RES + FIELD-LR + LIT)',
    'hi_res_low_res': 'HI-RES + FIELD-LR combined (novel datasets only)',
}


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare datasets for experiments')
    parser.add_argument('--data-dir', type=str, default='detector/data',
                       help='Base data directory')
    parser.add_argument('--dataset', type=str, choices=list(DATASET_REGISTRY.keys()),
                       default='hi_res', help='Dataset to prepare')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--force', action='store_true', help='Force regeneration')
    
    args = parser.parse_args()
    
    manager = DatasetManager(Path(args.data_dir))
    
    if args.dataset == 'low_res':
        output_dir, stats = manager.merge_low_res_parts(force=args.force)
        print(f"\nLow-res dataset ready: {output_dir}")
        print(f"Statistics: {stats}")
    else:
        dataset_dir, splits = manager.prepare_dataset(
            args.dataset,
            n_folds=args.n_folds,
            seed=args.seed,
            force=args.force
        )
        print(f"\nDataset ready: {dataset_dir}")
        print(f"Number of folds: {len(splits)}")
        for i, split in enumerate(splits):
            print(f"  Fold {i}: train={len(split.train)}, val={len(split.val)}, test={len(split.test)}")
