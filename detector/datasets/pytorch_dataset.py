"""
PyTorch Dataset for object detection models (Faster R-CNN, DETR, etc.)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class InsectDataset(Dataset):
    """PyTorch Dataset for insect detection in YOLO format."""
    
    def __init__(
        self,
        data_dir: Path,
        split_file: Path,
        img_size: int = 640,
        augment: bool = False,
        class_names: Dict[int, str] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory of dataset
            split_file: Path to txt file listing images
            img_size: Target image size
            augment: Whether to apply augmentation
            class_names: Dictionary mapping class IDs to names
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        self.class_names = class_names or {0: 'ScafoideusTitanus'}
        
        # Load image paths
        self.images = []
        with open(split_file) as f:
            for line in f:
                img_path = line.strip()
                if img_path:
                    full_path = Path(img_path)
                    if not full_path.is_absolute():
                        full_path = self.data_dir / img_path
                    if full_path.exists():
                        self.images.append(full_path)
        
        # Setup transforms
        self.transform = self._get_transforms()
    
    def _get_transforms(self) -> A.Compose:
        """Get albumentations transforms."""
        if self.augment:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
        else:
            return A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='yolo',
                label_fields=['class_labels'],
                min_visibility=0.3
            ))
    
    def _get_label_path(self, img_path: Path) -> Path:
        """Get label path for an image."""
        # Try different label path patterns
        candidates = [
            # images/[split]/img.jpg -> labels/[split]/img.txt (hi_res structure)
            img_path.parent.parent.parent / 'labels' / img_path.parent.name / (img_path.stem + '.txt'),
            img_path.parent.parent / 'labels' / img_path.parent.name / (img_path.stem + '.txt'),
            img_path.parent.parent / 'labels' / (img_path.stem + '.txt'),
            self.data_dir / 'labels' / (img_path.stem + '.txt'),
        ]
        
        for cand in candidates:
            if cand.exists():
                return cand
        
        return candidates[-1]  # Return last candidate even if doesn't exist
    
    def _load_labels(self, label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load YOLO format labels."""
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        # Clip box to image boundary to handle floating-point precision errors
                        x_min = max(0.0, x_center - width / 2)
                        y_min = max(0.0, y_center - height / 2)
                        x_max = min(1.0, x_center + width / 2)
                        y_max = min(1.0, y_center + height / 2)
                        width = x_max - x_min
                        height = y_max - y_min
                        if width <= 0 or height <= 0:
                            continue
                        x_center = x_min + width / 2
                        y_center = y_min + height / 2
                        boxes.append([x_center, y_center, width, height])
                        labels.append(cls)
        
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    
    def _yolo_to_pascal(self, boxes: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
        """Convert YOLO format (x_center, y_center, w, h) to Pascal VOC (x1, y1, x2, y2)."""
        if len(boxes) == 0:
            return boxes
        
        x_center = boxes[:, 0] * img_width
        y_center = boxes[:, 1] * img_height
        w = boxes[:, 2] * img_width
        h = boxes[:, 3] * img_height
        
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        
        return np.stack([x1, y1, x2, y2], axis=1)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path = self.images[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        orig_height, orig_width = image.shape[:2]
        
        # Load labels
        label_path = self._get_label_path(img_path)
        boxes, class_labels = self._load_labels(label_path)
        
        # Apply transforms
        if len(boxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=boxes.tolist(),
                class_labels=class_labels.tolist()
            )
            image = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            class_labels = np.array(transformed['class_labels'], dtype=np.int64)
        else:
            transformed = self.transform(image=image, bboxes=[], class_labels=[])
            image = transformed['image']
            boxes = np.array([], dtype=np.float32).reshape(0, 4)
            class_labels = np.array([], dtype=np.int64)
        
        # Convert YOLO to Pascal VOC format for detection models
        if len(boxes) > 0:
            boxes_pascal = self._yolo_to_pascal(boxes, self.img_size, self.img_size)
        else:
            boxes_pascal = np.array([], dtype=np.float32).reshape(0, 4)
        
        # Create target dict
        target = {
            'boxes': torch.as_tensor(boxes_pascal, dtype=torch.float32),
            'labels': torch.as_tensor(class_labels + 1, dtype=torch.int64),  # +1 for background
            'image_id': torch.tensor([idx]),
            'area': torch.as_tensor(
                (boxes_pascal[:, 2] - boxes_pascal[:, 0]) * (boxes_pascal[:, 3] - boxes_pascal[:, 1])
                if len(boxes_pascal) > 0 else [],
                dtype=torch.float32
            ),
            'iscrowd': torch.zeros(len(boxes_pascal), dtype=torch.int64)
        }
        
        return image, target


def collate_fn(batch):
    """Custom collate function for detection models."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


class YOLOFormatDataset(Dataset):
    """Simple dataset that keeps YOLO format for validation metrics."""
    
    def __init__(
        self,
        data_dir: Path,
        split_file: Path,
        img_size: int = 640
    ):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        
        self.images = []
        with open(split_file) as f:
            for line in f:
                img_path = line.strip()
                if img_path:
                    full_path = Path(img_path)
                    if not full_path.is_absolute():
                        full_path = self.data_dir / img_path
                    if full_path.exists():
                        self.images.append(full_path)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[Path, List[List[float]]]:
        """Return image path and list of [class, x_center, y_center, w, h]."""
        img_path = self.images[idx]
        
        # Find label
        label_path = self._get_label_path(img_path)
        
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([float(x) for x in parts[:5]])
        
        return img_path, labels
    
    def _get_label_path(self, img_path: Path) -> Path:
        candidates = [
            img_path.parent.parent / 'labels' / img_path.parent.name / (img_path.stem + '.txt'),
            img_path.parent.parent / 'labels' / (img_path.stem + '.txt'),
            self.data_dir / 'labels' / (img_path.stem + '.txt'),
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        return candidates[-1]


if __name__ == '__main__':
    # Test dataset
    from torch.utils.data import DataLoader
    
    data_dir = Path('detector/data/hi_res')
    split_file = data_dir / 'train.txt'
    
    if split_file.exists():
        dataset = InsectDataset(data_dir, split_file, img_size=640, augment=True)
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading
        img, target = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Boxes: {target['boxes'].shape}")
        print(f"Labels: {target['labels']}")
        
        # Test dataloader
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"Batch images: {len(batch[0])}")
        print(f"Batch targets: {len(batch[1])}")
