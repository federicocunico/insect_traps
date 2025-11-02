import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class ResizeWithBoxes:
    """Resize image and adjust bounding boxes accordingly."""
    
    def __init__(self, max_size: int = 800):
        """
        Args:
            max_size: Maximum size for the longest edge. Image will be resized
                     proportionally to fit within this size.
        """
        self.max_size = max_size
    
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        """
        Args:
            image: PIL Image
            target: Dictionary with 'boxes' key containing boxes in [x1, y1, x2, y2] format
        
        Returns:
            Resized image and updated target dictionary
        """
        orig_width, orig_height = image.size
        
        # Calculate new size keeping aspect ratio
        if max(orig_width, orig_height) > self.max_size:
            if orig_width > orig_height:
                new_width = self.max_size
                new_height = int(orig_height * (self.max_size / orig_width))
            else:
                new_height = self.max_size
                new_width = int(orig_width * (self.max_size / orig_height))
            
            # Resize image
            image = image.resize((new_width, new_height), Image.BILINEAR)
            
            # Scale bounding boxes
            scale_x = new_width / orig_width
            scale_y = new_height / orig_height
            
            if len(target['boxes']) > 0:
                boxes = target['boxes'].clone()
                boxes[:, [0, 2]] *= scale_x  # Scale x coordinates
                boxes[:, [1, 3]] *= scale_y  # Scale y coordinates
                target['boxes'] = boxes
            
            # Update area
            if len(target['boxes']) > 0:
                target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        return image, target


class InsectDetectionDataset(Dataset):
    """
    PyTorch Dataset for Insect Detection compatible with torchvision detection models.
    
    Loads images and YOLO-format labels from detector/data/InsectDetectionDataset
    and converts them to the format expected by Faster R-CNN and similar models.
    
    Args:
        root_dir: Path to InsectDetectionDataset folder
        split: One of 'train', 'val', or 'test'
        train_ratio: Percentage of data for training (default: 0.7)
        val_ratio: Percentage of data for validation (default: 0.2)
        test_ratio: Percentage of data for testing (default: 0.1)
        seed: Random seed for reproducible splits (default: 42)
        transform: Optional transform to apply to images
        include_background_only: If True, include images with no annotations (default: True)
                                 Setting to False excludes background-only images, which may
                                 help with some training scenarios but could increase false positives
    
    Returns boxes in [x1, y1, x2, y2] format (xyxy) for compatibility with torchvision models.
    """
    
    def __init__(
        self,
        root_dir: str = None,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        seed: int = 42,
        transform: Optional[callable] = None,
        include_background_only: bool = True,
        max_size: int = 800
    ):
        if root_dir is None:
            root_dir = Path(__file__).parent.parent / "data" / "InsectDetectionDataset"
        
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.labels_dir = self.root_dir / "labels"
        self.split = split
        self.transform = transform
        self.include_background_only = include_background_only
        self.resize_transform = ResizeWithBoxes(max_size=max_size) if max_size else None
        
        assert split in ['train', 'val', 'test'], f"Split must be 'train', 'val', or 'test', got '{split}'"
        assert (
            abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        ), "Ratios must sum to 1.0"

        self.image_files = self._load_image_files()
        self.indices = self._split_dataset(train_ratio, val_ratio, test_ratio, seed)

    def _load_image_files(self) -> List[Path]:
        """Load all image files that have corresponding label files."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        
        for img_path in sorted(self.images_dir.iterdir()):
            if img_path.suffix.lower() in image_extensions:
                label_path = self.labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    if self.include_background_only:
                        image_files.append(img_path)
                    else:
                        if self._has_annotations(label_path):
                            image_files.append(img_path)
        
        return image_files
    
    def _has_annotations(self, label_path: Path) -> bool:
        """Check if label file has any valid annotations."""
        if label_path.stat().st_size == 0:
            return False
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and len(line.split()) == 5:
                    return True
        return False
    
    def _split_dataset(
        self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
    ) -> np.ndarray:
        """Split dataset indices based on ratios with fixed random seed."""
        n_samples = len(self.image_files)

        np.random.seed(seed)
        indices = np.random.permutation(n_samples)

        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        if self.split == "train":
            return indices[:train_end]
        elif self.split == "val":
            return indices[train_end:val_end]
        else:  # test
            return indices[val_end:]

    def _load_yolo_annotations(
        self, label_path: Path, img_width: int, img_height: int
    ) -> Dict:
        """
        Load YOLO format annotations and convert to COCO format.

        YOLO format: class_id x_center y_center width height (normalized)
        COCO format: [x_min, y_min, width, height] (absolute pixels)
        """
        boxes = []
        labels = []

        if not label_path.exists():
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            x_center = x_center_norm * img_width
            y_center = y_center_norm * img_height
            width = width_norm * img_width
            height = height_norm * img_height

            x_min = x_center - width / 2
            y_min = y_center - height / 2

            boxes.append([x_min, y_min, width, height])
            labels.append(class_id)

        if len(boxes) == 0:
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get item at index.
        
        Returns:
            image: PIL Image or transformed tensor
            target: Dictionary with:
                - boxes: FloatTensor[N, 4] in format [x1, y1, x2, y2] (xyxy format for Faster R-CNN)
                - labels: Int64Tensor[N] with class labels
                - image_id: int
                - area: FloatTensor[N]
                - iscrowd: UInt8Tensor[N] (all zeros)
        """
        actual_idx = self.indices[idx]
        img_path = self.image_files[actual_idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        target = self._load_yolo_annotations(label_path, img_width, img_height)

        # Convert COCO format [x_min, y_min, width, height] to [x1, y1, x2, y2]
        if len(target['boxes']) > 0:
            boxes_coco = target['boxes']
            x1 = boxes_coco[:, 0]
            y1 = boxes_coco[:, 1]
            x2 = boxes_coco[:, 0] + boxes_coco[:, 2]
            y2 = boxes_coco[:, 1] + boxes_coco[:, 3]
            target['boxes'] = torch.stack([x1, y1, x2, y2], dim=1)
        
        target["image_id"] = actual_idx
        target["area"] = (target['boxes'][:, 2] - target['boxes'][:, 0]) * (target['boxes'][:, 3] - target['boxes'][:, 1])
        target["iscrowd"] = torch.zeros((len(target["boxes"]),), dtype=torch.int64)

        # Apply resize transform (resizes both image and boxes)
        if self.resize_transform is not None:
            image, target = self.resize_transform(image, target)

        # Apply regular transforms (only to image)
        if self.transform is not None:
            image = self.transform(image)

        return image, target
    
    def get_image_path(self, idx: int) -> Path:
        """Get the file path for image at index."""
        actual_idx = self.indices[idx]
        return self.image_files[actual_idx]

    def get_split_info(self) -> Dict[str, int]:
        """Get information about the dataset split."""
        return {
            "split": self.split,
            "n_samples": len(self.indices),
            "total_images": len(self.image_files),
        }


def create_dataloaders(
    root_dir: str = None,
    batch_size: int = 8,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
    train_transform: Optional[callable] = None,
    val_transform: Optional[callable] = None,
    test_transform: Optional[callable] = None,
    include_background_only: bool = True
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train, val, and test dataloaders.

    Args:
        root_dir: Path to InsectDetectionDataset folder
        batch_size: Batch size for dataloaders
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for splits
        num_workers: Number of worker processes
        train_transform: Transform for training set
        val_transform: Transform for validation set
        test_transform: Transform for test set
        include_background_only: If True, include images with no annotations (background only)

    Returns:
        Dictionary with 'train', 'val', and 'test' dataloaders
    """
    datasets = {
        "train": InsectDetectionDataset(
            root_dir=root_dir,
            split="train",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            transform=train_transform,
            include_background_only=include_background_only,
        ),
        "val": InsectDetectionDataset(
            root_dir=root_dir,
            split="val",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            transform=val_transform,
            include_background_only=include_background_only,
        ),
        "test": InsectDetectionDataset(
            root_dir=root_dir,
            split="test",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            transform=test_transform,
            include_background_only=include_background_only,
        ),
    }

    def collate_fn(batch):
        """Custom collate function for variable-sized annotations."""
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

    dataloaders = {
        split: torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        for split, dataset in datasets.items()
    }

    return dataloaders


if __name__ == "__main__":

    from torch.utils.data import DataLoader

    train_dataset = InsectDetectionDataset(
        split="train", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42
    )
    val_dataset = InsectDetectionDataset(
        split="val", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42
    )
    test_dataset = InsectDetectionDataset(
        split="test", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42
    )
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Iterate over batches
    for images, targets in train_loader:
        # images: list of PIL Images or tensors (if transform applied)
        # targets: list of dicts with 'boxes', 'labels', 'image_id', 'area', 'iscrowd'
        pass
