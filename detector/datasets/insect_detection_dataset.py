"""Dataset utilities for the InsectDetectionDataset in YOLO format.

Provides:
- InsectDetectionDataset: helper to inspect data folders and write a data YAML for
  ultralytics YOLO training.
- TorchYoloDataset: a torch.utils.data.Dataset that converts YOLO-format labels
  into torchvision-style targets (boxes in xyxy pixel coords + labels).
"""

from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union
import os
import tempfile
import random
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from detector.utils import yolo_to_xyxy, build_mosaic, random_resize


class InsectDetectionDataset(torch.utils.data.Dataset):
    """Wrapper for a YOLO-format dataset on disk.

    Expected layout:

      root/
        images/
          train/
          val/
        labels/
          train/
          val/

    Methods allow generating a data YAML for ultralytics and creating a
    Torch dataset for Faster R-CNN training.
    """

    def __init__(self, root: str, split: str = "train"):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"

        self.images = glob(str(self.images_dir / "**" / "*.jpg"), recursive=True)
        self.images = sorted(self.images)

        self.labels = glob(str(self.labels_dir / "**" / "*.txt"), recursive=True)
        self.labels = sorted(self.labels)

        assert len(self.images) > 0, "No images found in dataset"
        assert len(self.images) == len(
            self.labels
        ), "Number of images and labels must match"
        assert (
            self.images[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]
            == self.labels[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        ), "Image and label filenames do not match (0)"
        assert (
            self.images[-1].rsplit("/", 1)[-1].rsplit(".", 1)[0]
            == self.labels[-1].rsplit("/", 1)[-1].rsplit(".", 1)[0]
        ), "Image and label filenames do not match (-1)"

        self.num_classes = len(self.classes())

        if split not in ["train", "val"]:
            raise ValueError("split must be 'train' or 'val'")
        # filter images/labels to the split: train 80%, val 20%
        n = len(self.images)
        split_idx = int(n * 0.8)
        # random shuffle with fixed seed
        random.seed(42)
        indices = list(range(n))
        random.shuffle(indices)
        if split == "train":
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]
        self.images = [self.images[i] for i in selected_indices]
        self.labels = [self.labels[i] for i in selected_indices]

    def classes(self) -> List[str]:
        """Infer class indices from label files and return sorted list of names (strings).

        The returned list contains string class names; by default we return the
        numeric indices as strings (e.g., ['0','1','2']).
        """
        cls_set = set()
        if not self.labels_dir.exists():
            return []
        for p in self.labels_dir.rglob("*.txt"):
            try:
                with open(p, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_set.add(int(parts[0]))
            except Exception:
                continue
        if not cls_set:
            return []
        return [str(i) for i in sorted(cls_set)]

    @staticmethod
    def make_data_yaml(
        out_path: Optional[str] = None,
        train_dataset: "InsectDetectionDataset" = None,
        val_dataset: "InsectDetectionDataset" = None,
    ) -> str:
        """Create a YAML file compatible with ultralytics YOLO training.

        Usage:
          - Pass `train_dataset` and `val_dataset` (instances of this class) to generate
            'train' and 'val' paths from their roots (prefer images/train and images/val).
          - If neither dataset is provided, this will try to infer a default dataset
            located at 'detector/data/InsectDetectionDataset'.

        Returns path to the written YAML file.
        """
        # determine train/val paths and class names
        if train_dataset is not None and val_dataset is not None:
            # Prefer explicit lists for ultralytics by writing txt files with absolute image paths
            # This is helpful when images/train and images/val directories do not exist.
            train_list_fd, train_list_path = tempfile.mkstemp(suffix=".txt")
            val_list_fd, val_list_path = tempfile.mkstemp(suffix=".txt")
            os.close(train_list_fd)
            os.close(val_list_fd)
            # write absolute paths
            with open(train_list_path, "w") as f:
                for p in train_dataset.images:
                    f.write(str(Path(p).resolve()) + "\n")
            with open(val_list_path, "w") as f:
                for p in val_dataset.images:
                    f.write(str(Path(p).resolve()) + "\n")
            train_str = train_list_path
            val_str = val_list_path
            names = train_dataset.classes() or val_dataset.classes()
        else:
            # fallback: try to locate default data folder in repository
            default_root = Path("detector/data/InsectDetectionDataset")
            train_path = default_root / "images" / "train"
            val_path = default_root / "images" / "val"
            train_str = str(train_path) if train_path.exists() else str(default_root / "images")
            val_str = str(val_path) if val_path.exists() else str(default_root / "images")
            # infer class names by scanning labels
            labels_dir = default_root / "labels"
            cls_set = set()
            if labels_dir.exists():
                for p in labels_dir.rglob("*.txt"):
                    try:
                        with open(p, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if not parts:
                                    continue
                                cls_set.add(int(parts[0]))
                    except Exception:
                        continue
            names = [str(i) for i in sorted(cls_set)] if cls_set else []

        if not names:
            # default to 80 classes if none found
            names = [str(i) for i in range(80)]

        data = {
            "train": train_str,
            "val": val_str,
            "nc": len(names),
            "names": names,
        }

        if out_path is None:
            fd, path = tempfile.mkstemp(suffix=".yaml")
            os.close(fd)
            out_path = path
        with open(out_path, "w") as f:
            yaml.safe_dump(data, f)
        return out_path

    def __getitem__(self, index):
        img_path = self.images[index]
        label_path = self.labels[index]
        img = Image.open(img_path).convert("RGB")
        # example 0 0.599760 0.771121 0.052621 0.027587
        w, h = img.size
        boxes = []
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                clsid = int(parts[0])
                box = [float(x) for x in parts[1:5]]
                # xyxy = yolo_to_xyxy(box, w, h)
                boxes.append(box)
                labels.append(clsid)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return img, target

    def __len__(self):
        return len(self.images)


if __name__ == "__main__":
    # simple test
    dataset = InsectDetectionDataset("detector/data/InsectDetectionDataset")
    print("Number of images:", len(dataset))
    print("Classes:", dataset.classes())
    yaml_path = dataset.make_data_yaml()
    print("Wrote data YAML to:", yaml_path)
    for i, (img, target) in enumerate(dataset):
        if target["boxes"].size(0) > 0:
            print(f"Image {i}: size={img.size}, target={target}")
            break
