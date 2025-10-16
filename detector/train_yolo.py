"""Train a YOLO model using the ultralytics package.

This script expects data arranged in YOLO format under a directory like:

  dataset/
    images/
      train/
      val/
    labels/
      train/
      val/

It will generate a temporary data YAML that points to the image folders and a
small class list inferred from label files if available. The script uses the
ultralytics.YOLO API when installed.

Note: ultralytics must be installed in the environment used to run this.
"""
import argparse
import os
import tempfile
from pathlib import Path
from typing import List


def find_classes_from_labels(labels_dir: Path) -> List[str]:
    classes = set()
    if not labels_dir.exists():
        return []
    for p in labels_dir.rglob('*.txt'):
        try:
            with p.open('r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    classes.add(int(parts[0]))
        except Exception:
            continue
    if not classes:
        return []
    # Return sorted class names as strings of their indices (user can replace with names)
    return [str(i) for i in sorted(classes)]


def make_data_yaml(root: Path, names: List[str], imgsz: int = 640) -> str:
    # Expect structure: root/images/train, root/images/val
    train = root / 'images' / 'train'
    val = root / 'images' / 'val'
    data = {
        'train': str(train) if train.exists() else str(root / 'images'),
        'val': str(val) if val.exists() else str(root / 'images'),
        'nc': len(names),
        'names': names,
    }
    fd, path = tempfile.mkstemp(suffix='.yaml')
    os.close(fd)
    import yaml

    with open(path, 'w') as f:
        yaml.safe_dump(data, f)
    return path


def main():
    parser = argparse.ArgumentParser(description='Train YOLO model (ultralytics) on YOLO-format dataset')
    parser.add_argument('--data', '-d', required=True, help='Path to dataset root (images/labels)')
    parser.add_argument('--model', '-m', default='yolov8n.pt', help='ultralytics model name or weights (e.g., yolov8n.pt)')
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--batch', '-b', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--project', default='runs/train', help='project dir to save runs')
    parser.add_argument('--name', default='exp', help='run name')
    parser.add_argument('--device', default='', help='cuda device or cpu (empty => automatic)')
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        raise SystemExit(f'Data directory not found: {root}')

    # infer class list
    names = find_classes_from_labels(root / 'labels')
    if not names:
        # default to 80 COCO classes as names will be indices; user can override
        names = [str(i) for i in range(80)]

    data_yaml = make_data_yaml(root, names, imgsz=args.imgsz)
    print('Using data yaml:', data_yaml)

    # Import ultralytics lazily
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit('ultralytics package is required to run this script: pip install ultralytics')

    # Create model and start training
    model = YOLO(args.model)
    train_args = dict(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device or None,
    )
    print('Starting training with args:', train_args)
    model.train(**train_args)


if __name__ == '__main__':
    main()
