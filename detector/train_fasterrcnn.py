"""Train a Faster R-CNN detector using torchvision on a YOLO-format dataset.

Dataset layout expected (YOLO format):

  dataset/
    images/
      train/
      val/
    labels/
      train/
      val/

Label files are text files with lines: <class> <x_center> <y_center> <width> <height>
normalized to image size.

This script implements a minimal Dataset that reads YOLO labels and converts
them to torchvision detection targets (boxes in xyxy pixel coords + labels).
"""
import argparse
from pathlib import Path
import os
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


def yolo_to_xyxy(box, img_w, img_h):
    # box: [x_center, y_center, w, h] normalized
    xc, yc, w, h = box
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


class YoloDataset(Dataset):
    def __init__(self, root: Path, split: str = 'train', transforms=None):
        self.img_dir = root / 'images' / split
        self.lbl_dir = root / 'labels' / split
        self.images = sorted([p for p in self.img_dir.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl_path = self.lbl_dir / (img_path.stem + '.txt')
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        boxes = []
        labels = []
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:5]))
                    xyxy = yolo_to_xyxy(coords, w, h)
                    boxes.append(xyxy)
                    labels.append(cls + 1)  # reserve 0 for background in torchvision

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes: int):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


def evaluate(model, data_loader, device):
    # minimal evaluation: run inference and return number of detections
    model.eval()
    total = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for out in outputs:
                total += len(out.get('boxes', []))
    return total


def main():
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on YOLO-format dataset')
    parser.add_argument('--data', '-d', required=True, help='Path to dataset root (images/labels)')
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--batch', '-b', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out', default='runs/fasterrcnn.pth')
    parser.add_argument('--num-classes', type=int, default=2, help='number of classes INCLUDING background (i.e., 2 => 1 object class + background)')
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        raise SystemExit(f'Data directory not found: {root}')

    device = torch.device(args.device)

    # datasets + dataloaders
    train_ds = YoloDataset(root, split='train')
    val_ds = YoloDataset(root, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = get_model(args.num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        train_one_epoch(model, optimizer, train_loader, device)
        dets = evaluate(model, val_loader, device)
        print(f'Validation detections (count): {dets}')
        # save checkpoint each epoch
        out_path = args.out
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save({'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, out_path)


if __name__ == '__main__':
    main()
