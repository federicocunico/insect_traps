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
    parser.add_argument('--mosaic', action='store_true', help='enable mosaic augmentation dataset')
    parser.add_argument('--multiscale', action='store_true', help='enable random multiscale resizing during training')
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        raise SystemExit(f'Data directory not found: {root}')

    device = torch.device(args.device)

    # datasets + dataloaders
    try:
        # prefer the helper dataset and augmenter if available
        from detector.datasets.insect_detection_dataset import InsectDetectionDataset, TorchYoloDataset, MosaicTorchYoloDataset
        from detector.utils import get_data_augmentation, random_resize

        ds_helper = InsectDetectionDataset(root)
        split_root = ds_helper.split_train_val(train_ratio=0.8)
        root = Path(split_root)
        aug = get_data_augmentation(flip_prob=0.5, color_jitter={'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.1})
        # wrap augmenter to include multiscale if requested
        if args.multiscale:
            def aug_ms(img, boxes):
                img2, boxes2 = random_resize(img, boxes)
                return aug(img2, boxes2)
            aug_use = aug_ms
        else:
            aug_use = aug
        if args.mosaic:
            train_ds = MosaicTorchYoloDataset(root, split='train', transforms=aug_use)
        else:
            train_ds = TorchYoloDataset(root, split='train', transforms=aug_use)
        val_ds = TorchYoloDataset(root, split='val', transforms=None)
    except Exception:
        # fallback to the simple YoloDataset
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

    # Final evaluation & visualization on validation set
    try:
        from detector.visualize import draw_gt_and_preds, GT, Pred
        from detector.utils import evaluate_coco_map, iou as _iou
    except Exception:
        draw_gt_and_preds = None
        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            denom = area_a + area_b - inter
            if denom <= 0:
                return 0.0
            return inter / denom

    # Build COCO-style prediction and GT lists for mAP evaluation
    try:
        from detector.utils import evaluate_coco_map, xyxy_to_coco_bbox
    except Exception:
        evaluate_coco_map = None
        xyxy_to_coco_bbox = None

    model.eval()
    viz_dir = Path('runs/val_viz')
    viz_dir.mkdir(parents=True, exist_ok=True)
    gt_anns = []
    pred_anns = []
    categories = []
    # build category list from dataset classes
    class_ids = train_ds.__class__ if False else None
    # iterate over val set and collect annotations
    img_id = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            imgs = list(img.to(device) for img in images)
            outputs = model(imgs)
            out = outputs[0]
            boxes = out.get('boxes', []).cpu().tolist()
            scores = out.get('scores', []).cpu().tolist()
            labels = out.get('labels', []).cpu().tolist()

            gt_boxes = targets[0]['boxes'].cpu().tolist()
            gt_labels = targets[0]['labels'].cpu().tolist()

            # add GT anns
            for b, l in zip(gt_boxes, gt_labels):
                coco_box = xyxy_to_coco_bbox(tuple(b))
                gt_anns.append({'image_id': img_id, 'category_id': int(l) - 1, 'bbox': coco_box, 'area': coco_box[2] * coco_box[3], 'iscrowd': 0, 'width': int(images[0].shape[2]), 'height': int(images[0].shape[1])})

            # add pred anns
            for b, s, l in zip(boxes, scores, labels):
                coco_box = xyxy_to_coco_bbox(tuple(b))
                pred_anns.append({'image_id': img_id, 'category_id': int(l) - 1, 'bbox': coco_box, 'score': float(s)})

            # save visualization
            try:
                if draw_gt_and_preds:
                    gts = [GT(bbox=tuple(map(float, b)), label=str(int(l - 1))) for b, l in zip(gt_boxes, gt_labels)]
                    preds = [Pred(bbox=tuple(map(float, b)), score=float(s), label=str(int(l) - 1)) for b, s, l in zip(boxes, scores, labels)]
                    out_path = str(viz_dir / f'val_{i:04d}.png')
                    draw_gt_and_preds(images[0], gts, preds, out_path=out_path, iou_thresh=0.5)
            except Exception:
                pass
            img_id += 1

    if evaluate_coco_map:
        # construct categories from unique category ids found
        cat_ids = sorted({ann['category_id'] for ann in gt_anns})
        categories = [{'id': int(cid), 'name': str(cid)} for cid in cat_ids]
        try:
            # build images_info from GT anns
            images_info = {}
            for g in gt_anns:
                iid = g['image_id']
                images_info.setdefault(iid, {})['width'] = g.get('width', images_info.get(iid, {}).get('width', 0))
                images_info.setdefault(iid, {})['height'] = g.get('height', images_info.get(iid, {}).get('height', 0))
            res = evaluate_coco_map(gt_anns, pred_anns, categories, images_info=images_info)
            print('COCO mAP results:', res)
        except Exception as e:
            print('COCO evaluation failed:', e)
    else:
        print('pycocotools not installed; skipping COCO mAP evaluation')


if __name__ == '__main__':
    main()
