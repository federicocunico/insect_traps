"""Shared utilities for detection: box conversions, IoU, simple augmentations, and COCO evaluation.

This module is used by dataset loaders and trainers to avoid duplicated logic
between YOLO and Faster R-CNN code paths.
"""

from typing import Tuple, List, Callable, Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
import random
import json
import tempfile
import os


def yolo_to_xyxy(
    box: Tuple[float, float, float, float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    xc, yc, w, h = box
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return x1, y1, x2, y2


def xyxy_to_coco_bbox(b: Tuple[float, float, float, float]) -> List[float]:
    x1, y1, x2, y2 = b
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def iou(
    boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]
) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def random_resize(img, boxes, min_size=320, max_size=960, step=32):
    """Randomly resize image and scale boxes accordingly. Size chosen from [min_size, max_size] in steps."""
    import torchvision.transforms.functional as F

    w, h = img.size
    choice = random.randrange(min_size // step, max_size // step + 1) * step
    # preserve aspect ratio: scale shorter side to choice
    if w < h:
        new_w = choice
        new_h = int(h * (choice / w))
    else:
        new_h = choice
        new_w = int(w * (choice / h))
    img2 = img.resize((new_w, new_h))
    scale_x = new_w / w
    scale_y = new_h / h
    boxes2 = []
    for b in boxes:
        x1, y1, x2, y2 = b
        boxes2.append((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y))
    return img2, boxes2


def build_mosaic(images, boxes_list, labels_list=None, input_size=640):
    """Compose 4 images into a mosaic and adjust boxes accordingly.

    images: list of PIL Images
    boxes_list: list of lists of boxes in xyxy pixel coords (same length as images)
    labels_list: optional list of label-lists corresponding to boxes_list

    Returns (mosaic_img: PIL.Image, boxes: List[xyxy], labels: List[int]).
    """
    import math

    # target size
    s = input_size
    mosaic = Image.new("RGB", (s * 2, s * 2), (114, 114, 114))
    # place images at quadrants with random center
    yc = s
    xc = s
    combined_boxes = []
    combined_labels = []
    for i, img in enumerate(images[:4]):
        # resize img to random scale between 0.4 and 1.0 of s
        scale = random.uniform(0.4, 1.0)
        new_w = int(img.width * scale)
        new_h = int(img.height * scale)
        img_r = img.resize((new_w, new_h))
        # pick position
        if i == 0:
            x1a, y1a = max(0, xc - new_w), max(0, yc - new_h)
        elif i == 1:
            x1a, y1a = xc, max(0, yc - new_h)
        elif i == 2:
            x1a, y1a = max(0, xc - new_w), yc
        else:
            x1a, y1a = xc, yc
        mosaic.paste(img_r, (x1a, y1a))
        # adjust boxes
        boxes = boxes_list[i]
        labels = (
            labels_list[i]
            if labels_list is not None and i < len(labels_list)
            else [0] * len(boxes)
        )
        for bi, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            sx = new_w / img.width
            sy = new_h / img.height
            nx1 = x1 * sx + x1a
            ny1 = y1 * sy + y1a
            nx2 = x2 * sx + x1a
            ny2 = y2 * sy + y1a
            combined_boxes.append((nx1, ny1, nx2, ny2))
            combined_labels.append(labels[bi] if bi < len(labels) else 0)
    # final crop to s x s center
    cx = mosaic.width // 2
    cy = mosaic.height // 2
    x0 = max(0, cx - s // 2)
    y0 = max(0, cy - s // 2)
    mosaic_cropped = mosaic.crop((x0, y0, x0 + s, y0 + s))
    # shift boxes according to crop
    final_boxes = []
    final_labels = []
    for bi, b in enumerate(combined_boxes):
        nx1, ny1, nx2, ny2 = b
        nx1 -= x0
        ny1 -= y0
        nx2 -= x0
        ny2 -= y0
        # clip
        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(s, nx2)
        ny2 = min(s, ny2)
        if nx2 - nx1 > 1 and ny2 - ny1 > 1:
            final_boxes.append((nx1, ny1, nx2, ny2))
            final_labels.append(combined_labels[bi])
    return mosaic_cropped, final_boxes, final_labels


def evaluate_coco_map(
    gt_annotations: List[Dict[str, Any]],
    pred_annotations: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
    images_info: Dict[int, Dict[str, int]] = None,
) -> Dict[str, float]:
    """Evaluate COCO mAP using pycocotools. Inputs are lists of annotations/detections in COCO format.

    gt_annotations: list of COCO annotation dicts and images must be included in categories param or provided separately.
    pred_annotations: list of detection dicts with keys: image_id, category_id, bbox, score.
    categories: list of category dicts {'id': int, 'name': str}

    Returns a dict with mAP metrics (if pycocotools available) or raises informative error.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as e:
        raise RuntimeError(
            "pycocotools is required for COCO mAP evaluation. Install with `pip install pycocotools`"
        )

    # Build temporary COCO-style ground truth json
    tmp_gt = tempfile.mkstemp(suffix=".json")[1]
    tmp_dt = tempfile.mkstemp(suffix=".json")[1]

    # Build images dict and annotations list. Ensure we have width/height per image.
    images = {}
    anns = []
    ann_id = 1
    for ann in gt_annotations:
        # ann must include 'image_id', 'bbox', 'category_id'
        image_id = ann["image_id"]
        anns.append(
            dict(
                id=ann_id,
                image_id=image_id,
                category_id=ann["category_id"],
                bbox=ann["bbox"],
                area=ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                iscrowd=ann.get("iscrowd", 0),
            )
        )
        # prefer explicit image info passed in images_info, else use per-ann width/height keys, else infer later
        if images_info and image_id in images_info:
            images[image_id] = {
                "id": image_id,
                "width": int(images_info[image_id].get("width", 0)),
                "height": int(images_info[image_id].get("height", 0)),
            }
        else:
            images[image_id] = {
                "id": image_id,
                "width": int(ann.get("width", 0)),
                "height": int(ann.get("height", 0)),
            }
        ann_id += 1

    # If any image width/height is zero, try to infer from GT boxes (max x+w and y+h per image)
    need_infer = any(
        img.get("width", 0) == 0 or img.get("height", 0) == 0 for img in images.values()
    )
    if need_infer:
        # compute from annotations
        sizes = {}
        for a in anns:
            img_id = a["image_id"]
            x, y, w, h = a["bbox"]
            max_w = sizes.get(img_id, {}).get("w", 0)
            max_h = sizes.get(img_id, {}).get("h", 0)
            max_w = max(max_w, int(x + w))
            max_h = max(max_h, int(y + h))
            sizes.setdefault(img_id, {})["w"] = max_w
            sizes.setdefault(img_id, {})["h"] = max_h
        for img_id, s in sizes.items():
            if images.get(img_id, {}).get("width", 0) == 0:
                images[img_id]["width"] = s.get("w", 0)
            if images.get(img_id, {}).get("height", 0) == 0:
                images[img_id]["height"] = s.get("h", 0)

    gt_json = {
        "images": list(images.values()),
        "annotations": anns,
        "categories": categories,
    }
    with open(tmp_gt, "w") as f:
        json.dump(gt_json, f)

    # write predictions
    with open(tmp_dt, "w") as f:
        json.dump(pred_annotations, f)

    cocoGt = COCO(tmp_gt)
    cocoDt = cocoGt.loadRes(tmp_dt)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # read out mAPs into a dict
    stats = cocoEval.stats.tolist() if hasattr(cocoEval, "stats") else []
    keys = [
        "mAP",
        "mAP_50",
        "mAP_75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "AR_1",
        "AR_10",
        "AR_100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    res = {k: stats[i] if i < len(stats) else None for i, k in enumerate(keys)}

    try:
        os.remove(tmp_gt)
        os.remove(tmp_dt)
    except Exception:
        pass
    return res
