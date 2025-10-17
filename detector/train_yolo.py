import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import torch
from PIL import Image

from detector.datasets.insect_detection_dataset import InsectDetectionDataset
from detector.models.yolov5_model import YOLOv5Detector
from detector.utils import yolo_to_xyxy, xyxy_to_coco_bbox, evaluate_coco_map
from detector import visualize


def _extract_boxes_from_results(r):
    """Return list of (xyxy, conf, cls) from an ultralytics Results object or similar."""
    try:
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            return list(zip(xyxy, confs, clss))
    except Exception:
        pass
    # fallback: try attributes xyxy, conf, cls on r
    try:
        xyxy = getattr(r, "xyxy", None)
        confs = getattr(r, "conf", None)
        clss = getattr(r, "cls", None)
        if xyxy is not None:
            import numpy as np

            xyxy = np.array(xyxy)
            confs = np.array(confs) if confs is not None else np.ones(len(xyxy))
            clss = np.array(clss).astype(int) if clss is not None else np.zeros(len(xyxy), dtype=int)
            return list(zip(xyxy, confs, clss))
    except Exception:
        pass
    return []


def train(epochs: int = 50, batch: int = 64, imgsz: int = 1024, vis_n: int = 8):
    """Train YOLOv5 on the InsectDetectionDataset, evaluate with pycocotools and save visualizations.

    Produces:
    - trained weights under runs/train/exp (by ultralytics trainer)
    - predictions JSON (runs/train/exp/predictions.json)
    - visualizations under runs/vis/
    """

    train_dataset = InsectDetectionDataset("detector/data/InsectDetectionDataset", split="train")
    val_dataset = InsectDetectionDataset("detector/data/InsectDetectionDataset", split="val")
    num_classes = train_dataset.num_classes
    model_name = "yolov5mu"
    model = YOLOv5Detector.create_model(num_classes=num_classes, model_name=model_name, pretrained=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    print(f"Dataset images: {len(train_dataset)}, detected classes: {train_dataset.classes()} (nc={num_classes})")

    # create data yaml for ultralytics; pass datasets so make_data_yaml will generate list files
    data_yaml = InsectDetectionDataset.make_data_yaml(train_dataset=train_dataset, val_dataset=val_dataset)
    print("Wrote data yaml:", data_yaml)

    # TRAIN using available API
    print("Starting training...")
    model.train(data=data_yaml, epochs=epochs, batch=batch, imgsz=imgsz, device=str(device))

    # locate weights (best or last) in runs/train
    run_dir = None
    for p in sorted(Path("runs/train").glob("**/"), reverse=True):
        # find directory containing .pt weights
        if p.is_dir() and any(p.glob("*.pt")):
            run_dir = p
            break
    if run_dir is None:
        run_dir = Path("runs/train/exp")
    print("Run directory (guessed):", run_dir)

    # EVALUATION: build GT annotations and run model to get predictions (use val split)
    print("Building ground-truth COCO-style annotations from val set...")
    gt_annotations = []
    images_info: Dict[int, Dict] = {}
    ann_id = 1
    for img_id, img_path in enumerate(tqdm(val_dataset.images, desc="images"), start=1):
        try:
            img = Image.open(img_path)
            w, h = img.size
        except Exception:
            continue
        images_info[img_id] = {"id": img_id, "width": w, "height": h, "file_name": Path(img_path).name}
        # read corresponding label file
        label_path = str(Path(val_dataset.labels[img_id - 1]))
        try:
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    box = [float(x) for x in parts[1:5]]
                    x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
                    coco_bbox = xyxy_to_coco_bbox((x1, y1, x2, y2))
                    gt_annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": int(cls) + 1,
                            "bbox": coco_bbox,
                            "area": coco_bbox[2] * coco_bbox[3],
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
        except Exception:
            continue

    # categories
    names = train_dataset.classes()
    if not names:
        names = [str(i) for i in range(num_classes)]
    categories = [{"id": i + 1, "name": names[i] if i < len(names) else str(i)} for i in range(num_classes)]

    # Run inference on val images and collect predictions
    print("Running inference to collect predictions on val set...")
    pred_annotations = []
    for img_id, img_path in enumerate(tqdm(val_dataset.images, desc="predict"), start=1):
        try:
            results = model.predict(source=str(img_path), imgsz=imgsz, device=str(device), conf=0.001)
            if len(results) == 0:
                continue
            r = results[0]
            items = _extract_boxes_from_results(r)
            for xyxy, conf, cls in items:
                x1, y1, x2, y2 = [float(x) for x in xyxy]
                coco_bbox = xyxy_to_coco_bbox((x1, y1, x2, y2))
                pred_annotations.append(
                    {
                        "image_id": img_id,
                        "category_id": int(cls) + 1,
                        "bbox": coco_bbox,
                        "score": float(conf),
                    }
                )
        except Exception as e:
            print(f"Prediction failed for {img_path}: {e}")
            continue

    # Save predictions json for inspection
    pred_out = Path(run_dir) / "predictions.json"
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    with open(pred_out, "w") as f:
        json.dump(pred_annotations, f)

    # Evaluate using our utility that wraps pycocotools
    print("Evaluating with pycocotools via detector.utils.evaluate_coco_map...")
    try:
        metrics = evaluate_coco_map(gt_annotations, pred_annotations, categories, images_info)
        print("Evaluation metrics:")
        print(metrics)
    except Exception as e:
        print("COCO evaluation failed:", e)

    # VISUALIZE: draw GT and predictions on a small subset
    print("Generating visualizations...")
    vis_dir = Path("runs/vis")
    vis_dir.mkdir(parents=True, exist_ok=True)
    # map image_id -> list of preds/GTs
    preds_by_image = {}
    for p in pred_annotations:
        preds_by_image.setdefault(p["image_id"], []).append(p)

    gts_by_image = {}
    for g in gt_annotations:
        gts_by_image.setdefault(g["image_id"], []).append(g)

    sample_ids = list(images_info.keys())[:vis_n]
    from detector.visualize import GT, Pred, draw_gt_and_preds

    for i, img_id in enumerate(sample_ids):
        info = images_info.get(img_id)
        if not info:
            continue
        img_path_candidates = [p for p in val_dataset.images if Path(p).name == info.get("file_name")]
        if not img_path_candidates:
            continue
        img_path = img_path_candidates[0]
        # prepare GTs
        gts = []
        for g in gts_by_image.get(img_id, []):
            x, y, w, h = g["bbox"]
            gts.append(GT(bbox=(x, y, x + w, y + h), label=str(g["category_id"])))
        # prepare preds
        preds = []
        for p in preds_by_image.get(img_id, []):
            x, y, w, h = p["bbox"]
            preds.append(Pred(bbox=(x, y, x + w, y + h), score=p.get("score", 0.0), label=str(p["category_id"])))
        try:
            img = Image.open(img_path).convert("RGB")
            out_path = vis_dir / f"vis_{img_id}.png"
            draw_gt_and_preds(img, gts, preds, out_path=str(out_path))
        except Exception as e:
            print(f"Visualization failed for image {img_path}: {e}")

    print("Done: training, evaluation and visualization complete.")


if __name__ == "__main__":
    train()
