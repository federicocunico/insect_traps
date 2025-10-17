"""Visualization helpers for detector outputs.

Provides a small utility to draw bounding boxes (x1,y1,x2,y2) on images and
save or show the result. The function accepts numpy arrays, PIL Images or
framework tensors and will preserve the original image size when plotting.
"""
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from dataclasses import dataclass
from detector.utils import iou as _iou

try:
    # local helper from models
    from detector.models.utils import to_numpy
except Exception:
    # fallback minimal converter
    def to_numpy(x):
        if hasattr(x, 'numpy'):
            a = x.numpy()
            if a.ndim == 3 and a.shape[0] <= 4:
                a = np.transpose(a, (1, 2, 0))
            return a.astype(np.uint8)
        if isinstance(x, np.ndarray):
            return x.astype(np.uint8)
        try:
            from PIL import Image as PILImage

            if isinstance(x, PILImage.Image):
                return np.array(x)
        except Exception:
            pass
        raise TypeError('Unsupported image type for to_numpy')


def draw_detections(image, detections: List[object], out_path: Optional[str] = None, show: bool = False) -> Image.Image:
    """Draw detections on `image` and optionally save/show it.

    - image: numpy HxWxC uint8, PIL Image, or tensor convertible with to_numpy.
    - detections: iterable of objects with fields `bbox` = (x1,y1,x2,y2),
      `score`, and `label` (or use str(item)).
    - out_path: if provided, save PNG to this path.
    - show: if True, call Image.show().

    Returns PIL.Image.
    """
    arr = to_numpy(image)
    # Ensure HWC
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        # try to scale if floats in 0..1
        if arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    h, w = arr.shape[:2]
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        try:
            bbox = det.bbox
            score = getattr(det, 'score', None)
            label = getattr(det, 'label', None)
        except Exception:
            # fallback if detection is a tuple/list
            try:
                bbox, score = det
                label = ''
            except Exception:
                continue
        x1, y1, x2, y2 = bbox
        # handle normalized coords
        if 0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= y2 <= 1.0:
            x1 = int(x1 * w)
            x2 = int(x2 * w)
            y1 = int(y1 * h)
            y2 = int(y2 * h)
        else:
            x1 = int(round(x1))
            x2 = int(round(x2))
            y1 = int(round(y1))
            y2 = int(round(y2))

        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        text = ''
        if label is not None:
            text = str(label)
        if score is not None:
            if text:
                text = f"{text}: {score:.2f}"
            else:
                text = f"{score:.2f}"
        if text:
            text_pos = (x1 + 4, y1 + 4)
            draw.text(text_pos, text, fill='yellow', font=font)

    if out_path:
        try:
            img.save(out_path)
        except Exception:
            pass
    if show:
        try:
            img.show()
        except Exception:
            pass
    return img


def _iou(boxA, boxB):
    # box: [x1,y1,x2,y2]
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


@dataclass
class GT:
    bbox: Tuple[float, float, float, float]
    label: str


@dataclass
class Pred:
    bbox: Tuple[float, float, float, float]
    score: float
    label: str


def draw_gt_and_preds(image, gts: List[GT], preds: List[Pred], out_path: Optional[str] = None, iou_thresh: float = 0.5) -> Image.Image:
    """Draw ground-truth boxes in green and predictions in red; matched predictions are magenta.

    Also returns the PIL image and writes out_path if provided.
    """
    arr = to_numpy(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 and arr.min() >= 0.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)

    h, w = arr.shape[:2]
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    matched = set()
    # match preds to gts by IoU and label
    for pi, p in enumerate(preds):
        best_iou = 0.0
        best_gi = None
        for gi, g in enumerate(gts):
            if p.label != g.label:
                continue
            iouv = _iou(p.bbox, g.bbox)
            if iouv > best_iou:
                best_iou = iouv
                best_gi = gi
        color = 'red'
        if best_iou >= iou_thresh:
            color = 'magenta'
            matched.add(best_gi)
        x1, y1, x2, y2 = p.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{p.label}:{p.score:.2f}"
        draw.text((x1 + 4, y1 + 4), text, fill='yellow', font=font)

    # draw GTs that weren't matched in green
    for gi, g in enumerate(gts):
        if gi in matched:
            continue
        x1, y1, x2, y2 = g.bbox
        draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
        draw.text((x1 + 4, y1 + 4), str(g.label), fill='white', font=font)

    if out_path:
        try:
            img.save(out_path)
        except Exception:
            pass
    return img
