"""Utility helpers for detectors (preprocessing and postprocessing)."""
from typing import Tuple, List
import numpy as np


def to_numpy(image) -> np.ndarray:
    """Convert common image formats to HxWxC uint8 numpy array.

    Supports: numpy arrays, PIL Images (if PIL available), tensors with
    .cpu().numpy() for frameworks.
    """
    try:
        import PIL
        from PIL import Image as PILImage  # type: ignore
        # PILImage may be a module; the actual Image class is usually PIL.Image.Image
        if hasattr(PILImage, 'Image'):
            ImageType = PILImage.Image
        else:
            ImageType = PILImage
    except Exception:
        PIL = None
        PILImage = None
        ImageType = None

    if hasattr(image, "numpy"):
        arr = image.numpy()
        # If tensor is CHW, convert to HWC
        if arr.ndim == 3 and arr.shape[0] <= 4:
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.uint8)
    if ImageType is not None and isinstance(image, ImageType):
        return np.array(image)
    if isinstance(image, np.ndarray):
        return image.astype(np.uint8)
    raise TypeError("Unsupported image type for to_numpy")


def xyxy_to_xywh(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2 - x1, y2 - y1)


def xywh_to_xyxy(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def non_max_suppression(detections: List[Tuple[Tuple[float, float, float, float], float]], iou_thresh: float):
    """Very small NMS helper: detections is list of (bbox_xyxy, score).

    Returns indices to keep. This is a minimal pure-numpy NMS for scaffolding.
    """
    if not detections:
        return []
    boxes = np.array([d[0] for d in detections], dtype=np.float32)
    scores = np.array([d[1] for d in detections], dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep
