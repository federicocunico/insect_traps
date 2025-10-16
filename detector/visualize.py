"""Visualization helpers for detector outputs.

Provides a small utility to draw bounding boxes (x1,y1,x2,y2) on images and
save or show the result. The function accepts numpy arrays, PIL Images or
framework tensors and will preserve the original image size when plotting.
"""
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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
