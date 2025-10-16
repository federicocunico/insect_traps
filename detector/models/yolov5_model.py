"""YOLOv5 detector wrapper scaffold.

This file contains a thin wrapper around a YOLOv5 model. It purposely avoids
pinning to a specific implementation; by default it tries to load from
`torch.hub` if available. The methods implement the DetectorBase interface.
"""

from typing import Any, List, Optional

import torch
from detector.models.base import DetectorBase, Detection
from detector.models.utils import to_numpy


class YOLOv5Detector(DetectorBase):
    """YOLO detector using the `ultralytics` package when available.

    The ultralytics package exposes a simple model API where the model can be
    created by specifying a model name and `pretrained` flag, and provides
    inference helpers. We prefer using that over `torch.hub`.
    """

    def __init__(self, model: Any = None, device: str = "cpu"):
        self.model = model
        self.device = device

    @classmethod
    def create_model(
        cls,
        device: str = "cpu",
        model_name: str = "yolov5s",
        pretrained: bool = True,
        num_classes: Optional[int] = None,
    ) -> "YOLOv5Detector":
        """Create a YOLO model via ultralytics if installed.

        Args:
            device: device string (e.g., 'cpu' or 'cuda').
            model_name: ultralytics model name or path.
            pretrained: whether to load pretrained weights.
            num_classes: optional number of output classes; if provided the
                model will be adapted when supported.
        """
        if not pretrained:
            raise NotImplementedError("Non-pretrained YOLO models not supported yet")

        # Create via ultralytics YOLO interface. The API accepts model names
        # like 'yolov5s', or a path to a YAML/config. YOLO(...) returns a model
        # object with .predict/.train/.load methods.
        model = torch.hub.load("ultralytics/yolov5", model_name)

        if num_classes is not None:
            # number of classes is handled during training using ultralytics API. We can't edit the model directly.
            print(
                "Warning: num_classes parameter is ignored for YOLOv5Detector. Adjust during training/inference."
            )
            model.num_classes = num_classes  # for reference only
        try:
            # Move to device when possible
            if hasattr(model, "to"):
                model.to(device)
        except Exception:
            # not fatal
            pass
        return cls(model=model, device=device)

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True) -> None:
        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")
        try:
            # ultralytics YOLO has .load() which accepts path
            if hasattr(self.model, "load"):
                self.model.load(checkpoint_path)
                return
            # fallback: try torch load
            import torch

            state = torch.load(checkpoint_path, map_location=self.device)
            # common checkpoints contain nested keys like {'model_state_dict': {...}}
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']

            # If the wrapped model has an inner .model with load_state_dict, try to
            # normalize keys in the checkpoint to match the target model's keys.
            if hasattr(self.model, "model") and hasattr(self.model.model, "load_state_dict"):
                m = self.model.model
                target_keys = set(m.state_dict().keys())
                if not isinstance(state, dict):
                    raise RuntimeError("Loaded checkpoint is not a state dict")

                def _best_mapping(sd: dict):
                    # Try several common prefixes to strip to match target keys
                    prefixes = ['', 'model.', 'model.model.', 'module.', 'model.module.']
                    best = None
                    best_count = -1
                    for p in prefixes:
                        mapped = {}
                        for k, v in sd.items():
                            nk = k
                            if nk.startswith(p):
                                nk = nk[len(p):]
                            mapped[nk] = v
                        count = len(set(mapped.keys()) & target_keys)
                        if count > best_count:
                            best_count = count
                            best = mapped
                    return best, best_count

                mapped_state, match_count = _best_mapping(state)
                if match_count <= 0:
                    # No matching keys — try to use nested 'model' key
                    if 'model' in state and isinstance(state['model'], dict):
                        mapped_state, match_count = _best_mapping(state['model'])
                if match_count <= 0:
                    # give up, try direct load and let torch report details
                    m.load_state_dict(state, strict=strict)
                else:
                    m.load_state_dict(mapped_state, strict=strict)
                return
            raise RuntimeError("Unable to load checkpoint into ultralytics model")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    def predict(
        self, image: Any, conf_thresh: float = 0.25, iou_thresh: float = 0.45
    ) -> List[Detection]:
        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")
        arr = to_numpy(image)
        try:
            # Prefer structured predict API if available (ultralytics)
            if hasattr(self.model, 'predict'):
                results = self.model.predict(arr, conf=conf_thresh, iou=iou_thresh)
                detections: List[Detection] = []
                for r in results:
                    boxes = getattr(r, 'boxes', None)
                    if boxes is None:
                        continue
                    xyxy = getattr(boxes, 'xyxy', None)
                    confs = getattr(boxes, 'conf', None)
                    clss = getattr(boxes, 'cls', None)
                    if xyxy is None:
                        continue
                    for i in range(len(xyxy)):
                        conf = float(confs[i]) if confs is not None else 0.0
                        if conf < conf_thresh:
                            continue
                        b = xyxy[i].tolist() if hasattr(xyxy[i], 'tolist') else tuple(xyxy[i])
                        clsid = int(clss[i]) if clss is not None else 0
                        detections.append(Detection(bbox=(b[0], b[1], b[2], b[3]), score=conf, label=str(clsid)))
                return detections

            # Fallback: many hub/models are callable and return an object with xyxy
            res = self.model(arr)
            if hasattr(res, 'xyxy'):
                out = res.xyxy[0]
                detections = []
                # out rows: [x1, y1, x2, y2, conf, cls]
                for row in out.tolist():
                    conf = float(row[4])
                    if conf < conf_thresh:
                        continue
                    detections.append(Detection(bbox=(row[0], row[1], row[2], row[3]), score=conf, label=str(int(row[5]))))
                return detections
        except Exception as e:
            raise RuntimeError(f"YOLO inference failed: {e}")


def __test__():
    model = YOLOv5Detector.create_model(
        device="cpu", model_name="yolov5m", pretrained=True, num_classes=2
    )
    dummy_img = torch.randn(3, 512, 512, dtype=torch.float32)
    dets = model.predict(dummy_img, conf_thresh=0.1)
    print(dets)


if __name__ == "__main__":
    # smoke test: download a test image (Zidane), run infer, save, reload checkpoint and infer again
    import tempfile
    import os
    from urllib.request import urlretrieve
    from PIL import Image
    import numpy as np

    IMG_URL = "https://ultralytics.com/images/zidane.jpg"
    out_img = '/tmp/yolov5_zidane.png'

    # download image to temp file
    fd, img_path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    try:
        urlretrieve(IMG_URL, img_path)
        img = Image.open(img_path).convert('RGB')
        # create model and run inference
        model = YOLOv5Detector.create_model(device="cpu", model_name="yolov5m", pretrained=True, num_classes=2)
        print('Running initial inference on zidane.jpg...')
        try:
            dets = model.predict(img, conf_thresh=0.1)
            print('Detections:', dets)
        except Exception as e:
            print('Initial inference failed:', e)

        # save checkpoint
        fd2, path = tempfile.mkstemp(suffix='.pt')
        os.close(fd2)
        try:
            saved = False
            try:
                if hasattr(model.model, 'save'):
                    model.model.save(path)
                    saved = True
            except Exception:
                saved = False
            if not saved:
                try:
                    torch.save(model.model.state_dict(), path)
                    saved = True
                except Exception:
                    saved = False
            print('Checkpoint saved:', saved, path)

            # reload into new instance and run inference again
            model2 = YOLOv5Detector.create_model(device='cpu', model_name='yolov5m', pretrained=True)
            try:
                model2.load_checkpoint(path)
                dets_after = model2.predict(img, conf_thresh=0.1)
                print('Detections after reload:', dets_after)
                try:
                    from detector.visualize import draw_detections
                    draw_detections(img, dets_after, out_path=out_img)
                    print('Saved visualization to', out_img)
                except Exception as _:
                    pass
            except Exception as e:
                print('Reload failed:', e)
        finally:
            try:
                os.remove(path)
            except Exception:
                pass
    finally:
        try:
            os.remove(img_path)
        except Exception:
            pass
