"""YOLOv11 detector scaffold.

YOLOv11 is included as a placeholder. Replace implementation details with the
actual YOLOv11 repo or package you intend to use. The interface matches
DetectorBase.
"""
from typing import Any, List, Optional
from detector.models.base import DetectorBase, Detection
from detector.models.utils import to_numpy


class YOLOv11Detector(DetectorBase):
    """YOLOv11 detector wrapper scaffold using ultralytics-like API.

    This scaffold mirrors the `YOLOv5Detector` design and assumes the
    ultralytics `YOLO` class or a compatible API is available. Replace with
    a concrete YOLOv11 implementation as needed.
    """

    def __init__(self, model: Any = None, device: str = "cpu"):
        self.model = model
        self.device = device

    @classmethod
    def create_model(cls, device: str = "cpu", model_name: str = "yolov11", pretrained: bool = True, num_classes: Optional[int] = None) -> "YOLOv11Detector":
        """Create a YOLOv11 model using torch.hub (ultralytics/yolov5) as fallback.

        This mirrors the YOLOv5 scaffold: we try to load a model via
        `torch.hub.load('ultralytics/yolov5', model_name)`. The `model_name` can
        be a yolov5 model id such as 'yolov5s', or a path to custom weights.
        For a true YOLOv11 implementation replace this with the appropriate
        repository or package loader.
        """
        try:
            import torch
        except Exception:
            raise RuntimeError("PyTorch is required to create YOLOv11Detector")

        # For now use ultralytics/yolov5 hub as a pragmatic fallback for model
        # creation. Replace with actual YOLOv11 repo when available.
        model = None
        try:
            model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=pretrained)
            if hasattr(model, "to"):
                model.to(device)
        except Exception:
            # Leave model as None; user will get a clear error at runtime
            model = None

        if num_classes is not None and model is not None:
            # We can't reliably modify number of classes on the hub model here;
            # leave a non-fatal warning and attach attribute for user reference.
            try:
                model.num_classes = int(num_classes)
            except Exception:
                pass

        return cls(model=model, device=device)

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True) -> None:
        if self.model is None:
            raise RuntimeError("Model not created or ultralytics not installed")
        try:
            if hasattr(self.model, 'load'):
                self.model.load(checkpoint_path)
                return
            import torch

            state = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state, dict) and 'model_state_dict' in state:
                state = state['model_state_dict']
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']

            if hasattr(self.model, 'model') and hasattr(self.model.model, 'load_state_dict'):
                m = self.model.model
                target_keys = set(m.state_dict().keys())
                if not isinstance(state, dict):
                    raise RuntimeError('Loaded checkpoint is not a state dict')

                def _best_mapping(sd: dict):
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
                if match_count <= 0 and 'model' in state and isinstance(state['model'], dict):
                    mapped_state, match_count = _best_mapping(state['model'])
                if match_count <= 0:
                    m.load_state_dict(state, strict=strict)
                else:
                    m.load_state_dict(mapped_state, strict=strict)
                return
            raise RuntimeError('Unable to load checkpoint into YOLOv11 model')
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

    def predict(self, image: Any, conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> List[Detection]:
        if self.model is None:
            raise RuntimeError("Model not created or ultralytics not installed")
        arr = to_numpy(image)
        try:
            # If model exposes a predict API (ultralytics), use it
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

            # fallback: hub/callable
            res = self.model(arr)
            if hasattr(res, 'xyxy'):
                out = res.xyxy[0]
                detections = []
                for row in out.tolist():
                    conf = float(row[4])
                    if conf < conf_thresh:
                        continue
                    detections.append(Detection(bbox=(row[0], row[1], row[2], row[3]), score=conf, label=str(int(row[5]))))
                return detections
        except Exception as e:
            raise RuntimeError(f"YOLOv11 inference failed: {e}")


def __test__():
    """Small smoke test: create, infer, save, load, infer again."""
    import tempfile
    import os
    from urllib.request import urlretrieve
    from PIL import Image

    IMG_URL = "https://ultralytics.com/images/zidane.jpg"
    out_img = '/tmp/yolov11_zidane.png'

    # create model
    model = YOLOv11Detector.create_model(device='cpu', model_name='yolov5s', pretrained=True)
    if model.model is None:
        print('YOLOv11Detector: no model available (torch.hub load failed)')
        return

    # download image
    fd, img_path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    try:
        urlretrieve(IMG_URL, img_path)
        img = Image.open(img_path).convert('RGB')

        print('Running initial inference on zidane.jpg...')
        dets1 = model.predict(img, conf_thresh=0.1)
        print('Detections before save:', dets1)

        # Save weights (try model.save if available, otherwise save state_dict)
        fd2, path = tempfile.mkstemp(suffix='.pt')
        os.close(fd2)
        try:
            saved = False
            if hasattr(model.model, 'save'):
                try:
                    model.model.save(path)
                    saved = True
                except Exception:
                    saved = False
            if not saved:
                try:
                    import torch as _torch

                    if hasattr(model.model, 'state_dict'):
                        _torch.save(model.model.state_dict(), path)
                        saved = True
                    elif hasattr(model, 'state_dict'):
                        _torch.save(model.state_dict(), path)
                        saved = True
                except Exception:
                    saved = False

            print('Saved checkpoint:', saved, path)

            # Load into new instance and run inference again
            model2 = YOLOv11Detector.create_model(device='cpu', model_name='yolov5s', pretrained=True)
            try:
                model2.load_checkpoint(path)
                dets2 = model2.predict(img, conf_thresh=0.1)
                print('Detections after load:', dets2)
                try:
                    from detector.visualize import draw_detections
                    draw_detections(img, dets2, out_path=out_img)
                    print('Saved visualization to', out_img)
                except Exception:
                    pass
            except Exception as e:
                print('Load or inference failed on reloaded model:', e)
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


if __name__ == '__main__':
    __test__()
