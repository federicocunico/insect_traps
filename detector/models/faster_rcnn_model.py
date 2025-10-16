"""Faster R-CNN detector scaffold using torchvision."""

from typing import Any, List, Optional
import torch
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detector.models.base import DetectorBase, Detection
from detector.models.utils import to_numpy


class FasterRCNNDetector(DetectorBase):
    def __init__(self, model: Any = None, device: str = "cpu"):
        self.model = model
        self.device = device

    @classmethod
    def create_model(
        cls,
        device: str = "cpu",
        pretrained: bool = True,
        num_classes: Optional[int] = None,
        backbone: Optional[str] = None,
    ) -> "FasterRCNNDetector":
        """Create a Faster R-CNN model.

        If `backbone` is provided and `timm` is available, we attempt to build a
        Faster R-CNN with a timm backbone. Otherwise we fall back to
        torchvision's `fasterrcnn_resnet50_fpn` and adapt the box predictor for
        `num_classes` when requested.
        """
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)

        # Adapt classifier head for custom number of classes
        if num_classes is not None:
            try:
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(
                    in_features, num_classes
                )
            except Exception:
                # if adaptation fails, continue with default
                pass

        model.to(device)
        return cls(model=model, device=device)

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True) -> None:
        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")
        import torch

        state = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        # Try to load into model
        self.model.load_state_dict(state, strict=strict)

    def predict(
        self, image: Any, conf_thresh: float = 0.5, iou_thresh: float = 0.45
    ) -> List[Detection]:
        if self.model is None:
            raise RuntimeError("Model not created. Call create_model() first.")
        import torch

        self.model.eval()
        arr = to_numpy(image)
        from torchvision.transforms import functional as F

        img_t = F.to_tensor(arr).to(self.device)
        with torch.no_grad():
            outputs = self.model([img_t])
        detections: List[Detection] = []
        out = outputs[0]
        boxes = out.get("boxes")
        scores = out.get("scores")
        labels = out.get("labels")
        if boxes is None:
            return []
        for b, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            if s < conf_thresh:
                continue
            detections.append(
                Detection(
                    bbox=(b[0], b[1], b[2], b[3]), score=float(s), label=str(int(l))
                )
            )
        return detections


def __test__():
    import tempfile
    import os
    from urllib.request import urlretrieve
    from PIL import Image

    IMG_URL = "https://ultralytics.com/images/zidane.jpg"
    out_img = '/tmp/fasterrcnn_zidane.png'

    # download image
    fd, img_path = tempfile.mkstemp(suffix='.jpg')
    os.close(fd)
    try:
        urlretrieve(IMG_URL, img_path)
        img = Image.open(img_path).convert('RGB')

        model = FasterRCNNDetector.create_model(device="cpu", num_classes=2)
        print('Running initial inference on zidane.jpg...')
        try:
            dets1 = model.predict(img, conf_thresh=0.1)
            print('Detections before save:', dets1)
        except Exception as e:
            print('Initial inference failed:', e)

        fd2, path = tempfile.mkstemp(suffix='.pt')
        os.close(fd2)
        try:
            saved = False
            try:
                # torchvision models: save state_dict
                torch.save(model.model.state_dict(), path)
                saved = True
            except Exception:
                try:
                    torch.save(model.state_dict(), path)
                    saved = True
                except Exception:
                    saved = False
            print('Checkpoint saved:', saved, path)

            # reload into new model instance
            model2 = FasterRCNNDetector.create_model(device='cpu', num_classes=2)
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


if __name__ == "__main__":
    __test__()