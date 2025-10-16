"""
Abstract base class and shared datatypes for detectors.

This module defines a simple interface that all detectors in this project
should implement. It is intentionally lightweight to allow multiple backends
(PyTorch, ONNX, TensorFlow, custom C++) to be wrapped behind the same API.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict
import abc


@dataclass
class Detection:
    """Represents a single detection result.

    Attributes:
        bbox: (x1, y1, x2, y2) in pixel coordinates or normalized depending on model.
        score: confidence score (0-1).
        label: class label name or id.
        extra: optional backend specific data.
    """
    bbox: Tuple[float, float, float, float]
    score: float
    label: str
    extra: Optional[Dict[str, Any]] = None


class DetectorBase(abc.ABC):
    """Abstract detector interface.

    Implementations must provide ways to create a model, load weights and run
    inference. The signatures are intentionally minimal but include parameters
    for future extensibility.
    """

    @classmethod
    @abc.abstractmethod
    def create_model(cls, device: str = "cpu", **kwargs) -> "DetectorBase":
        """Create and return an instance of the detector configured for `device`.

        Must not load any checkpoint unless path provided via kwargs.
        """

    @abc.abstractmethod
    def load_checkpoint(self, checkpoint_path: str, strict: bool = True) -> None:
        """Load weights from checkpoint into the model instance.

        Implementations should raise if the load fails.
        """

    @abc.abstractmethod
    def predict(self, image: Any, conf_thresh: float = 0.25, iou_thresh: float = 0.45) -> List[Detection]:
        """Run inference on a single image and return list of Detection.

        `image` can be a numpy array, PIL.Image, or a framework tensor depending
        on implementation. Implementations should document expected input types.
        """

    def to(self, device: str):
        """Optional helper to move model to another device. Default is no-op.

        Implementations using frameworks should override this.
        """
        return self
