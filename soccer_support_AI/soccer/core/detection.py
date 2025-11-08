from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .types import Detection


@dataclass
class DetectorConfig:
    model_name: str = "yolov8n.pt"
    device: str | None = None
    conf: float = 0.3
    target_class: str | None = "person"


class YoloDetector:
    """Thin wrapper around ultralytics YOLO for person detection."""

    def __init__(self, config: DetectorConfig):
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "ultralytics is missing. Install requirements.txt before running detection."
            ) from exc

        self.config = config
        self.model = YOLO(config.model_name)
        self.model.fuse()
        # Build lookup for class filtering once
        names = self.model.model.names if hasattr(self.model, "model") else self.model.names
        if isinstance(names, dict):
            self.class_map = {int(k): v for k, v in names.items()}
        else:
            self.class_map = {i: name for i, name in enumerate(names)}
        self.allowed_classes = self._resolve_allowed_classes()

    def _resolve_allowed_classes(self) -> set[int] | None:
        if not self.config.target_class:
            return None
        target = self.config.target_class.lower()
        return {cid for cid, name in self.class_map.items() if name.lower() == target}

    def detect(self, frame: np.ndarray) -> Sequence[Detection]:
        results = self.model.predict(
            frame,
            conf=self.config.conf,
            device=self.config.device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        detections: List[Detection] = []
        for bbox, score, cls_id in zip(xyxy, scores, classes):
            if self.allowed_classes is not None and cls_id not in self.allowed_classes:
                continue
            detections.append(Detection(tuple(map(float, bbox)), float(score), int(cls_id)))
        return detections
