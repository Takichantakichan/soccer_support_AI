from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass
class Detection:
    bbox: BBox
    score: float
    class_id: int

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "score": self.score,
            "class_id": self.class_id,
        }


@dataclass
class TrackRecord:
    track_id: int
    frame_index: int
    bbox: BBox
    score: float

    @property
    def centroid(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def to_dict(self) -> dict:
        cx, cy = self.centroid
        return {
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "bbox": list(self.bbox),
            "score": self.score,
            "centroid": [cx, cy],
        }


TrackList = List[TrackRecord]
DetectionList = Sequence[Detection]
