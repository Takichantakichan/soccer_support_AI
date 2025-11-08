from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Detection, TrackRecord


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.3
    max_age: int = 30  # frames to keep track alive without matches
    min_hits: int = 1  # frames before track considered reliable


@dataclass
class TrackState:
    track_id: int
    bbox: Tuple[float, float, float, float]
    score: float
    last_frame: int
    hits: int = 1
    age: int = 0

    def to_record(self, frame_index: int) -> TrackRecord:
        return TrackRecord(
            track_id=self.track_id,
            frame_index=frame_index,
            bbox=self.bbox,
            score=self.score,
        )


class IOUTracker:
    """Lightweight tracker that links detections by maximizing IoU overlap."""

    def __init__(self, config: TrackerConfig | None = None):
        self.config = config or TrackerConfig()
        self.tracks: Dict[int, TrackState] = {}
        self.next_id = 1

    def _bbox_iou(self, box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter_area / max(area_a + area_b - inter_area, 1e-6)

    def _build_cost_matrix(self, detections: Sequence[Detection]):
        track_boxes = [track.bbox for track in self.tracks.values()]
        det_boxes = [det.bbox for det in detections]
        if not track_boxes or not det_boxes:
            return np.zeros((len(track_boxes), len(det_boxes)), dtype=float)
        cost = np.zeros((len(track_boxes), len(det_boxes)), dtype=float)
        for i, t_box in enumerate(track_boxes):
            for j, d_box in enumerate(det_boxes):
                iou = self._bbox_iou(t_box, d_box)
                cost[i, j] = 1.0 - iou
        return cost

    def _match(self, detections: Sequence[Detection]):
        if not self.tracks or not detections:
            return [], set(range(len(self.tracks))), set(range(len(detections)))
        cost = self._build_cost_matrix(detections)
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        track_items = list(self.tracks.items())
        for row, col in zip(row_ind, col_ind):
            track_id, track = track_items[row]
            iou = 1.0 - cost[row, col]
            if iou < self.config.iou_threshold:
                continue
            matches.append((track_id, col))
            unmatched_tracks.discard(row)
            unmatched_detections.discard(col)
        return matches, unmatched_tracks, unmatched_detections

    def update(self, detections: Sequence[Detection], frame_index: int) -> List[TrackRecord]:
        records: List[TrackRecord] = []
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update matched tracks
        track_items = list(self.tracks.items())
        for track_id, det_idx in matches:
            detection = detections[det_idx]
            track = self.tracks[track_id]
            track.bbox = detection.bbox
            track.score = detection.score
            track.last_frame = frame_index
            track.hits += 1
            track.age = 0
            records.append(track.to_record(frame_index))

        # Age unmatched tracks and remove stale ones
        for idx in unmatched_tracks:
            track_id, track = track_items[idx]
            track.age += 1
            if track.age > self.config.max_age:
                del self.tracks[track_id]

        # Spawn new tracks
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                score=detection.score,
                last_frame=frame_index,
            )
            records.append(self.tracks[track_id].to_record(frame_index))

        min_hits = self.config.min_hits
        return [r for r in records if self.tracks[r.track_id].hits >= min_hits]
