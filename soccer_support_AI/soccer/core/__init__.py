"""Core utilities for soccer_support_AI pipeline."""

from .types import Detection, TrackRecord
from .detection import YoloDetector
from .tracking import IOUTracker
from .homography import (
    load_point_pairs,
    compute_homography,
    save_homography,
    load_homography,
    project_points,
)
from .warp import project_track_records
from .metrics import ExpectedThreatTable, compute_xt

__all__ = [
    "Detection",
    "TrackRecord",
    "YoloDetector",
    "IOUTracker",
    "load_point_pairs",
    "compute_homography",
    "save_homography",
    "load_homography",
    "project_points",
    "project_track_records",
    "ExpectedThreatTable",
    "compute_xt",
]
