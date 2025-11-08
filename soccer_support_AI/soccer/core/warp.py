from __future__ import annotations

from typing import Iterable, List

import numpy as np

from .homography import project_points
from .types import TrackRecord


def project_track_records(records: Iterable[TrackRecord], H: np.ndarray) -> List[dict]:
    records = list(records)
    if not records:
        return []
    centroids = [rec.centroid for rec in records]
    projected = project_points(H, centroids)
    payload = []
    for rec, (px, py) in zip(records, projected):
        data = rec.to_dict()
        data.update({"pitch": [float(px), float(py)]})
        payload.append(data)
    return payload
