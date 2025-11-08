from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml

from .types import Point


@dataclass
class PointPair:
    image: Point
    pitch: Point


def load_point_pairs(csv_path: str | Path) -> List[PointPair]:
    df = pd.read_csv(csv_path)
    required = {"image_x", "image_y", "pitch_x", "pitch_y"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required}")
    pairs = [
        PointPair((row.image_x, row.image_y), (row.pitch_x, row.pitch_y))
        for row in df.itertuples()
    ]
    return pairs


def compute_homography(pairs: Iterable[PointPair]) -> np.ndarray:
    pairs = list(pairs)
    if len(pairs) < 4:
        raise ValueError("Need at least four point pairs to compute homography")
    image_pts = np.array([p.image for p in pairs], dtype=np.float32)
    pitch_pts = np.array([p.pitch for p in pairs], dtype=np.float32)
    H, mask = cv2.findHomography(image_pts, pitch_pts, cv2.RANSAC)
    if H is None:
        raise RuntimeError("OpenCV failed to compute homography")
    return H


def save_homography(H: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    data = {"homography": H.tolist()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def load_homography(path: str | Path) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "homography" not in data:
        raise ValueError("YAML missing 'homography'")
    H = np.array(data["homography"], dtype=np.float64)
    return H


def project_points(H: np.ndarray, points: Iterable[Point]) -> np.ndarray:
    pts = np.array([[x, y, 1.0] for x, y in points], dtype=np.float64).T
    projected = H @ pts
    projected /= projected[2:3]
    return projected[:2].T
