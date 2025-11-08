from __future__ import annotations

import contextlib
from typing import Generator, Iterable, Tuple

import cv2


Frame = Tuple[int, any]


@contextlib.contextmanager
def open_video(path: str) -> Iterable[cv2.VideoCapture]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    try:
        yield cap
    finally:
        cap.release()


def iter_frames(cap: cv2.VideoCapture) -> Generator[Tuple[int, any], None, None]:
    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        yield frame_idx, frame
        frame_idx += 1
