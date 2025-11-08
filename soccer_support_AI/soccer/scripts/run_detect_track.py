#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm

from soccer.core.detection import DetectorConfig, YoloDetector
from soccer.core.tracking import IOUTracker, TrackerConfig
from soccer.core.video_io import iter_frames, open_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection + tracking on a broadcast video")
    parser.add_argument("--input", required=True, help="Path to input video (mp4)")
    parser.add_argument("--out", required=True, help="Output JSON for track data")
    parser.add_argument("--detector", default="yolov8n.pt", help="Ultralytics YOLO weights")
    parser.add_argument("--confidence", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--device", default=None, help="Torch device (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Tracker IoU threshold")
    parser.add_argument("--max-age", type=int, default=30, help="Frames before a lost track is dropped")
    parser.add_argument("--min-hits", type=int, default=1, help="Frames required before track is emitted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = YoloDetector(
        DetectorConfig(
            model_name=args.detector,
            conf=args.confidence,
            device=args.device,
        )
    )
    tracker = IOUTracker(
        TrackerConfig(
            iou_threshold=args.iou_threshold,
            max_age=args.max_age,
            min_hits=args.min_hits,
        )
    )

    out_path = Path(args.out)
    records = []
    metadata = {}

    with open_video(args.input) as cap:
        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "frame_height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        for frame_idx, frame in tqdm(iter_frames(cap), total=metadata["frame_count"], desc="tracking"):
            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame_idx)
            for track in tracks:
                records.append(track.to_dict())

    payload = {
        "video": metadata,
        "tracks": records,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
