#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from soccer.core.homography import load_homography
from soccer.core.types import TrackRecord
from soccer.core.warp import project_track_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project image-space tracks onto pitch coordinates")
    parser.add_argument("--tracks", required=True, help="JSON produced by run_detect_track.py")
    parser.add_argument("--H", required=True, help="YAML homography file")
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    track_data = json.loads(Path(args.tracks).read_text(encoding="utf-8"))
    records = [
        TrackRecord(
            track_id=int(entry["track_id"]),
            frame_index=int(entry["frame_index"]),
            bbox=tuple(entry["bbox"]),
            score=float(entry["score"]),
        )
        for entry in track_data.get("tracks", [])
    ]
    H = load_homography(args.H)
    projected = project_track_records(records, H)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "track_id",
        "frame_index",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "score",
        "pitch_x",
        "pitch_y",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in projected:
            x1, y1, x2, y2 = item["bbox"]
            pitch_x, pitch_y = item["pitch"]
            writer.writerow(
                {
                    "track_id": item["track_id"],
                    "frame_index": item["frame_index"],
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "score": item["score"],
                    "pitch_x": pitch_x,
                    "pitch_y": pitch_y,
                }
            )
    print(f"Projected coordinates saved to {args.out}")


if __name__ == "__main__":
    main()
