#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from soccer.core.homography import compute_homography, load_point_pairs, save_homography


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute homography from annotated point pairs")
    parser.add_argument("--point-csv", required=True, help="CSV with columns image_x,image_y,pitch_x,pitch_y")
    parser.add_argument("--out", required=True, help="YAML file to store 3x3 homography matrix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = load_point_pairs(args.point_csv)
    H = compute_homography(pairs)
    save_homography(H, args.out)
    print(f"Homography saved to {args.out}")


if __name__ == "__main__":
    main()
