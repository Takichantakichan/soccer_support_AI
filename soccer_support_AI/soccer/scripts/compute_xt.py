#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soccer.core.metrics import ExpectedThreatTable, compute_xt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute xT contributions from projected tracks")
    parser.add_argument("--xy", required=True, help="CSV with projected pitch coordinates")
    parser.add_argument("--xt-table", required=True, help="CSV containing x_bin,y_bin,value columns")
    parser.add_argument("--out", required=True, help="Output CSV for per-track xT scores")
    parser.add_argument("--pitch-length", type=float, default=105.0, help="Pitch length in meters")
    parser.add_argument("--pitch-width", type=float, default=68.0, help="Pitch width in meters")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = pd.read_csv(args.xy)
    required_cols = {"track_id", "frame_index", "pitch_x", "pitch_y"}
    if not required_cols.issubset(samples.columns):
        raise ValueError(f"Input CSV missing columns: {required_cols}")
    xt_table = ExpectedThreatTable(
        csv_path=args.xt_table,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
    )
    result = compute_xt(samples, xt_table)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"xT scores saved to {args.out}")


if __name__ == "__main__":
    main()
