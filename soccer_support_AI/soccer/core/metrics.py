from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExpectedThreatTable:
    csv_path: str
    pitch_length: float = 105.0
    pitch_width: float = 68.0

    def __post_init__(self):
        df = pd.read_csv(self.csv_path)
        required = {"x_bin", "y_bin", "value"}
        if not required.issubset(df.columns):
            raise ValueError("xT table must have columns x_bin, y_bin, value")
        self.nx = int(df["x_bin"].max()) + 1
        self.ny = int(df["y_bin"].max()) + 1
        grid = np.full((self.ny, self.nx), 0.0, dtype=np.float64)
        for row in df.itertuples():
            grid[int(row.y_bin), int(row.x_bin)] = float(row.value)
        self.grid = grid

    def value_at(self, px: float, py: float) -> float:
        nx, ny = self.nx, self.ny
        x_norm = np.clip(px / self.pitch_length, 0.0, 0.999)
        y_norm = np.clip(py / self.pitch_width, 0.0, 0.999)
        x_idx = int(x_norm * nx)
        y_idx = int(y_norm * ny)
        return float(self.grid[y_idx, x_idx])


def compute_xt(samples: pd.DataFrame, xt_table: ExpectedThreatTable) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(columns=["track_id", "xt"])
    samples = samples.copy()
    samples.sort_values(["track_id", "frame_index"], inplace=True)
    samples["xt_value"] = samples.apply(
        lambda row: xt_table.value_at(row.pitch_x, row.pitch_y), axis=1
    )
    samples["xt_delta"] = samples.groupby("track_id")["xt_value"].diff().fillna(0.0)
    agg = (
        samples.groupby("track_id")["xt_delta"].sum().reset_index().rename(columns={"xt_delta": "xt"})
    )
    return agg
