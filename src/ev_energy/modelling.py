"""Train/test splitting and evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

TRIP_FEATURES = ["distance_km", "duration_min", "avg_speed_kmh",
                 "hour", "weekday", "temp_max_c", "temp_min_c"]
ELECTRICAL_FEATURES = ["current_a", "voltage_v"]


def split_by_vehicle(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split into train/test so that no vehicle appears in both sets.

    Args:
        df: cleaned table containing a vehicle_id column.
        test_size: approximate fraction of vehicles (not trips) held out.
        seed: random seed for repeatability.

    Returns:
        (train_df, test_df)
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["vehicle_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def evaluate(y_true, y_pred) -> dict:
    """Return MAE, RMSE (both in kWh) and R^2 for a set of predictions."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }
