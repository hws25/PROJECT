"""Data loading and quality checks.

Inputs : path to the Chalmers EV trip CSV (13 columns, one row per trip).
Outputs: a pandas DataFrame with tidy snake_case column names.
Invalid values are FLAGGED, not silently deleted (see docs/decisions_log.md).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "energy_kwh"

RENAME = {
    "Trip Energy Consumption": "energy_kwh",
    "Vehicle ID": "vehicle_id",
    "Trip Distance": "distance_km",
    "Time of Day": "hour",
    "Day of the Week": "weekday",
    "Longitude": "lon",
    "Latitude": "lat",
    "Speed": "speed_raw",
    "Current": "current_a",
    "Total Voltage": "voltage_v",
    "Maximum Cell Temperature of Battery": "temp_max_c",
    "Minimum Cell Temperature of Battery": "temp_min_c",
    "Trip Time Length": "duration_min",
}

# (low, high) physically plausible bounds used to FLAG rows. Justify each in the report.
PLAUSIBLE_RANGES = {
    "energy_kwh": (0, 22.4),     # cannot exceed the 22.4 kWh pack (dataset explanatory file)
    "distance_km": (0, 170),     # reported driving range of 170 km
    "hour": (0, 24),             # departure hour, 24-hour clock
    "weekday": (0, 6),           # day of week coded 0-6
    "speed_raw": (0, 150),       # documented as km/h but reaches 930, so >150 treated as error
    "voltage_v": (250, 400),     # loose: ~99.8% of values lie in 290-317 V
    "temp_max_c": (-30, 60),     # generic lithium-ion limits; data spans about -3 to 34 C
    "temp_min_c": (-30, 60),
    "duration_min": (1, 600),    # 1 min to 10 h; data max is 393 min
    "lat": (30, 32),             # loose bounds around Shanghai
    "lon": (120, 123),
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the CSV, rename columns, add derived avg_speed_kmh = distance / duration."""
    df = pd.read_csv(path).rename(columns=RENAME)
    df["avg_speed_kmh"] = df["distance_km"] / (df["duration_min"] / 60)
    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column count of values outside PLAUSIBLE_RANGES, plus missing counts."""
    rows = []
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        bad = ~df[col].between(lo, hi)
        rows.append(
            {"column": col, "missing": int(df[col].isna().sum()),
             "outside_range": int(bad.sum()), "range": f"[{lo}, {hi}]",
             "min": df[col].min(), "max": df[col].max()}
        )
    return pd.DataFrame(rows).set_index("column")


# Columns removed before modelling (see docs/decisions_log.md)
DROP_COLUMNS = ["speed_raw", "lat", "lon"]
# Columns whose out-of-range rows are removed (unreliable raw columns are dropped instead)
CHECK_COLUMNS = ["energy_kwh", "distance_km", "hour", "weekday", "voltage_v",
                 "temp_max_c", "temp_min_c", "duration_min"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the trip table.

    Rules:
      1. Remove rows outside PLAUSIBLE_RANGES for the columns in CHECK_COLUMNS.
      2. Drop speed_raw (contradicts its documentation), lat and lon (impossible values).
      3. Keep every other row; extreme but plausible trips are genuine.

    Args:
        df: table from load_data().

    Returns:
        New DataFrame with the unreliable rows and columns removed.

    Raises:
        KeyError: if an expected column is missing.
    """
    keep = pd.Series(True, index=df.index)
    for col in CHECK_COLUMNS:
        lo, hi = PLAUSIBLE_RANGES[col]
        keep &= df[col].between(lo, hi)   # NaN also fails, so missing values are removed
    return df.loc[keep].drop(columns=DROP_COLUMNS).reset_index(drop=True)
