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
    "energy_kwh": (0, 23),       # The dataset explanatory file states that 22.4 kWh is how much the Rowe e50 pack holds, data max is 20.83
    "distance_km": (0, 170),     # Dataset file states reported range of 170km
    "hour": (0, 24),             # Time of day 24hr
    "weekday": (0, 6),           # 0-6 day of the week
    "speed_raw": (0, 150),       # raw speed column reaches 930, so we definitely need to clean some errors from the raw data
    "voltage_v": (250, 400),     # estimate of a Rowe e50 pack voltage lies between 290 and 320 V
    "temp_max_c": (-30, 60),     # generic safe limits for lithium cells
    "temp_min_c": (-30, 60),     
    "duration_min": (1, 600),    # 
    "lat": (30, 32),             # loose bounds surrounding shanghai
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
