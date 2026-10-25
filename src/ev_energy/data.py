"""Data loading, quality checks and cleaning for the EV trip dataset.

This module turns the raw Chalmers "Electric Vehicle Trip Energy Consumption"
CSV (one row per trip) into a tidy table that is ready for modelling.

The workflow is deliberately split into three steps so each can be inspected:

1. :func:`load_data` reads the file, renames columns and adds average speed.
2. :func:`quality_report` counts values outside physically plausible ranges
   (it only *reports*; nothing is deleted).
3. :func:`clean_data` removes the rows and columns judged unreliable.

The reasoning behind each range and each removal is in ``docs/decisions_log.md``.

Example:
    >>> from ev_energy import load_data, quality_report, clean_data
    >>> raw = load_data("data/raw/ev_trip_energy.csv")
    >>> quality_report(raw)          # inspect problems first
    >>> trips = clean_data(raw)      # then remove them
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Name of the column the models try to predict (trip energy in kWh).
TARGET = "energy_kwh"

#: Maps the long column names in the raw CSV to short snake_case names.
#: Units are added to the names so they are never ambiguous later.
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

#: ``{column: (low, high)}`` physically plausible bounds, inclusive.
#: Values outside a range are treated as data errors. Each bound is justified
#: in the trailing comment (the report's Appendix A repeats the reasoning).
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

#: Columns removed before modelling because their values are unreliable.
#: ``speed_raw`` contradicts its documentation; ``lat``/``lon`` hold impossible values.
DROP_COLUMNS = ["speed_raw", "lat", "lon"]

#: Columns whose out-of-range rows are removed. The three unreliable columns
#: above are excluded here: dropping the column is better than losing every row.
CHECK_COLUMNS = ["energy_kwh", "distance_km", "hour", "weekday", "voltage_v",
                 "temp_max_c", "temp_min_c", "duration_min"]


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw trip CSV and prepare column names and average speed.

    The raw ``Speed`` column is unreliable (see :data:`PLAUSIBLE_RANGES`), so
    a trustworthy speed is derived instead as distance divided by duration.

    Args:
        path: Location of the raw CSV file.

    Returns:
        One row per trip, with snake_case column names (see :data:`RENAME`)
        plus a new ``avg_speed_kmh`` column.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If ``distance_km`` or ``duration_min`` are missing after
            renaming, i.e. the file is not the expected dataset.

    Example:
        >>> df = load_data("data/raw/ev_trip_energy.csv")
        >>> df.shape[0]
        10151
    """
    df = pd.read_csv(path).rename(columns=RENAME)
    # Minutes -> hours so that distance (km) / time (h) gives km/h.
    df["avg_speed_kmh"] = df["distance_km"] / (df["duration_min"] / 60)
    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise data quality per column without changing the data.

    Use this *before* :func:`clean_data` to see what would be removed and why.

    Args:
        df: Table returned by :func:`load_data`.

    Returns:
        One row per column in :data:`PLAUSIBLE_RANGES`, indexed by column name,
        with the columns ``missing`` (count of NaN), ``outside_range`` (count
        outside the plausible range), ``range``, ``min`` and ``max``.

    Raises:
        KeyError: If a column listed in :data:`PLAUSIBLE_RANGES` is missing.
    """
    rows = []
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        # between() is inclusive and returns False for NaN, so NaNs count as bad.
        bad = ~df[col].between(lo, hi)
        rows.append(
            {"column": col, "missing": int(df[col].isna().sum()),
             "outside_range": int(bad.sum()), "range": f"[{lo}, {hi}]",
             "min": df[col].min(), "max": df[col].max()}
        )
    return pd.DataFrame(rows).set_index("column")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the trip table.

    Rules:
        1. Remove rows outside :data:`PLAUSIBLE_RANGES` for the columns in
           :data:`CHECK_COLUMNS` (missing values are also removed).
        2. Drop the columns in :data:`DROP_COLUMNS` entirely.
        3. Keep every other row: extreme but plausible trips are genuine data.

    The input table is not modified.

    Args:
        df: Table returned by :func:`load_data`.

    Returns:
        New DataFrame with a fresh 0..n-1 index, without the unreliable rows
        and columns. On the Chalmers file this removes one row (a 435 V
        voltage reading), leaving 10,150 trips.

    Raises:
        KeyError: If an expected column is missing from ``df``.

    Example:
        >>> trips = clean_data(load_data("data/raw/ev_trip_energy.csv"))
        >>> len(trips)
        10150
    """
    # Start by keeping everything, then switch off any row that fails a check.
    keep = pd.Series(True, index=df.index)
    for col in CHECK_COLUMNS:
        lo, hi = PLAUSIBLE_RANGES[col]
        keep &= df[col].between(lo, hi)   # NaN also fails, so missing values are removed
    # reset_index so downstream code can rely on a clean 0..n-1 index.
    return df.loc[keep].drop(columns=DROP_COLUMNS).reset_index(drop=True)
