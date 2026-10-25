"""Splitting, evaluation and analysis helpers for the EV energy models.

The functions here are model-agnostic: they work with any scikit-learn
regressor, which keeps the notebooks short and the logic testable.

Main pieces:

* :func:`split_by_vehicle` - train/test split that keeps each vehicle in one set.
* :func:`evaluate` and :func:`fit_and_time` - accuracy metrics and run time.
* :func:`fixed_rate_predict` - the simple kWh-per-km baseline.
* :func:`grouped_permutation_importance` - which inputs the model relies on.
* :func:`error_by_group` - where (e.g. which trip lengths) the model errs.

Example:
    >>> from sklearn.linear_model import LinearRegression
    >>> from ev_energy import (load_data, clean_data, split_by_vehicle,
    ...                        fit_and_time, TRIP_FEATURES)
    >>> train, test = split_by_vehicle(clean_data(load_data("data/raw/ev_trip_energy.csv")))
    >>> pred, metrics = fit_and_time(LinearRegression(), train, test, TRIP_FEATURES)
    >>> round(metrics["MAE"], 2)    # mean absolute error in kWh
    0.81
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

#: Inputs known (or estimable) before a trip starts.
TRIP_FEATURES = ["distance_km", "duration_min", "avg_speed_kmh",
                 "hour", "weekday", "temp_max_c", "temp_min_c"]

#: Inputs only measured *during* the trip; they make predictions look better
#: than a genuine pre-trip forecast would be, so they are tested separately.
ELECTRICAL_FEATURES = ["current_a", "voltage_v"]


def split_by_vehicle(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Split trips into train and test sets so no vehicle appears in both.

    Trips from one vehicle resemble each other. A random trip-level split would
    let the model see the test vehicles during training and overstate accuracy,
    so whole vehicles are held out instead.

    Args:
        df: Cleaned table containing a ``vehicle_id`` column.
        test_size: Approximate fraction of *vehicles* (not trips) held out.
        seed: Random seed, so the same split is produced every run.

    Returns:
        A ``(train_df, test_df)`` tuple of independent copies of the rows.

    Raises:
        KeyError: If ``df`` has no ``vehicle_id`` column.
        ValueError: If ``test_size`` leaves no vehicles in either set.

    Example:
        >>> train, test = split_by_vehicle(clean_trips)
        >>> set(train.vehicle_id).isdisjoint(test.vehicle_id)
        True
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    # groups= tells the splitter which rows belong together (same vehicle).
    train_idx, test_idx = next(splitter.split(df, groups=df["vehicle_id"]))
    # .copy() so later edits to the pieces never alter the original table.
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def evaluate(y_true, y_pred) -> dict:
    """Score predictions with MAE, RMSE and R-squared.

    Args:
        y_true: Actual values (array-like).
        y_pred: Predicted values, same length as ``y_true``.

    Returns:
        Dict with ``MAE`` (mean absolute error, kWh), ``RMSE`` (root mean
        squared error, kWh, which penalises large misses more) and ``R2``
        (share of variation explained; 1 is perfect).

    Raises:
        ValueError: If the inputs have different lengths or are empty.

    Example:
        >>> evaluate([1, 2, 3], [1, 2, 3])["MAE"]
        0.0
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        # sklearn returns squared error; take the root to get back to kWh.
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def fixed_rate_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict energy as distance multiplied by one average kWh-per-km rate.

    This is the simple baseline the machine-learning models need to beat.
    The rate comes from the training data only, so there is no test leakage.

    Args:
        train: Table with ``energy_kwh`` and ``distance_km`` used to find the rate.
        test: Table with ``distance_km`` to predict for.

    Returns:
        Array of predicted energy (kWh), one value per row of ``test``.

    Raises:
        KeyError: If a required column is missing.

    Example:
        >>> fixed_rate_predict(train, test)[:2]
        array([1.9, 4.2])
    """
    # Total energy / total distance weights long trips more than averaging
    # per-trip rates, which matches how the baseline would be used in practice.
    rate = train["energy_kwh"].sum() / train["distance_km"].sum()
    return (test["distance_km"] * rate).to_numpy()


def fit_and_time(model, train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    """Fit a model, predict the test set, and time both steps.

    Args:
        model: Any unfitted scikit-learn regressor. It is fitted in place.
        train: Training rows containing ``features`` and ``energy_kwh``.
        test: Test rows containing the same columns.
        features: Names of the input columns the model uses.

    Returns:
        A ``(predictions, metrics)`` tuple. ``predictions`` is an array of
        kWh values; ``metrics`` holds ``MAE``, ``RMSE``, ``R2``, ``fit_s``
        (seconds to train) and ``predict_s`` (seconds to predict the test set).

    Raises:
        KeyError: If a feature or ``energy_kwh`` is missing from the tables.

    Example:
        >>> pred, m = fit_and_time(LinearRegression(), train, test, TRIP_FEATURES)
        >>> m["fit_s"] >= 0
        True
    """
    # perf_counter is a high-resolution clock suited to timing short tasks.
    t0 = time.perf_counter()
    model.fit(train[features], train["energy_kwh"])
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(test[features])
    predict_s = time.perf_counter() - t0

    metrics = evaluate(test["energy_kwh"], pred)
    metrics.update({"fit_s": fit_s, "predict_s": predict_s})
    return pred, metrics


def grouped_permutation_importance(model, X: pd.DataFrame, y, groups: dict,
                                   n_repeats: int = 10, seed: int = 0) -> pd.DataFrame:
    """Measure how much each group of inputs matters to a fitted model.

    For each named group, the rows of its columns are shuffled (all columns in
    the group get the same new row order, so correlated inputs such as the two
    temperatures stay consistent with each other) and the rise in MAE is
    recorded. A bigger rise means the model relied on that group more.

    Args:
        model: A fitted regressor with a ``predict`` method.
        X: Test inputs.
        y: True target values for ``X``.
        groups: ``{group_name: [column names]}``, for example
            ``{"temperature": ["temp_max_c", "temp_min_c"]}``.
        n_repeats: Number of random shuffles per group (more = steadier).
        seed: Random seed for repeatability.

    Returns:
        DataFrame indexed by group name with ``mae_increase_mean`` and
        ``mae_increase_std`` (kWh), sorted with the most important group first.

    Raises:
        KeyError: If a listed column is not in ``X``.

    Example:
        >>> imp = grouped_permutation_importance(
        ...     model, X_test, y_test, {"distance": ["distance_km"]})
        >>> imp.index[0]
        'distance'
    """
    rng = np.random.default_rng(seed)
    base = mean_absolute_error(y, model.predict(X))   # error with nothing shuffled
    rows = []
    for name, cols in groups.items():
        increases = []
        for _ in range(n_repeats):
            Xp = X.copy()
            order = rng.permutation(len(X))
            # .to_numpy()[order] reorders rows by position and avoids pandas
            # re-aligning them by index, which would undo the shuffle.
            Xp[cols] = X[cols].to_numpy()[order]
            increases.append(mean_absolute_error(y, model.predict(Xp)) - base)
        rows.append({"group": name, "mae_increase_mean": np.mean(increases),
                     "mae_increase_std": np.std(increases)})
    return pd.DataFrame(rows).set_index("group").sort_values("mae_increase_mean", ascending=False)


def error_by_group(y_true, y_pred, groups) -> pd.DataFrame:
    """Summarise prediction errors within each category of a grouping variable.

    Group by an *input* (for example binned distance), not by the actual
    energy: grouping by the target makes the model look biased even when it
    is not (a selection effect).

    Args:
        y_true: Actual values.
        y_pred: Predicted values, same length as ``y_true``.
        groups: Series of the same length giving each row's category; its
            index is used to align the results.

    Returns:
        DataFrame indexed by category with ``trips`` (count), ``mean_residual``
        (actual - predicted; positive means under-prediction) and ``MAE``.

    Raises:
        ValueError: If the inputs have different lengths.

    Example:
        >>> error_by_group([5, 5], [4, 4], pd.Series(["long", "long"]))
              trips  mean_residual  MAE
        long      2            1.0  1.0
    """
    # Residual = actual - predicted, so a positive value means the model was too low.
    resid = pd.Series(np.asarray(y_true) - np.asarray(y_pred), index=groups.index)
    # observed=True ignores empty categories when grouping by a binned variable.
    out = pd.DataFrame({"trips": resid.groupby(groups, observed=True).size(),
                        "mean_residual": resid.groupby(groups, observed=True).mean(),
                        "MAE": resid.abs().groupby(groups, observed=True).mean()})
    return out
