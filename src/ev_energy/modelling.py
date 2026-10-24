"""Train/test splitting and evaluation helpers."""
from __future__ import annotations

import time

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


def fixed_rate_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predict energy as distance x (average kWh per km in the training data).

    This is the "simple fixed consumption rate" that the machine-learning
    models need to beat. The rate uses training data only (no test leakage).
    """
    rate = train["energy_kwh"].sum() / train["distance_km"].sum()
    return (test["distance_km"] * rate).to_numpy()


def fit_and_time(model, train: pd.DataFrame, test: pd.DataFrame, features: list[str]):
    """Fit a model, predict the test set, and time both steps.

    Args:
        model: any scikit-learn regressor.
        train, test: tables containing the features and the energy_kwh target.
        features: column names used as inputs.

    Returns:
        (predictions, metrics) where metrics holds MAE, RMSE, R2,
        fit_s (seconds to train) and predict_s (seconds to predict the test set).
    """
    t0 = time.perf_counter()
    model.fit(train[features], train["energy_kwh"])
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = model.predict(test[features])
    predict_s = time.perf_counter() - t0
    metrics = evaluate(test["energy_kwh"], pred)
    metrics.update({"fit_s": fit_s, "predict_s": predict_s})
    return pred, metrics


def grouped_permutation_importance(model, X: pd.DataFrame, y, groups: dict, n_repeats: int = 10,
                                   seed: int = 0) -> pd.DataFrame:
    """Permutation importance where related columns are shuffled together.

    For each named group of columns, the rows of those columns are shuffled
    (all columns of the group with the same row order, so their relationship to
    each other is kept) and the increase in MAE is recorded. A bigger increase
    means the model relied more on that group.

    Args:
        model: a fitted regressor.
        X: test features (DataFrame).
        y: true target values for X.
        groups: {group_name: [column names]}, e.g. {"temperature": ["temp_max_c", "temp_min_c"]}.
        n_repeats: number of random shuffles per group.
        seed: random seed.

    Returns:
        DataFrame with mean and std of the MAE increase (kWh), sorted largest first.
    """
    rng = np.random.default_rng(seed)
    base = mean_absolute_error(y, model.predict(X))
    rows = []
    for name, cols in groups.items():
        increases = []
        for _ in range(n_repeats):
            Xp = X.copy()
            order = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[order]
            increases.append(mean_absolute_error(y, model.predict(Xp)) - base)
        rows.append({"group": name, "mae_increase_mean": np.mean(increases),
                     "mae_increase_std": np.std(increases)})
    return pd.DataFrame(rows).set_index("group").sort_values("mae_increase_mean", ascending=False)


def error_by_group(y_true, y_pred, groups) -> pd.DataFrame:
    """Summarise errors within each category of `groups` (e.g. a binned variable).

    Args:
        y_true, y_pred: actual and predicted values.
        groups: a pandas Series of the same length giving each row's category.

    Returns:
        DataFrame with the number of trips, mean residual (actual - predicted;
        positive means under-prediction) and MAE for each category.
    """
    resid = pd.Series(np.asarray(y_true) - np.asarray(y_pred), index=groups.index)
    out = pd.DataFrame({"trips": resid.groupby(groups, observed=True).size(),
                        "mean_residual": resid.groupby(groups, observed=True).mean(),
                        "MAE": resid.abs().groupby(groups, observed=True).mean()})
    return out
