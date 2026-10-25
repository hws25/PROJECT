"""Tests for the baseline, metrics and timing helpers (using tiny toy data)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ev_energy import evaluate, fit_and_time, fixed_rate_predict


def _toy():
    """Four trips at exactly 0.2 kWh/km, split into train and test halves."""
    df = pd.DataFrame({"distance_km": [10, 20, 30, 40], "energy_kwh": [2.0, 4.0, 6.0, 8.0]})
    return df.iloc[:2], df.iloc[2:]


def test_fixed_rate_recovers_exact_rate():
    """On perfectly linear toy data the baseline recovers the exact rate."""
    train, test = _toy()
    assert np.allclose(fixed_rate_predict(train, test), [6.0, 8.0])


def test_evaluate_perfect_prediction():
    """Perfect predictions give zero error and R2 of 1."""
    m = evaluate([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2"] == 1


def test_fit_and_time_returns_timings():
    """fit_and_time returns correct predictions and non-negative timings."""
    train, test = _toy()
    pred, m = fit_and_time(LinearRegression(), train, test, ["distance_km"])
    assert np.allclose(pred, [6.0, 8.0])
    assert m["fit_s"] >= 0 and m["predict_s"] >= 0
