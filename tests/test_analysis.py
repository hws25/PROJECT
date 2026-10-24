import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ev_energy import error_by_group, grouped_permutation_importance


def test_permutation_importance_ranks_the_useful_feature_first():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"useful": rng.normal(size=300), "noise": rng.normal(size=300)})
    y = 3 * X["useful"]
    model = LinearRegression().fit(X, y)
    imp = grouped_permutation_importance(model, X, y, {"useful": ["useful"], "noise": ["noise"]})
    assert imp.index[0] == "useful"
    assert imp.loc["noise", "mae_increase_mean"] < 1e-9


def test_error_by_group_sign_and_counts():
    groups = pd.Series(["low", "low", "high", "high"])
    out = error_by_group([1, 1, 5, 5], [1, 1, 4, 4], groups)
    assert out.loc["high", "mean_residual"] == 1      # actual > predicted -> positive
    assert out.loc["low", "MAE"] == 0
    assert out["trips"].sum() == 4
