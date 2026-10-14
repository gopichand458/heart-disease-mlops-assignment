import numpy as np
import pandas as pd

from src.features import (CATEGORICAL, ENGINEERED, NUMERIC, add_features,
                          build_preprocessor, load_xy)


def _sample() -> pd.DataFrame:
    return pd.DataFrame([{
        "age": 60.0, "sex": 1.0, "cp": 4.0, "trestbps": 140.0, "chol": 250.0,
        "fbs": 0.0, "restecg": 1.0, "thalach": 120.0, "exang": 1.0,
        "oldpeak": 2.0, "slope": 2.0, "ca": np.nan, "thal": 7.0,
    }])


def test_add_features_hr_reserve():
    out = add_features(_sample())
    assert out.loc[0, "hr_reserve"] == (220 - 60) - 120


def test_load_xy_shapes_and_binary_target():
    X, y = load_xy()
    assert len(X) == len(y) > 250
    assert set(y.unique()) == {0, 1}
    assert "target" not in X.columns


def test_preprocessor_handles_missing_values():
    X, _ = load_xy()
    prep = build_preprocessor().fit(X)
    out = prep.transform(add_features(_sample()))
    assert not np.isnan(out).any()


def test_feature_lists_are_disjoint():
    assert not set(NUMERIC + ENGINEERED) & set(CATEGORICAL)
