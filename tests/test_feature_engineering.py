import numpy as np
import pandas as pd

from Tools.feature_enginner import FeatureEngineering


def _frame(glucose_values):
    size = len(glucose_values)
    return pd.DataFrame(
        {
            "sysBP": np.linspace(110, 140, size),
            "diaBP": np.linspace(70, 90, size),
            "totChol": np.linspace(180, 240, size),
            "glucose": glucose_values,
            "cigsPerDay": np.zeros(size),
            "BMI": np.linspace(22, 31, size),
        }
    )


def test_held_out_row_uses_training_statistics():
    training = _frame([80.0, 100.0, 120.0, 140.0])
    held_out = _frame([1_000.0])

    transformer = FeatureEngineering(training).fit()
    transformed = transformer.transform_new(held_out)

    assert transformer.glucose_mean == 110.0
    assert transformed.loc[0, "glucose"] < 1_000.0
    assert np.isfinite(transformed.loc[0, "glucose_norm"])
    assert np.isfinite(transformed.loc[0, "difference_bp_ratio"])


def test_transform_preserves_row_count_and_adds_expected_features():
    training = _frame([80.0, 100.0, 120.0, 140.0])

    transformed = FeatureEngineering(training).transform()

    assert len(transformed) == len(training)
    assert {
        "pulse_pressure",
        "difference_bp_ratio",
        "obese_flag",
        "glucose_norm",
    }.issubset(transformed.columns)
