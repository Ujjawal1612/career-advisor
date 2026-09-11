import pandas as pd

from career import FEATURES, load_data


def test_dataset_has_required_features():
    df = load_data()
    assert set(FEATURES).issubset(df.columns)
    assert "ROLE" in df.columns
    assert len(df) > 0


def test_features_are_numeric():
    df = load_data()
    assert all(pd.api.types.is_numeric_dtype(df[column]) for column in FEATURES)
