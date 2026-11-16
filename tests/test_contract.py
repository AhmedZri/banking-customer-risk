import numpy as np
import pytest
from banking_risk.data import (
    synthetic_customers,
    validate_data,
    split_data,
    FEATURES,
    TARGET,
)


def test_no_id_or_target_features():
    assert len(FEATURES) == 23
    assert TARGET not in FEATURES and "ID" not in FEATURES


def test_disjoint_reproducible_stratified_split():
    df = synthetic_customers(500)
    a = split_data(df)
    b = split_data(df)
    assert [len(v) for v in a] == [300, 100, 100]
    for u, v in zip(a, b):
        assert u.index.equals(v.index)
    assert not set(a[0].index) & set(a[1].index)
    assert not set(a[0].index) & set(a[2].index)
    assert all(abs(v[TARGET].mean() - df[TARGET].mean()) < 0.03 for v in a)


@pytest.mark.parametrize(
    "kind", ["missing", "inf", "target", "age", "category", "duplicate_id"]
)
def test_bad_dataset_rejected(kind):
    df = synthetic_customers(100)
    if kind == "missing":
        df = df.drop(columns="LIMIT_BAL")
    if kind == "inf":
        df["LIMIT_BAL"] = df["LIMIT_BAL"].astype(float)
        df.loc[0, "LIMIT_BAL"] = np.inf
    if kind == "target":
        df.loc[0, TARGET] = 2
    if kind == "age":
        df.loc[0, "AGE"] = -2
    if kind == "category":
        df.loc[0, "SEX"] = 7
    if kind == "duplicate_id":
        df.loc[1, "ID"] = df.loc[0, "ID"]
    with pytest.raises(ValueError):
        validate_data(df)


def test_extreme_finite_value_rejected_before_estimator():
    df = synthetic_customers(100)
    df["BILL_AMT1"] = df["BILL_AMT1"].astype(float)
    df.loc[0, "BILL_AMT1"] = 1e308
    with pytest.raises(ValueError):
        validate_data(df)
