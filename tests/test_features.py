"""Feature engineering and the leakage properties of the fitted pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config
from src.data import clean
from src.features import FeatureEngineer, Winsorizer, build_pipeline, feature_names


def test_engineered_columns_are_added(raw_frame):
    out = FeatureEngineer().transform(clean(raw_frame))
    for col in ["TotalPastDue", "HasAnyDelinquency", "WeightedDelinquency",
                "MonthlyDebt", "DisposableIncome", "IncomePerDependent",
                "RealEstateShare", "LogMonthlyIncome"]:
        assert col in out.columns


def test_delinquency_aggregates_are_consistent():
    df = pd.DataFrame([{
        "NumberOfTime30-59DaysPastDueNotWorse": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 2,
        "NumberOfTimes90DaysLate": 3,
        "DebtRatio": 0.5, "MonthlyIncome": 1000.0, "NumberOfDependents": 1.0,
        "NumberRealEstateLoansOrLines": 2, "NumberOfOpenCreditLinesAndLoans": 4,
        "RevolvingUtilizationOfUnsecuredLines": 0.3, "age": 40,
    }])
    out = FeatureEngineer().transform(df)

    assert out.loc[0, "TotalPastDue"] == 6
    assert out.loc[0, "HasAnyDelinquency"] == 1
    assert out.loc[0, "WeightedDelinquency"] == 1 * 1 + 2 * 2 + 3 * 3
    assert out.loc[0, "MonthlyDebt"] == pytest.approx(500.0)
    assert out.loc[0, "DisposableIncome"] == pytest.approx(500.0)
    assert out.loc[0, "IncomePerDependent"] == pytest.approx(500.0)
    assert out.loc[0, "RealEstateShare"] == pytest.approx(0.5)


def test_no_infinities_survive():
    """Zero open credit lines would produce inf in RealEstateShare."""
    df = pd.DataFrame([{
        "NumberOfTime30-59DaysPastDueNotWorse": 0,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfTimes90DaysLate": 0,
        "DebtRatio": 0.5, "MonthlyIncome": 1000.0, "NumberOfDependents": 0.0,
        "NumberRealEstateLoansOrLines": 2, "NumberOfOpenCreditLinesAndLoans": 0,
        "RevolvingUtilizationOfUnsecuredLines": 0.3, "age": 40,
    }])
    out = FeatureEngineer().transform(df)
    assert not np.isinf(out.select_dtypes("number").to_numpy()).any()


def test_winsorizer_learns_caps_from_fit_data_only():
    train = pd.DataFrame({"x": list(range(100))})
    w = Winsorizer(columns=["x"], quantile=0.9).fit(train)
    cap = w.caps_["x"]

    # A far larger value at transform time is clipped to the *training* cap --
    # this is what stops test-set extremes from leaking into preprocessing.
    out = w.transform(pd.DataFrame({"x": [10_000]}))
    assert out.loc[0, "x"] == cap
    assert cap < 100


def test_winsorizer_ignores_absent_columns():
    w = Winsorizer(columns=["missing_col"], quantile=0.9).fit(pd.DataFrame({"x": [1, 2]}))
    assert w.caps_ == {}


def test_pipeline_imputes_missing_income(raw_frame):
    df = clean(raw_frame)
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    assert X["MonthlyIncome"].isna().any()

    pipe = build_pipeline(LogisticRegression(max_iter=500))
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]

    assert not np.isnan(proba).any()
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.parametrize("use_smote", [False, True])
def test_pipeline_runs_with_and_without_smote(raw_frame, use_smote):
    df = clean(raw_frame)
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]

    pipe = build_pipeline(LogisticRegression(max_iter=500), use_smote=use_smote)
    pipe.fit(X, y)

    # SMOTE must not change how many rows come back at predict time.
    assert len(pipe.predict(X)) == len(X)


def test_feature_names_match_transform_output(raw_frame):
    """feature_names() is used to report importances; it must not drift."""
    df = clean(raw_frame).drop(columns=[config.TARGET])
    out = FeatureEngineer().transform(df)
    assert feature_names() == list(out.columns)
