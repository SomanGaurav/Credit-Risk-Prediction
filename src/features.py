"""Feature engineering and the fitted preprocessing pipeline.

Everything that learns from data (winsorising caps, imputation medians, scaling
statistics) lives inside a scikit-learn Pipeline so it is fit on training folds
only. The same object is serialised and used for serving, which is what keeps
training and inference from drifting apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config

# Columns whose extreme tails are data-entry artefacts rather than real signal.
# Utilisation and DebtRatio are meant to be ratios but reach 50,708 and 329,664.
WINSORIZE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
]
WINSORIZE_QUANTILE = 0.995


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add derived columns. Stateless -- pure row-wise arithmetic, so no leakage."""

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Delinquency history is the strongest signal in the data; aggregate it.
        X["TotalPastDue"] = X[config.PAST_DUE_COLS].sum(axis=1)
        X["HasAnyDelinquency"] = (X["TotalPastDue"] > 0).astype(int)
        # Severity-weighted: a 90-day late is worse than a 30-day late.
        X["WeightedDelinquency"] = (
            1 * X["NumberOfTime30-59DaysPastDueNotWorse"]
            + 2 * X["NumberOfTime60-89DaysPastDueNotWorse"]
            + 3 * X["NumberOfTimes90DaysLate"]
        )

        # DebtRatio is only a true ratio when income is known. Where income is
        # present, recover the absolute monthly debt burden and what is left over.
        income = X["MonthlyIncome"]
        X["MonthlyDebt"] = X["DebtRatio"] * income
        X["DisposableIncome"] = income - X["MonthlyDebt"]

        # Income has to stretch across the household.
        X["IncomePerDependent"] = income / (X["NumberOfDependents"].fillna(0) + 1)

        # What share of the credit lines are secured by property.
        X["RealEstateShare"] = X["NumberRealEstateLoansOrLines"] / X[
            "NumberOfOpenCreditLinesAndLoans"
        ].replace(0, np.nan)

        # Log-compress the heavy right tails so linear and distance-based models
        # can use them. log1p is safe: all these columns are non-negative.
        for col in ["MonthlyIncome", "DebtRatio", "RevolvingUtilizationOfUnsecuredLines"]:
            X[f"Log{col}"] = np.log1p(X[col].clip(lower=0))

        return X.replace([np.inf, -np.inf], np.nan)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip configured columns at a quantile learned from the training data."""

    def __init__(self, columns=None, quantile: float = WINSORIZE_QUANTILE):
        self.columns = columns
        self.quantile = quantile

    def fit(self, X: pd.DataFrame, y=None):
        cols = [c for c in (self.columns or []) if c in X.columns]
        self.caps_ = {c: X[c].quantile(self.quantile) for c in cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, cap in self.caps_.items():
            X[col] = X[col].clip(upper=cap)
        return X


def preprocessing_steps() -> list[tuple]:
    """Engineer -> winsorize -> median-impute -> scale.

    Returned as a flat step list rather than a nested Pipeline: imblearn rejects
    a Pipeline as an intermediate step, so the stages are spliced in directly.

    Scaling is required by LogisticRegression, SVM and MLP and is harmless for the
    tree ensembles, so one shared preprocessor keeps the comparison honest.
    """
    return [
        ("engineer", FeatureEngineer()),
        ("winsorize", Winsorizer(columns=WINSORIZE_COLS)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]


def build_preprocessor() -> Pipeline:
    return Pipeline(preprocessing_steps())


def build_pipeline(model, use_smote: bool = False):
    """Attach a classifier to the preprocessing stages.

    When use_smote is set, an imblearn Pipeline is used so resampling runs on the
    training fold only -- never on the validation fold and never at predict time.
    """
    steps = preprocessing_steps()
    if use_smote:
        steps.append(("smote", SMOTE(random_state=config.RANDOM_STATE)))
        return ImbPipeline(steps + [("model", model)])
    return Pipeline(steps + [("model", model)])


def feature_names() -> list[str]:
    """Column order produced by FeatureEngineer, for reporting importances."""
    sample = pd.DataFrame(
        [dict.fromkeys(config.FEATURES, 1.0)]
    ).assign(HasDelinquencySentinel=0, MonthlyIncomeMissing=0, NumberOfDependentsMissing=0)
    return list(FeatureEngineer().transform(sample).columns)
