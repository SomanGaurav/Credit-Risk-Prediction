"""Shared fixtures.

Tests build their own tiny model rather than depending on models/model.joblib,
so the suite runs green on a fresh clone and in CI where no artefact exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture
def raw_row() -> dict:
    """One applicant in the raw dataset's column names."""
    return {
        "RevolvingUtilizationOfUnsecuredLines": 0.766,
        "age": 45,
        "NumberOfTime30-59DaysPastDueNotWorse": 2,
        "DebtRatio": 0.803,
        "MonthlyIncome": 9120.0,
        "NumberOfOpenCreditLinesAndLoans": 13,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 6,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2.0,
    }


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A synthetic frame carrying every quirk the cleaner has to handle:
    sentinel codes, a zero age, missing income and missing dependents."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "SeriousDlqin2yrs": rng.integers(0, 2, n),
        "RevolvingUtilizationOfUnsecuredLines": rng.random(n) * 2,
        "age": rng.integers(21, 80, n),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 3, n),
        "DebtRatio": rng.random(n) * 3,
        "MonthlyIncome": rng.random(n) * 10000,
        "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n),
        "NumberOfTimes90DaysLate": rng.integers(0, 3, n),
        "NumberRealEstateLoansOrLines": rng.integers(0, 5, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 3, n),
        "NumberOfDependents": rng.integers(0, 4, n).astype(float),
    })
    # Sentinels appear together across all three delinquency columns.
    for col in config.PAST_DUE_COLS:
        df.loc[0:2, col] = 98
        df.loc[3, col] = 96
    df.loc[10, "age"] = 0
    df.loc[11:20, "MonthlyIncome"] = np.nan
    df.loc[15:18, "NumberOfDependents"] = np.nan
    return df
