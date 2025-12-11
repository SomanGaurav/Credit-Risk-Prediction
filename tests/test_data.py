"""Cleaning rules. These encode decisions that are easy to silently undo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data import MIN_AGE, clean, split


def test_sentinels_are_flagged_and_capped(raw_frame):
    out = clean(raw_frame)

    # Rows 0-3 carry 96/98 in every delinquency column.
    assert out.loc[0:3, "HasDelinquencySentinel"].eq(1).all()
    assert out.loc[4:, "HasDelinquencySentinel"].eq(0).all()

    # The sentinel values themselves must be gone.
    for col in config.PAST_DUE_COLS:
        assert not out[col].isin(config.PAST_DUE_SENTINELS).any()
        assert out[col].max() <= 20


def test_age_floor_applied(raw_frame):
    out = clean(raw_frame)
    assert out["age"].min() >= MIN_AGE
    assert out.loc[10, "age"] == MIN_AGE


def test_missingness_flags_match_nulls(raw_frame):
    out = clean(raw_frame)
    assert out["MonthlyIncomeMissing"].sum() == raw_frame["MonthlyIncome"].isna().sum()
    assert (out["NumberOfDependentsMissing"].sum()
            == raw_frame["NumberOfDependents"].isna().sum())
    # Flags record the nulls; clean() does not impute them away.
    assert out["MonthlyIncome"].isna().any()


def test_clean_does_not_mutate_input(raw_frame):
    before = raw_frame.copy()
    clean(raw_frame)
    pd.testing.assert_frame_equal(raw_frame, before)


def test_clean_works_without_target(raw_frame):
    """The scoring CSV has no label column; cleaning must not require one."""
    unlabelled = raw_frame.drop(columns=[config.TARGET])
    out = clean(unlabelled)
    assert config.TARGET not in out.columns
    assert "HasDelinquencySentinel" in out.columns


def test_split_is_stratified_and_disjoint(raw_frame):
    df = clean(raw_frame)
    train, test = split(df)

    assert len(train) + len(test) == len(df)
    assert len(test) == int(round(len(df) * config.TEST_SIZE))
    # Stratification should hold the positive rate steady across splits.
    assert np.isclose(train[config.TARGET].mean(), test[config.TARGET].mean(), atol=0.05)


def test_split_is_deterministic(raw_frame):
    df = clean(raw_frame)
    a, _ = split(df)
    b, _ = split(df)
    pd.testing.assert_frame_equal(a, b)
