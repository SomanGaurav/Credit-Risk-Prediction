"""Load the raw Kaggle CSV and apply structural cleaning.

Only *deterministic, row-level* fixes live here — things that are true of a single
record regardless of any other record. Anything that has to be learned from the
data (imputation values, winsorising cut-offs, scaling) belongs in the fitted
pipeline in features.py, so it can be fit on train folds only and never leak.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

# Largest value in each delinquency column that is a genuine count rather than a
# sentinel code. Measured on the raw training set; used to cap 96/98 back to a
# plausible magnitude while keep_sentinel_flag preserves the fact that they were special.
_MAX_LEGIT_PAST_DUE = {
    "NumberOfTime30-59DaysPastDueNotWorse": 13,
    "NumberOfTimes90DaysLate": 17,
    "NumberOfTime60-89DaysPastDueNotWorse": 11,
}

MIN_AGE = 18


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV. The first column is an unnamed row index."""
    return pd.read_csv(path or config.RAW_TRAINING_CSV, index_col=0)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply structural fixes. Safe to call on unlabelled scoring data."""
    df = df.copy()

    # 96 and 98 are sentinel codes, not counts. They co-occur across all three
    # delinquency columns in the same records, and those records default at ~55%
    # versus a ~7% base rate -- so the sentinel itself is signal worth keeping.
    sentinel_mask = df[config.PAST_DUE_COLS[0]].isin(config.PAST_DUE_SENTINELS)
    df["HasDelinquencySentinel"] = sentinel_mask.astype(int)
    for col in config.PAST_DUE_COLS:
        df[col] = df[col].mask(
            df[col].isin(config.PAST_DUE_SENTINELS), _MAX_LEGIT_PAST_DUE[col]
        )

    # A single record has age 0; ages below 18 cannot hold credit.
    df["age"] = df["age"].clip(lower=MIN_AGE)

    # Missingness here is informative, not random: income is absent for ~20% of
    # records and those records behave differently. Flag before imputing.
    df["MonthlyIncomeMissing"] = df["MonthlyIncome"].isna().astype(int)
    df["NumberOfDependentsMissing"] = df["NumberOfDependents"].isna().astype(int)

    return df


def split(df: pd.DataFrame):
    """Stratified train/test split -- stratification matters at a 6.7% positive rate."""
    train, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=df[config.TARGET],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = clean(load_raw())
    train, test = split(df)

    train.to_parquet(config.TRAIN_PARQUET, index=False)
    test.to_parquet(config.TEST_PARQUET, index=False)

    print(f"train {train.shape} -> {config.TRAIN_PARQUET.name}")
    print(f"test  {test.shape} -> {config.TEST_PARQUET.name}")
    print(f"positive rate: train {train[config.TARGET].mean():.4f} "
          f"test {test[config.TARGET].mean():.4f}")


if __name__ == "__main__":
    main()
