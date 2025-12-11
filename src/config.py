"""Central paths and constants. Everything else imports from here."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_TRAINING_CSV = RAW_DIR / "cs-training.csv"
RAW_SCORING_CSV = RAW_DIR / "cs-test.csv"  # Kaggle holdout: unlabelled, demo input only

TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"

MODEL_PATH = MODELS_DIR / "model.joblib"
MODEL_META_PATH = MODELS_DIR / "model_meta.json"

TARGET = "SeriousDlqin2yrs"

# The 10 raw predictors, in the order the API expects them.
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# The three delinquency-count columns share a quirk: 96 and 98 are not counts,
# they are sentinel codes from the original data collection. See data.py.
PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
PAST_DUE_SENTINELS = (96, 98)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Fold-level parallelism. Keep at 1: the heavy estimators (the boosters, the
# random forest) already use every core internally, and nesting joblib around
# them oversubscribes the box badly -- 12 worker processes each spawning 12
# threads made a 1-minute fit take over 10.
CV_N_JOBS = 1

MLFLOW_EXPERIMENT = "credit-risk"
MLFLOW_TRACKING_URI = f"file://{ROOT / 'mlruns'}"
