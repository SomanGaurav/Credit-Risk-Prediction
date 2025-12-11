# Credit Risk Prediction

Predicts the probability that a loan applicant becomes seriously delinquent
(90+ days past due) within two years, trained on the Kaggle
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset
(150k applicants, ~6.7% default rate). Covers the full lifecycle: cleaning,
EDA, feature engineering, model comparison, hyperparameter tuning, a FastAPI
service, and the surrounding MLOps (MLflow, DVC, Docker, CI).

## Current model

The shipped model is a **class-weighted LightGBM baseline** — the full model
zoo (Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost,
LightGBM, CatBoost, MLP) and hyperparameter search (Grid, Random, Optuna) are
implemented and ready to run, but haven't been executed to completion yet.
See [Training the full zoo](#training-the-full-zoo--tuning) to run them and
promote a challenger.

Held-out test performance (30,000 rows, threshold chosen from out-of-fold
training predictions to maximize F1 — not fit on this test set):

| metric | value |
|---|---|
| ROC-AUC | 0.865 |
| PR-AUC | 0.405 |
| F1 | 0.446 |
| Precision | 0.405 |
| Recall | 0.497 |
| Accuracy | 0.918 |

At a 6.7% base rate, accuracy is a misleading headline number (predicting
"never defaults" scores 93%) — PR-AUC and recall are the metrics that matter
here. Curves and confusion matrix: [reports/figures/](reports/figures/).

## Project layout

```
src/            data cleaning, features, model zoo, training, tuning, evaluation
api/            FastAPI service (schemas + app)
tests/          pytest suite (unit + API contract + model artifact tests)
data/raw/       source CSVs, tracked with DVC
data/processed/ train/test parquet, produced by src/data.py
models/         serialized pipeline + metadata (committed — see below)
reports/        metrics, EDA and evaluation figures
dvc.yaml        prepare -> build_model -> evaluate pipeline
```

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

## Reproducing the pipeline

```bash
dvc repro
```

Runs `src/data.py` (clean + stratified split) -> `src/build_model.py` (fit
baseline, pick decision threshold from out-of-fold predictions) ->
`src/evaluate.py` (held-out metrics and plots). Each stage also runs
standalone as `python -m src.<name>`.

EDA (writes `reports/figures/`, not part of the DVC pipeline):

```bash
python -m src.eda
```

## Training the full zoo & tuning

Not run by default — CPU cost adds up across 8 models x 5 folds. To compare
the full zoo and log every run to MLflow:

```bash
python -m src.train                    # class-weighted models
python -m src.train --smote            # same zoo, SMOTE instead
python -m src.train --only xgboost lightgbm   # subset

mlflow ui --backend-store-uri file:./mlruns   # inspect runs at localhost:5000
```

Hyperparameter search (grid on a decision tree, random on a random forest,
Optuna/TPE on LightGBM — see `src/tune.py` for why each method is paired with
that model):

```bash
python -m src.tune --method grid
python -m src.tune --method random --trials 25
python -m src.tune --method optuna --trials 40
```

To promote a challenger to the served model:

```bash
python -m src.build_model --params reports/best_params_optuna.json
python -m src.evaluate
```

Then commit the updated `models/model.joblib` and `models/model_meta.json`.

**Stopping a run:** these are plain foreground Python processes — `Ctrl+C`,
or from another shell:

```bash
pkill -f "src.train"
pkill -f "src.tune"
```

## API

```bash
uvicorn api.main:app --reload
```

Docs at `http://localhost:8000/docs`.

| endpoint | purpose |
|---|---|
| `GET /health` | liveness; `200` even if the model failed to load (`status: degraded`) |
| `GET /model-info` | model metadata, feature list, test-set metrics |
| `POST /predict` | score one applicant |
| `POST /batch_predict` | score up to 10,000 applicants |

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "RevolvingUtilizationOfUnsecuredLines": 0.766, "age": 45,
  "NumberOfTime30-59DaysPastDueNotWorse": 2, "DebtRatio": 0.803,
  "MonthlyIncome": 9120, "NumberOfOpenCreditLinesAndLoans": 13,
  "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 6,
  "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 2
}'
# {"probability":0.832,"prediction":1,"risk_band":"high","threshold":0.774}
```

`MonthlyIncome` and `NumberOfDependents` accept `null` — ~20%/3% of the
training data had them missing, and the pipeline imputes internally.

## Tests

```bash
pytest tests/ -v
```

Data-cleaning, feature-pipeline and API tests run against synthetic fixtures
(no model needed). `tests/test_model_artifact.py` exercises the real
serialized pipeline and auto-skips if `models/model.joblib` is absent.

## Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api
# or: docker compose up
```

The image installs `requirements-api.txt` only (serving deps — no MLflow,
Optuna, DVC or matplotlib) and copies the committed `models/` artifact, so it
builds from a fresh clone with no training step or DVC remote required.

## MLOps

- **Experiment tracking** — MLflow, local file store at `mlruns/` (`mlflow ui`)
- **Data versioning** — DVC tracks `data/raw/*.csv`; no remote configured, add
  one with `dvc remote add -d <name> <url>` for team sharing
- **Pipeline** — `dvc.yaml` / `dvc.lock` define and cache prepare -> train -> evaluate
- **Model artifact** — committed to git rather than a registry, since there's
  no DVC/MLflow remote wired up here; `mlflow.sklearn.log_model(...,
  registered_model_name="credit-risk")` in `src/build_model.py` registers to
  the local MLflow store when `--no-register` is omitted
- **CI** — `.github/workflows/ci.yml`: lint (ruff) + pytest, then a Docker build
- **Cloud deploy** — not deployed; the Dockerfile/compose file are deploy-ready
  for Render/Railway/Fly (build from `Dockerfile`, expose port 8000, healthcheck
  at `/health`)

## Data notes

Cleaning decisions (`src/data.py`), each checked against the actual data
before applying:

- `NumberOfTime{30-59,60-89,90}DaysLate` use **96/98 as sentinel codes**, not
  counts (269 rows, always across all three columns together). Default rate
  in those rows is 54.7% vs. a 6.7% base rate, so they're flagged
  (`HasDelinquencySentinel`) rather than dropped, and capped to the max
  legitimate value per column.
- One row has `age == 0`; floored to 18.
- `MonthlyIncome` (19.8% missing) and `NumberOfDependents` (2.6% missing) are
  **not missing at random** — flagged before imputation (median, inside the
  fitted pipeline, train-fold only).
- `DebtRatio` exceeds 1 mostly where income is missing (a known quirk: the
  raw debt figure sits in a ratio field). Left as-is; `Winsorizer` in
  `src/features.py` caps the extreme tail at the 99.5th percentile learned
  from training data.

See `reports/figures/` for distributions, default rate by decile, and the
Spearman correlation matrix.
