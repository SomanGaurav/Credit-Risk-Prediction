"""Cross-validate the model zoo and log every run to MLflow.

    python -m src.train              # class-weighted models
    python -m src.train --smote      # same zoo, SMOTE instead of class weights

Selection metric is PR-AUC (average precision), not ROC-AUC: at a 6.7% positive
rate ROC-AUC is dominated by the easy negatives, while PR-AUC tracks the thing a
lender actually cares about -- precision among the applicants we flag.
"""

from __future__ import annotations

import argparse
import json
import warnings

import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import config, features
from src.models import get_models

# sklearn 1.5.2 passes an `iprint` option that newer scipy no longer accepts.
# Harmless, but it fires once per fit and drowns the log.
warnings.filterwarnings("ignore", message="Unknown solver options")

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}
SELECTION_METRIC = "pr_auc"


def load_train():
    df = pd.read_parquet(config.TRAIN_PARQUET)
    return df.drop(columns=[config.TARGET]), df[config.TARGET]


def evaluate_model(name: str, model, X, y, use_smote: bool) -> dict:
    pipe = features.build_pipeline(model, use_smote=use_smote)
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=config.CV_N_JOBS)

    result = {"model": name, "smote": use_smote}
    for metric in SCORING:
        vals = scores[f"test_{metric}"]
        result[metric] = float(vals.mean())
        result[f"{metric}_std"] = float(vals.std())
    result["fit_seconds"] = float(scores["fit_time"].mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smote", action="store_true",
                        help="resample with SMOTE instead of class weighting")
    parser.add_argument("--only", nargs="*", help="run a subset of the zoo by name")
    args = parser.parse_args()

    X, y = load_train()
    models = get_models(balanced=not args.smote)
    if args.only:
        models = {k: v for k, v in models.items() if k in args.only}

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    strategy = "smote" if args.smote else "class_weight"
    print(f"Cross-validating {len(models)} models on {X.shape} "
          f"(imbalance strategy: {strategy})\n")

    results = []
    for name, model in models.items():
        print(f"  {name:22s} ...", end="", flush=True)
        with mlflow.start_run(run_name=f"cv-{name}-{strategy}"):
            res = evaluate_model(name, model, X, y, use_smote=args.smote)

            mlflow.set_tags({"stage": "model_comparison", "model": name,
                             "imbalance_strategy": strategy})
            mlflow.log_params({"model": name, "imbalance_strategy": strategy,
                               "cv_folds": config.CV_FOLDS,
                               **{f"model__{k}": v for k, v in
                                  model.get_params().items() if _loggable(v)}})
            mlflow.log_metrics({k: v for k, v in res.items()
                                if isinstance(v, float)})
            results.append(res)
        print(f" pr_auc {res['pr_auc']:.4f}  roc_auc {res['roc_auc']:.4f}  "
              f"({res['fit_seconds']:.0f}s/fold)")

    table = (pd.DataFrame(results)
             .sort_values(SELECTION_METRIC, ascending=False)
             .reset_index(drop=True))

    print(f"\n--- ranked by {SELECTION_METRIC} ---")
    cols = ["model", "pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / f"model_comparison_{strategy}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out.relative_to(config.ROOT)}")
    print(f"best: {table.loc[0, 'model']} ({SELECTION_METRIC}={table.loc[0, SELECTION_METRIC]:.4f})")


def _loggable(value) -> bool:
    return isinstance(value, (str, int, float, bool, type(None)))


if __name__ == "__main__":
    main()
