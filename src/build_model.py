"""Fit the champion pipeline on the training split and serialise it for serving.

    python -m src.build_model                                   # baseline LightGBM
    python -m src.build_model --params reports/best_params_optuna.json

The decision threshold is chosen here, from out-of-fold predictions on the
*training* split. That matters: picking it on the test split would quietly tune
the model to the data used to report its performance.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src import config, features
from src.models import get_models
from src.train import load_train

BASELINE_MODEL = "lightgbm"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def select_threshold(y_true, y_proba) -> tuple[float, float]:
    """Pick the probability cut-off that maximises F1.

    0.5 is a poor default at a 6.7% base rate -- it optimises accuracy, which a
    model can score 93% on by predicting 'never defaults'. F1 balances catching
    defaulters against burying the credit team in false positives.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one more point than thresholds.
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.nanargmax(f1))
    return float(thresholds[best]), float(f1[best])


def build_model(params: dict | None = None):
    """Baseline is the class-weighted LightGBM from the zoo; --params overrides it."""
    if params:
        clean = {k.replace("model__", ""): v for k, v in params.items()}
        return LGBMClassifier(verbose=-1, n_jobs=-1,
                              random_state=config.RANDOM_STATE, **clean)
    return get_models(balanced=True)[BASELINE_MODEL]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", help="JSON file written by src.tune")
    parser.add_argument("--no-register", action="store_true",
                        help="skip the MLflow model registry step")
    args = parser.parse_args()

    params, source = None, "baseline"
    if args.params:
        payload = json.loads((config.ROOT / args.params).read_text())
        params, source = payload["params"], f"tuned:{payload['method']}"

    X, y = load_train()
    model = build_model(params)
    pipe = features.build_pipeline(model)

    print(f"Fitting {BASELINE_MODEL} ({source}) on {X.shape}")

    # Out-of-fold probabilities: every prediction comes from a model that did not
    # see that row, so the threshold is not fit to its own training data.
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba",
                            n_jobs=config.CV_N_JOBS)[:, 1]
    threshold, oof_f1 = select_threshold(y, oof)
    print(f"  out-of-fold threshold {threshold:.4f} (F1 {oof_f1:.4f}); "
          f"F1 at 0.5 would be {f1_score(y, oof >= 0.5):.4f}")

    pipe.fit(X, y)
    print("  fitted on full training split")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)

    meta = {
        "model_name": BASELINE_MODEL,
        "source": source,
        "model_class": type(model).__name__,
        "model_params": {k: v for k, v in model.get_params().items()
                         if isinstance(v, (str, int, float, bool, type(None)))},
        "decision_threshold": threshold,
        "oof_f1": oof_f1,
        "raw_features": config.FEATURES,
        "engineered_features": features.feature_names(),
        "n_training_rows": int(len(X)),
        "training_positive_rate": float(y.mean()),
        "sklearn_pipeline_steps": [name for name, _ in pipe.steps],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
    }
    config.MODEL_META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"  wrote {config.MODEL_PATH.relative_to(config.ROOT)} "
          f"({config.MODEL_PATH.stat().st_size / 1e6:.1f} MB)")
    print(f"  wrote {config.MODEL_META_PATH.relative_to(config.ROOT)}")

    if not args.no_register:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=f"final-{BASELINE_MODEL}-{source}"):
            mlflow.set_tags({"stage": "final_model", "model": BASELINE_MODEL,
                             "source": source})
            mlflow.log_params(meta["model_params"])
            mlflow.log_metrics({"oof_f1": oof_f1, "decision_threshold": threshold})
            mlflow.log_dict(meta, "model_meta.json")
            mlflow.sklearn.log_model(pipe, name="model",
                                     registered_model_name="credit-risk")
        print("  registered in MLflow as 'credit-risk'")


if __name__ == "__main__":
    main()
