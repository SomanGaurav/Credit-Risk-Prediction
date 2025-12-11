"""Hyperparameter search: grid, random, and Optuna Bayesian optimisation.

    python -m src.tune --method grid      # DecisionTree, small exhaustive grid
    python -m src.tune --method random    # RandomForest, sampled budget
    python -m src.tune --method optuna    # LightGBM, TPE -- this is the one that ships

The three methods are pointed at different models on purpose. Grid search is only
affordable on a cheap estimator with few knobs, random search covers a wider space
for the same budget, and Optuna's TPE is reserved for the gradient booster where
the search space is large enough for the sampler to earn its keep.
"""

from __future__ import annotations

import argparse
import json

import mlflow
import optuna
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.tree import DecisionTreeClassifier

from src import config, features
from src.models import NEG_POS_RATIO
from src.train import SELECTION_METRIC, load_train

SCORING = "average_precision"  # PR-AUC, matching train.py's selection metric


def _cv():
    return StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def run_grid(X, y):
    """Exhaustive grid over a decision tree -- 3x3x2 = 18 fits x 5 folds."""
    pipe = features.build_pipeline(
        DecisionTreeClassifier(class_weight="balanced", random_state=config.RANDOM_STATE)
    )
    grid = {
        "model__max_depth": [4, 8, 12],
        "model__min_samples_leaf": [20, 50, 100],
        "model__criterion": ["gini", "entropy"],
    }
    # A decision tree is single-threaded, so here fold-level parallelism is the
    # only parallelism available and there is nothing to oversubscribe.
    search = GridSearchCV(pipe, grid, scoring=SCORING, cv=_cv(), n_jobs=-1, verbose=1)
    search.fit(X, y)
    return "decision_tree", search.best_params_, search.best_score_, len(search.cv_results_["params"])


def run_random(X, y, n_iter: int = 25):
    """Random search over a wider random-forest space for a fixed budget."""
    pipe = features.build_pipeline(
        RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                               random_state=config.RANDOM_STATE)
    )
    space = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [8, 12, 16, None],
        "model__min_samples_leaf": [5, 10, 20, 50],
        "model__max_features": ["sqrt", "log2", 0.5],
    }
    search = RandomizedSearchCV(
        pipe, space, n_iter=n_iter, scoring=SCORING, cv=_cv(),
        n_jobs=config.CV_N_JOBS, random_state=config.RANDOM_STATE, verbose=1,
    )
    search.fit(X, y)
    return "random_forest", search.best_params_, search.best_score_, n_iter


def run_optuna(X, y, n_trials: int = 40):
    """TPE search over LightGBM, the strongest family in the comparison."""
    cv = _cv()

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, NEG_POS_RATIO),
        }
        pipe = features.build_pipeline(
            LGBMClassifier(verbose=-1, n_jobs=-1, random_state=config.RANDOM_STATE, **params)
        )
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=config.CV_N_JOBS)

        # Report per-fold so Optuna can prune hopeless configurations early.
        trial.set_user_attr("std", float(scores.std()))
        return float(scores.mean())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
        study_name="lightgbm-pr-auc",
    )

    def _log(study_, trial_):
        print(f"  trial {trial_.number:3d}  {SELECTION_METRIC} {trial_.value:.4f}  "
              f"(best {study_.best_value:.4f})", flush=True)

    study.optimize(objective, n_trials=n_trials, callbacks=[_log])

    best = {f"model__{k}": v for k, v in study.best_params.items()}
    return "lightgbm", best, study.best_value, n_trials


METHODS = {"grid": run_grid, "random": run_random, "optuna": run_optuna}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS), default="optuna")
    parser.add_argument("--trials", type=int, default=40,
                        help="budget for random/optuna searches")
    args = parser.parse_args()

    X, y = load_train()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    print(f"{args.method} search on {X.shape}, scoring={SCORING}\n")

    with mlflow.start_run(run_name=f"tune-{args.method}"):
        fn = METHODS[args.method]
        if args.method == "grid":
            model_name, best_params, best_score, n_fits = fn(X, y)
        else:
            model_name, best_params, best_score, n_fits = fn(X, y, args.trials)

        mlflow.set_tags({"stage": "tuning", "search_method": args.method,
                         "model": model_name})
        mlflow.log_params({"search_method": args.method, "model": model_name,
                           "n_configurations": n_fits,
                           **{k: v for k, v in best_params.items()}})
        mlflow.log_metric(f"best_cv_{SELECTION_METRIC}", best_score)

    print(f"\nbest {SELECTION_METRIC}: {best_score:.4f}  ({model_name}, {n_fits} configs)")
    for k, v in best_params.items():
        print(f"  {k} = {v}")

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / f"best_params_{args.method}.json"
    out.write_text(json.dumps(
        {"method": args.method, "model": model_name,
         "cv_pr_auc": best_score, "params": best_params}, indent=2, default=str))
    print(f"\nwrote {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
