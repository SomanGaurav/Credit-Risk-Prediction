"""The candidate model zoo.

Each entry is a fresh estimator configured for a ~6.7% positive rate. Where a
library exposes class weighting we use it, since it is cheaper and generally
stronger than resampling on this dataset; SMOTE is compared separately via the
--smote flag on train.py.
"""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_STATE

# Ratio of negatives to positives, used by the boosters' scale_pos_weight.
NEG_POS_RATIO = 14.0


def get_models(balanced: bool = True) -> dict:
    """Build the zoo. When balanced is False, models train on the raw prior --
    used for the SMOTE comparison, where resampling already rebalances the data."""
    cw = "balanced" if balanced else None
    spw = NEG_POS_RATIO if balanced else 1.0

    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight=cw, random_state=RANDOM_STATE
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, class_weight=cw, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            class_weight=cw, n_jobs=-1, random_state=RANDOM_STATE
        ),
        # A kernel SVM is O(n^2) and intractable at 120k rows. LinearSVC keeps the
        # hinge-loss formulation and stays linear in n; the calibration wrapper
        # turns its decision function into the probabilities the API needs.
        "svm_linear": CalibratedClassifierCV(
            LinearSVC(C=0.1, class_weight=cw, dual="auto", max_iter=5000,
                      random_state=RANDOM_STATE),
            method="sigmoid", cv=3,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="auc", n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            verbose=-1, n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "catboost": CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            scale_pos_weight=spw, verbose=0, random_seed=RANDOM_STATE,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=60, early_stopping=True,
            random_state=RANDOM_STATE
        ),
    }
