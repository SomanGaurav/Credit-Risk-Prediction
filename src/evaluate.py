"""Score the serialised model on the held-out test split.

    python -m src.evaluate

This is the only script that touches data/processed/test.parquet. The threshold
comes from model_meta.json (chosen during build_model on out-of-fold training
predictions), so nothing here is fit to the test data.
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config

METRICS_PATH = config.REPORTS_DIR / "test_metrics.json"


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote reports/figures/{name}")


def plot_curves(y, proba, threshold: float) -> None:
    fpr, tpr, _ = roc_curve(y, proba)
    precision, recall, thresholds = precision_recall_curve(y, proba)
    roc_auc, pr_auc = roc_auc_score(y, proba), average_precision_score(y, proba)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax1.plot(fpr, tpr, color="#c44e52", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="random")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax1.legend(loc="lower right")

    ax2.plot(recall, precision, color="#4c72b0", lw=2, label=f"PR (AP = {pr_auc:.4f})")
    ax2.axhline(y.mean(), ls="--", c="grey", lw=1, label=f"base rate = {y.mean():.3f}")
    # Mark where the shipped threshold actually sits on the trade-off curve.
    idx = int(np.searchsorted(thresholds, threshold))
    idx = min(idx, len(recall) - 1)
    ax2.plot(recall[idx], precision[idx], "o", ms=9, color="#c44e52",
             label=f"threshold = {threshold:.3f}")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    ax2.legend(loc="upper right")

    _save(fig, "evaluation_curves.png")


def plot_confusion(y, pred) -> None:
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["repaid", "default"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="predicted", ylabel="actual", title="Confusion matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    _save(fig, "confusion_matrix.png")


def main() -> None:
    if not config.MODEL_PATH.exists():
        raise SystemExit("No model found. Run: python -m src.build_model")

    pipe = joblib.load(config.MODEL_PATH)
    meta = json.loads(config.MODEL_META_PATH.read_text())
    threshold = meta["decision_threshold"]

    df = pd.read_parquet(config.TEST_PARQUET)
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]

    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "model": meta["model_name"],
        "source": meta["source"],
        "threshold": threshold,
        "n_test_rows": int(len(y)),
        "test_positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        # Threshold-free: these describe ranking quality, not the cut-off.
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
    }

    print(f"Evaluating {meta['model_name']} ({meta['source']}) "
          f"on {len(y):,} held-out rows, threshold {threshold:.4f}\n")
    for k in ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]:
        print(f"  {k:10s} {metrics[k]:.4f}")

    print("\n" + classification_report(y, pred, target_names=["repaid", "default"],
                                       digits=4))

    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"caught {tp:,} of {tp + fn:,} defaulters; "
          f"flagged {fp:,} good customers ({fp / (fp + tn):.1%} of them)")

    plot_curves(y, proba, threshold)
    plot_confusion(y, pred)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"  wrote {METRICS_PATH.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
