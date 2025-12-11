"""Exploratory analysis on the training split. Writes figures to reports/figures/.

Run after src.data. Deliberately reads only the *train* split so that nothing
learned here informs choices that are then evaluated on test.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid")

# Heavy right tails make raw histograms unreadable; clip at this quantile for display only.
_DISPLAY_CLIP = 0.99


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(config.ROOT)}")


def plot_target_balance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    counts = df[config.TARGET].value_counts().sort_index()
    ax.bar(["repaid", "default"], counts.values, color=["#4c72b0", "#c44e52"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Class balance")
    ax.set_ylim(0, counts.max() * 1.18)
    _save(fig, "target_balance.png")


def plot_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    for ax, col in zip(axes.ravel(), config.FEATURES):
        data = df[col].clip(upper=df[col].quantile(_DISPLAY_CLIP))
        sns.histplot(data=data, bins=40, ax=ax, color="#4c72b0")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
    for ax in axes.ravel()[len(config.FEATURES):]:
        ax.set_visible(False)
    fig.suptitle(f"Feature distributions (clipped at {_DISPLAY_CLIP:.0%} for display)", y=0.995)
    _save(fig, "distributions.png")


def plot_default_rate_by_feature(df: pd.DataFrame) -> None:
    """Default rate across deciles -- shows monotonicity and where risk concentrates."""
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    for ax, col in zip(axes.ravel(), config.FEATURES):
        try:
            bins = pd.qcut(df[col], 10, duplicates="drop")
        except ValueError:
            continue
        rate = df.groupby(bins, observed=True)[config.TARGET].mean()
        ax.plot(range(len(rate)), rate.values, marker="o", color="#c44e52")
        ax.axhline(df[config.TARGET].mean(), ls="--", c="grey", lw=1, label="base rate")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("decile")
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(config.FEATURES):]:
        ax.set_visible(False)
    fig.suptitle("Default rate by feature decile", y=0.995)
    _save(fig, "default_rate_by_decile.png")


def plot_correlation(df: pd.DataFrame):
    cols = config.FEATURES + [config.TARGET]
    corr = df[cols].corr(method="spearman")  # rank-based: robust to the heavy tails
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, annot_kws={"size": 7}, ax=ax)
    ax.set_title("Spearman correlation")
    _save(fig, "correlation.png")
    return corr


def main() -> None:
    df = pd.read_parquet(config.TRAIN_PARQUET)
    print(f"EDA on train split {df.shape}\n")

    plot_target_balance(df)
    plot_distributions(df)
    plot_default_rate_by_feature(df)
    corr = plot_correlation(df)

    print("\n--- correlation with target (Spearman, sorted) ---")
    target_corr = corr[config.TARGET].drop(config.TARGET).sort_values(key=abs, ascending=False)
    print(target_corr.round(3).to_string())

    print("\n--- collinear feature pairs (|rho| > 0.5) ---")
    feats = corr.loc[config.FEATURES, config.FEATURES]
    seen = set()
    for a in config.FEATURES:
        for b in config.FEATURES:
            if a != b and (b, a) not in seen and abs(feats.loc[a, b]) > 0.5:
                seen.add((a, b))
                print(f"  {a} ~ {b}: {feats.loc[a, b]:.3f}")
    if not seen:
        print("  none")

    print("\n--- default rate by engineered flag ---")
    for flag in ["HasDelinquencySentinel", "MonthlyIncomeMissing", "NumberOfDependentsMissing"]:
        rates = df.groupby(flag)[config.TARGET].agg(["mean", "size"])
        print(f"  {flag}:")
        for val, row in rates.iterrows():
            print(f"    ={val}: {row['mean']:.4f} default over {int(row['size']):,} rows")


if __name__ == "__main__":
    main()
