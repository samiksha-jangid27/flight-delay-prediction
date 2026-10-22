"""Regenerable high-resolution evaluation figures (headless backend)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from .calibration import reliability


def render(frames: dict, output: Path, *, thresholds=None, variants=None) -> None:
    """Export ranking, reliability, proper-score and operating-point charts."""
    output.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=220)
        plt.close(fig)

    for kind in ["roc", "precision_recall"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, frame in frames.items():
            y, p = frame.y_true, frame.y_probability
            if kind == "roc":
                x, z, _ = roc_curve(y, p)
            else:
                z, x, _ = precision_recall_curve(y, p)
            ax.plot(x, z, label=name)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], "k--", label="Chance")
        else:
            ax.axhline(
                next(iter(frames.values())).y_true.mean(),
                color="k",
                ls="--",
                label="Prevalence",
            )
        ax.set(
            xlabel="False positive rate" if kind == "roc" else "Recall",
            ylabel="True positive rate" if kind == "roc" else "Precision",
            title=kind.replace("_", " ").title(),
        )
        ax.legend()
        save(fig, kind)
    if variants:
        for name, maps in variants.items():
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot([0, 1], [0, 1], "k--")
            for method, frame in maps.items():
                bins = reliability(frame.y_true, frame.y_probability)
                ax.plot(bins.mean_probability, bins.observed_rate, "o-", label=method)
                bins.to_csv(output / f"{name}_{method}_reliability.csv", index=False)
            ax.set(
                xlabel="Mean predicted probability",
                ylabel="Observed delay rate",
                title=f"{name}: reliability (10 equal-width bins)",
            )
            ax.legend()
            save(fig, f"{name}_reliability")
    if thresholds:
        for name, table in thresholds.items():
            for columns, title in [
                (["precision", "recall", "f1"], "threshold_metrics"),
                (["fp", "fn"], "error_tradeoff"),
            ]:
                fig, ax = plt.subplots(figsize=(7, 5))
                for col in columns:
                    ax.plot(table.threshold, table[col], label=col)
                ax.set(
                    xlabel="Probability threshold",
                    ylabel="Count" if columns == ["fp", "fn"] else "Score",
                    title=name,
                )
                ax.legend()
                save(fig, f"{name}_{title}")


def summary_plots(results: dict, output: Path) -> None:
    """Plot proper scores, final metric comparison and confusion matrices."""
    output.mkdir(parents=True, exist_ok=True)
    for metric in ["brier", "log_loss", "average_precision", "roc_auc", "f1"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(list(results), [r[metric] for r in results.values()])
        ax.tick_params(axis="x", rotation=25)
        ax.set(ylabel=metric, title=metric.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(output / f"{metric}.png", dpi=220)
        plt.close(fig)
    for name, r in results.items():
        fig, ax = plt.subplots(figsize=(5, 4))
        matrix = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xlabel="Predicted class",
            ylabel="True class",
            title=name,
        )
        fig.tight_layout()
        fig.savefig(output / f"{name}_confusion.png", dpi=220)
        plt.close(fig)
