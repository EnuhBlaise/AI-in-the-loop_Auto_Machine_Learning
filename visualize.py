#!/usr/bin/env python3
"""Generate a performance visualization across all experiments."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_experiments(experiments_dir: str = "experiments") -> list[dict]:
    """Load all experiment summaries from the experiments directory."""
    base = Path(experiments_dir)
    experiments = []
    for exp_dir in sorted(base.iterdir()):
        summary_path = exp_dir / "summary.json"
        if exp_dir.is_dir() and summary_path.exists():
            with open(summary_path) as f:
                experiments.append(json.load(f))
    return experiments


def plot_experiment_performance(experiments: list[dict], output_path: str = "experiment_performance.png"):
    """
    Create a grouped bar chart comparing val_auroc, val_f1, and val_accuracy
    across all experiments, annotated with architecture labels.
    """
    if not experiments:
        print("No experiments found.")
        return

    ids = [e["id"] for e in experiments]
    architectures = [e["config"]["model"]["architecture"] for e in experiments]
    labels = [f"{eid}\n({arch})" for eid, arch in zip(ids, architectures)]

    auroc = [e["metrics"].get("val_auroc", 0) for e in experiments]
    f1 = [e["metrics"].get("val_f1", 0) for e in experiments]
    accuracy = [e["metrics"].get("val_accuracy", 0) for e in experiments]

    x = np.arange(len(ids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(ids) * 2.2), 6))

    bars1 = ax.bar(x - width, auroc, width, label="val_auroc", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x, f1, width, label="val_f1", color="#4CAF50", edgecolor="white")
    bars3 = ax.bar(x + width, accuracy, width, label="val_accuracy", color="#FF9800", edgecolor="white")

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    # Highlight best AUROC
    best_idx = int(np.argmax(auroc))
    ax.annotate(
        "BEST",
        xy=(best_idx - width, auroc[best_idx]),
        xytext=(0, 18),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="#D32F2F",
        arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=1.5),
    )

    ax.set_xlabel("Experiment", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("BioML Autopilot — Experiment Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved performance visualization to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    experiments = load_experiments()
    plot_experiment_performance(experiments)
