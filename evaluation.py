"""
evaluation.py
==============
Performance evaluation utilities:
  - Accuracy, Precision, Recall, F1-score (per-class and weighted)
  - Confusion matrix heatmap
  - Bar chart comparing all classifier accuracies
  - Saves all figures and a summary CSV to E:/photos ml/project/results/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

RESULTS_DIR = r"E:\photos ml\project\results"
CLASS_NAMES = [
    "class_01", "class_02", "class_03", "class_04", "class_05",
    "class_06", "class_07", "class_08", "class_09", "class_10",
]


def evaluate_classifier(name, model, X_test, y_test):
    """
    Evaluate a trained model on the test set.
    Returns a dict of scalar metrics and saves a confusion matrix PNG.
    """
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred,        average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (weighted)")
    print(f"  Recall    : {rec:.4f}  (weighted)")
    print(f"  F1-Score  : {f1:.4f}  (weighted)")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # ── Confusion matrix figure ────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    safe_name = name.replace(" ", "_").replace("/", "_")
    fig_path  = os.path.join(RESULTS_DIR, f"cm_{safe_name}.png")
    fig.savefig(fig_path, dpi=120)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {fig_path}")

    return {
        "Classifier": name,
        "Accuracy":   round(acc,  4),
        "Precision":  round(prec, 4),
        "Recall":     round(rec,  4),
        "F1-Score":   round(f1,   4),
    }


def save_summary(results_list):
    """
    Save a CSV and a bar-chart PNG comparing all classifiers.
    results_list: list of dicts returned by evaluate_classifier().
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.DataFrame(results_list)
    df_sorted = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(RESULTS_DIR, "results_summary.csv")
    df_sorted.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"  SUMMARY (sorted by Accuracy)")
    print(f"{'='*60}")
    print(df_sorted.to_string(index=False))
    print(f"\nSummary CSV saved -> {csv_path}")

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    metrics   = ["Accuracy", "Precision", "Recall", "F1-Score"]
    n_clf     = len(df_sorted)
    n_metrics = len(metrics)
    x         = np.arange(n_clf)
    width     = 0.18

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = ["#3b82d4", "#10b981", "#f59e0b", "#ef4444"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df_sorted[metric], width, label=metric, color=color, alpha=0.85)

    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(df_sorted["Classifier"], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classifier Performance Comparison", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar_group_start in range(n_clf):
        for m_idx, metric in enumerate(metrics):
            val = df_sorted.loc[bar_group_start, metric]
            xpos = bar_group_start + m_idx * width
            ax.text(xpos, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, "performance_comparison.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"Performance chart saved -> {chart_path}")

    return df_sorted
