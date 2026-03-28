"""Clinical analysis, bias detection, and interpretability for Triagegeist.

Provides:
- SHAP-based model interpretability
- Demographic triage bias analysis (sex, age, language, insurance)
- Vital sign distribution by acuity
- Confusion/error analysis for misclassified patients
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from .config import TARGET_COL, ACUITY_LABELS


def plot_acuity_distribution(y, ax=None):
    """Bar plot of triage acuity class distribution."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    counts = pd.Series(y).value_counts().sort_index()
    colors = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1976d2"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{count:,}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Triage Acuity (ESI)")
    ax.set_ylabel("Count")
    ax.set_title("Triage Acuity Distribution")
    ax.set_xticks(counts.index)
    ax.set_xticklabels([f"{k}: {v}" for k, v in ACUITY_LABELS.items()], fontsize=8)
    return ax


def plot_confusion_matrix(y_true, y_pred, normalize=True, ax=None):
    """Confusion matrix heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=list(ACUITY_LABELS.values()),
                yticklabels=list(ACUITY_LABELS.values()), ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Normalized)" if normalize else "Confusion Matrix")
    return ax


def analyze_demographic_bias(df, pred_col="pred_acuity"):
    """Analyze triage bias across demographic groups.

    Returns a dict of DataFrames showing mean predicted vs actual acuity
    by sex, age_group, language, and insurance_type.
    """
    bias_results = {}
    demo_cols = ["sex", "age_group", "language", "insurance_type"]

    for col in demo_cols:
        if col not in df.columns:
            continue
        group = df.groupby(col).agg(
            mean_actual=(TARGET_COL, "mean"),
            mean_predicted=(pred_col, "mean"),
            count=(TARGET_COL, "count"),
        ).reset_index()
        group["bias_delta"] = group["mean_predicted"] - group["mean_actual"]
        group = group.sort_values("bias_delta", ascending=False)
        bias_results[col] = group

    return bias_results


def plot_bias_analysis(bias_results, figsize=(14, 10)):
    """Visualize demographic bias deltas."""
    n_plots = len(bias_results)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, (col, data) in enumerate(bias_results.items()):
        ax = axes[idx]
        data_sorted = data.sort_values("bias_delta")
        colors = ["#d32f2f" if d > 0.05 else "#1976d2" if d < -0.05 else "#9e9e9e"
                  for d in data_sorted["bias_delta"]]
        ax.barh(data_sorted[col].astype(str), data_sorted["bias_delta"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Bias Δ (predicted − actual acuity)")
        ax.set_title(f"Triage Bias by {col.replace('_', ' ').title()}")
        ax.tick_params(axis="y", labelsize=8)

    for idx in range(n_plots, 4):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig


def plot_vitals_by_acuity(df, figsize=(16, 10)):
    """Box plots of key vitals stratified by triage acuity."""
    vitals = ["heart_rate", "systolic_bp", "respiratory_rate",
              "temperature_c", "spo2", "shock_index", "news2_score", "gcs_total"]
    vitals = [v for v in vitals if v in df.columns]

    n_cols = 4
    n_rows = (len(vitals) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    colors = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1976d2"]
    palette = {i+1: c for i, c in enumerate(colors)}

    for idx, vital in enumerate(vitals):
        sns.boxplot(data=df, x=TARGET_COL, y=vital, ax=axes[idx],
                    palette=palette, fliersize=1)
        axes[idx].set_title(vital.replace("_", " ").title())
        axes[idx].set_xlabel("Acuity")

    for idx in range(len(vitals), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Vital Signs Distribution by Triage Acuity", fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def error_analysis(df, y_true_col=TARGET_COL, y_pred_col="pred_acuity"):
    """Analyze characteristics of misclassified patients."""
    df = df.copy()
    df["error"] = df[y_pred_col] - df[y_true_col]
    df["abs_error"] = df["error"].abs()
    df["is_correct"] = (df["error"] == 0).astype(int)
    df["undertriage"] = (df[y_pred_col] > df[y_true_col]).astype(int)  # predicted less urgent
    df["overtriage"] = (df[y_pred_col] < df[y_true_col]).astype(int)   # predicted more urgent

    summary = {
        "accuracy": df["is_correct"].mean(),
        "mean_abs_error": df["abs_error"].mean(),
        "undertriage_rate": df["undertriage"].mean(),
        "overtriage_rate": df["overtriage"].mean(),
        "undertriage_by_acuity": df.groupby(y_true_col)["undertriage"].mean().to_dict(),
        "overtriage_by_acuity": df.groupby(y_true_col)["overtriage"].mean().to_dict(),
    }
    return summary, df
