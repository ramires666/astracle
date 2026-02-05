"""
High-visibility plotting helpers for Moon-cycle research.

This module focuses only on charts and presentation:
- confusion matrix
- stacked price chart with predicted/true direction backgrounds
- rolling metrics chart
- optional probability distribution chart
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score

from .eval_utils import (
    compute_binary_metrics,
    compute_rolling_metrics,
    compute_statistical_significance,
)


@dataclass(frozen=True)
class VisualizationConfig:
    """Small container for visual style and rolling window defaults."""

    rolling_window_days: int = 90
    rolling_min_periods: int = 30
    up_color: str = "#14b86d"
    down_color: str = "#d9534f"
    price_color: str = "#f2f2f2"


def _draw_direction_background(
    ax: plt.Axes,
    dates: pd.Series,
    labels: np.ndarray,
    y_min: float,
    y_max: float,
    up_color: str,
    down_color: str,
    alpha: float = 0.18,
) -> None:
    """
    Draw UP/DOWN color background under a price line.

    - Green for UP (1)
    - Red for DOWN (0)
    """
    labels = np.asarray(labels, dtype=np.int32)

    ax.fill_between(
        pd.to_datetime(dates),
        y_min,
        y_max,
        where=labels == 1,
        color=up_color,
        alpha=alpha,
        step="mid",
        linewidth=0,
    )
    ax.fill_between(
        pd.to_datetime(dates),
        y_min,
        y_max,
        where=labels == 0,
        color=down_color,
        alpha=alpha,
        step="mid",
        linewidth=0,
    )


def _draw_split_bands(ax: plt.Axes, df_plot: pd.DataFrame) -> None:
    """
    Draw subtle top bands showing train/val/test/future timeline regions.

    This directly addresses the requirement for transparent split boundaries.
    """
    if "split_role" not in df_plot.columns:
        return

    role_colors = {
        "train": "#ffd166",
        "val": "#8ecae6",
        "test": "#c77dff",
        "future": "#adb5bd",
    }

    y_min, y_max = ax.get_ylim()
    band_height = (y_max - y_min) * 0.03
    band_bottom = y_max - band_height

    roles = df_plot["split_role"].fillna("unknown").astype(str).to_numpy()
    dates = pd.to_datetime(df_plot["date"]).reset_index(drop=True)

    if len(dates) == 0:
        return

    segment_start = 0
    for i in range(1, len(roles) + 1):
        is_boundary = (i == len(roles)) or (roles[i] != roles[segment_start])
        if not is_boundary:
            continue

        role = roles[segment_start]
        color = role_colors.get(role, "#6c757d")
        x0 = dates.iloc[segment_start]
        x1 = dates.iloc[i - 1]

        ax.axvspan(x0, x1, ymin=0.97, ymax=1.0, color=color, alpha=0.65, linewidth=0)

        # Label only if the segment is reasonably long, so labels do not overlap.
        if (i - segment_start) >= 10:
            x_mid = x0 + (x1 - x0) / 2
            ax.text(
                x_mid,
                band_bottom + band_height * 0.5,
                role.upper(),
                ha="center",
                va="center",
                color="#1f2937",
                fontsize=10,
                fontweight="bold",
            )

        segment_start = i


def plot_confusion_matrix_pair(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    """Plot count and normalized confusion matrices side by side."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["DOWN", "UP"],
        yticklabels=["DOWN", "UP"],
        ax=axes[0],
    )
    axes[0].set_title("Confusion matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        cbar=False,
        xticklabels=["DOWN", "UP"],
        yticklabels=["DOWN", "UP"],
        ax=axes[1],
    )
    axes[1].set_title("Confusion matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_price_background_pair(
    df_plot: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    vis_cfg: VisualizationConfig,
    title: str,
) -> None:
    """
    Plot two stacked charts for easy visual comparison:
    - Top chart: predicted labels
    - Bottom chart: true labels
    """
    data = df_plot.copy().reset_index(drop=True)
    dates = pd.to_datetime(data["date"])
    prices = data["close"].astype(float)

    y_min = float(prices.min())
    y_max = float(prices.max())
    margin = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
    y_min_plot = y_min - margin
    y_max_plot = y_max + margin

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)

    axes[0].plot(dates, prices, color=vis_cfg.price_color, linewidth=1.8)
    _draw_direction_background(
        ax=axes[0],
        dates=dates,
        labels=y_pred,
        y_min=y_min_plot,
        y_max=y_max_plot,
        up_color=vis_cfg.up_color,
        down_color=vis_cfg.down_color,
        alpha=0.20,
    )
    axes[0].set_ylim(y_min_plot, y_max_plot)
    axes[0].set_title("Predicted labels over price")
    axes[0].grid(alpha=0.25, linestyle=":")
    _draw_split_bands(axes[0], data)

    axes[1].plot(dates, prices, color=vis_cfg.price_color, linewidth=1.8)
    _draw_direction_background(
        ax=axes[1],
        dates=dates,
        labels=y_true,
        y_min=y_min_plot,
        y_max=y_max_plot,
        up_color=vis_cfg.up_color,
        down_color=vis_cfg.down_color,
        alpha=0.20,
    )
    axes[1].set_ylim(y_min_plot, y_max_plot)
    axes[1].set_title("True labels over price")
    axes[1].grid(alpha=0.25, linestyle=":")
    _draw_split_bands(axes[1], data)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_rolling_metrics(
    rolling_df: pd.DataFrame,
    vis_cfg: VisualizationConfig,
    title: str,
) -> None:
    """Plot rolling accuracy + rolling recalls."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    axes[0].plot(rolling_df["date"], rolling_df["rolling_accuracy"], color="#f4d35e", label="Rolling ACC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title(f"Rolling accuracy ({vis_cfg.rolling_window_days}-day window)")
    axes[0].grid(alpha=0.25, linestyle=":")
    axes[0].legend(loc="lower right")

    axes[1].plot(rolling_df["date"], rolling_df["rolling_recall_down"], color=vis_cfg.down_color, label="Recall DOWN")
    axes[1].plot(rolling_df["date"], rolling_df["rolling_recall_up"], color=vis_cfg.up_color, label="Recall UP")
    axes[1].plot(rolling_df["date"], rolling_df["rolling_recall_min"], color="#4dabf7", linewidth=2.2, label="Recall MIN")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Rolling class recalls")
    axes[1].grid(alpha=0.25, linestyle=":")
    axes[1].legend(loc="lower right")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_probability_diagnostics(y_true: np.ndarray, y_prob_up: np.ndarray, title: str) -> None:
    """
    Show probability histogram by true class.

    This helps verify that model probabilities are not degenerate around one value.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob_up = np.asarray(y_prob_up, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.hist(y_prob_up[y_true == 0], bins=20, alpha=0.6, label="True DOWN", color="#d9534f")
    ax.hist(y_prob_up[y_true == 1], bins=20, alpha=0.6, label="True UP", color="#14b86d")

    try:
        auc = float(roc_auc_score(y_true, y_prob_up))
        auc_text = f"ROC-AUC={auc:.3f}"
    except Exception:
        auc_text = "ROC-AUC=NA"

    ax.set_title(f"{title} ({auc_text})")
    ax.set_xlabel("Predicted probability of UP")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.2, linestyle=":")
    plt.tight_layout()
    plt.show()


def evaluate_with_visuals(
    df_plot: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob_up: Optional[np.ndarray] = None,
    title: str = "Moon-cycle model evaluation",
    vis_cfg: VisualizationConfig = VisualizationConfig(),
    show_visuals: bool = True,
) -> Dict[str, object]:
    """
    Run the full visual evaluation package and return all computed objects.

    By default we keep this very visual, because the project request explicitly
    prefers concrete, chart-based explanations over abstract numbers.
    """
    metrics = compute_binary_metrics(y_true=y_true, y_pred=y_pred)
    significance = compute_statistical_significance(y_true=y_true, y_pred=y_pred, random_baseline=0.5)
    rolling_df = compute_rolling_metrics(
        dates=df_plot["date"],
        y_true=y_true,
        y_pred=y_pred,
        window=vis_cfg.rolling_window_days,
        min_periods=vis_cfg.rolling_min_periods,
    )

    if show_visuals:
        plot_confusion_matrix_pair(y_true=y_true, y_pred=y_pred, title=title)
        plot_price_background_pair(
            df_plot=df_plot,
            y_pred=y_pred,
            y_true=y_true,
            vis_cfg=vis_cfg,
            title=title,
        )
        plot_rolling_metrics(rolling_df=rolling_df, vis_cfg=vis_cfg, title=title)
        if y_prob_up is not None:
            plot_probability_diagnostics(y_true=y_true, y_prob_up=y_prob_up, title=title)

    print("=" * 80)
    print(title)
    print("=" * 80)
    print(
        "ACC={acc:.4f} | BAL_ACC={bal:.4f} | MCC={mcc:.4f} | "
        "R_DOWN={rd:.4f} | R_UP={ru:.4f} | R_MIN={rmin:.4f}".format(
            acc=metrics["accuracy"],
            bal=metrics["balanced_accuracy"],
            mcc=metrics["mcc"],
            rd=metrics["recall_down"],
            ru=metrics["recall_up"],
            rmin=metrics["recall_min"],
        )
    )
    print(
        "Significance vs random(50%): "
        f"p-value={significance['p_value_vs_random']:.6g}, "
        f"95% CI for ACC=[{significance['ci95_low']:.4f}, {significance['ci95_high']:.4f}]"
    )
    print("=" * 80)

    return {
        "metrics": metrics,
        "significance": significance,
        "rolling": rolling_df,
    }
