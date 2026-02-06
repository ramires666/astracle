"""High-visibility plotting helpers for Moon-cycle research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score

from .eval_utils import (
    compute_binary_metrics,
    compute_rolling_metrics,
    compute_statistical_significance,
)


@dataclass(frozen=True)
class VisualizationConfig:
    """Visual style and rolling-window defaults (dark-theme-first)."""

    rolling_window_days: int = 90
    rolling_min_periods: int = 30
    line_break_gap_days: int = 3
    up_color: str = "#2fd384"
    down_color: str = "#ff6b6b"
    price_color: str = "#e2e8f0"
    recall_min_color: str = "#5cc8ff"
    figure_bg: str = "#060a14"
    axes_bg: str = "#0e1525"
    grid_color: str = "#3b4f69"
    rolling_grid_alpha: float = 0.60
    probability_bins: int = 64
    text_color: str = "#e2e8f0"
    muted_text: str = "#9fb2c8"
    split_train_color: str = "#f6bd60"
    split_val_color: str = "#7dd3fc"
    split_test_color: str = "#c4b5fd"
    split_future_color: str = "#94a3b8"
    cm_counts_cmap: str = "Blues"
    cm_norm_cmap: str = "Greens"


def _style_axis(
    ax: plt.Axes,
    vis_cfg: VisualizationConfig,
    with_grid: bool = True,
    grid_alpha: float = 0.35,
) -> None:
    """Apply consistent dark styling to one axis."""
    ax.set_facecolor(vis_cfg.axes_bg)
    for spine in ax.spines.values():
        spine.set_color(vis_cfg.grid_color)
    ax.tick_params(colors=vis_cfg.text_color)
    ax.yaxis.get_offset_text().set_color(vis_cfg.text_color)
    ax.xaxis.label.set_color(vis_cfg.text_color)
    ax.yaxis.label.set_color(vis_cfg.text_color)
    ax.title.set_color(vis_cfg.text_color)
    if with_grid:
        ax.grid(alpha=grid_alpha, linestyle=":", color=vis_cfg.grid_color)


def _style_figure(fig: plt.Figure, vis_cfg: VisualizationConfig, title: str) -> None:
    """Apply figure-level dark theme and title color."""
    fig.patch.set_facecolor(vis_cfg.figure_bg)
    fig.suptitle(title, color=vis_cfg.text_color)


def _style_legend(ax: plt.Axes, vis_cfg: VisualizationConfig, loc: str = "lower right") -> None:
    """Render legend with dark panel and readable text."""
    legend = ax.legend(loc=loc)
    if legend is None:
        return
    legend.get_frame().set_facecolor(vis_cfg.axes_bg)
    legend.get_frame().set_edgecolor(vis_cfg.grid_color)
    legend.get_frame().set_alpha(0.95)
    for text in legend.get_texts():
        text.set_color(vis_cfg.text_color)


def _set_heatmap_annotation_contrast(
    ax: plt.Axes,
    data: np.ndarray,
    cmap_name: str,
) -> None:
    """
    Make heatmap annotation text readable on both bright and dark cells.

    We compute cell luminance from the colormap and choose dark or light text.
    This avoids unreadable combinations like white text on bright yellow/green.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return

    norm = mcolors.Normalize(vmin=float(np.nanmin(arr)), vmax=float(np.nanmax(arr)))
    cmap = cm.get_cmap(cmap_name)
    n_cols = arr.shape[1]

    for idx, text_obj in enumerate(ax.texts):
        i = idx // n_cols
        j = idx % n_cols
        if i >= arr.shape[0] or j >= arr.shape[1]:
            continue

        rgba = cmap(norm(arr[i, j]))
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        text_obj.set_color("#0b1220" if luminance > 0.55 else "#f8fafc")
        text_obj.set_fontweight("bold")


def _line_with_gap_breaks(
    dates: pd.Series,
    values: pd.Series,
    max_gap_days: int,
) -> tuple[pd.Series, pd.Series]:
    """
    Insert NaN after large date gaps so matplotlib does not draw fake diagonals.

    This is especially useful for walk-forward test-only slices where dates are
    not continuous and direct line plotting would connect distant chunks.
    """
    x = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    y = pd.Series(values).astype(float).reset_index(drop=True)
    day_gaps = x.diff().dt.days.fillna(1)
    y = y.mask(day_gaps > max_gap_days)
    return x, y


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


def _draw_split_bands(ax: plt.Axes, df_plot: pd.DataFrame, vis_cfg: VisualizationConfig) -> None:
    """
    Draw subtle top bands showing train/val/test/future timeline regions.

    This directly addresses the requirement for transparent split boundaries.
    """
    if "split_role" not in df_plot.columns:
        return

    role_colors = {
        "train": vis_cfg.split_train_color,
        "val": vis_cfg.split_val_color,
        "test": vis_cfg.split_test_color,
        "future": vis_cfg.split_future_color,
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
                color="#111827",
                fontsize=10,
                fontweight="bold",
            )

        segment_start = i


def plot_confusion_matrix_pair(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    vis_cfg: VisualizationConfig,
) -> None:
    """Plot count and normalized confusion matrices side by side."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(vis_cfg.figure_bg)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=vis_cfg.cm_counts_cmap,
        cbar=False,
        xticklabels=["DOWN", "UP"],
        yticklabels=["DOWN", "UP"],
        ax=axes[0],
        annot_kws={"color": vis_cfg.text_color, "fontsize": 12},
    )
    _set_heatmap_annotation_contrast(axes[0], cm, vis_cfg.cm_counts_cmap)
    axes[0].set_title("Confusion matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    _style_axis(axes[0], vis_cfg, with_grid=False)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap=vis_cfg.cm_norm_cmap,
        cbar=False,
        xticklabels=["DOWN", "UP"],
        yticklabels=["DOWN", "UP"],
        ax=axes[1],
        annot_kws={"color": vis_cfg.text_color, "fontsize": 12},
    )
    _set_heatmap_annotation_contrast(axes[1], cm_norm, vis_cfg.cm_norm_cmap)
    axes[1].set_title("Confusion matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    _style_axis(axes[1], vis_cfg, with_grid=False)

    _style_figure(fig, vis_cfg, title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_price_background_pair(
    df_plot: pd.DataFrame,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    vis_cfg: VisualizationConfig,
    title: str,
    metrics: Optional[Dict[str, float]] = None,
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
    fig.patch.set_facecolor(vis_cfg.figure_bg)
    line_x, line_y = _line_with_gap_breaks(dates, prices, max_gap_days=vis_cfg.line_break_gap_days)

    axes[0].plot(line_x, line_y, color=vis_cfg.price_color, linewidth=2.0)
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
    if metrics is None:
        axes[0].set_title("Predicted labels over price")
    else:
        axes[0].set_title(
            "Predicted labels | ACC={:.3f} | R_MIN={:.3f} | MCC={:.3f}".format(
                float(metrics["accuracy"]),
                float(metrics["recall_min"]),
                float(metrics["mcc"]),
            )
        )
    _style_axis(axes[0], vis_cfg, with_grid=True)
    _draw_split_bands(axes[0], data, vis_cfg)

    axes[1].plot(line_x, line_y, color=vis_cfg.price_color, linewidth=2.0)
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
    if metrics is None:
        axes[1].set_title("True labels over price")
    else:
        axes[1].set_title(
            "True labels | R_DOWN={:.3f} | R_UP={:.3f} | GAP={:.3f}".format(
                float(metrics["recall_down"]),
                float(metrics["recall_up"]),
                float(metrics["recall_gap"]),
            )
        )
    _style_axis(axes[1], vis_cfg, with_grid=True)
    _draw_split_bands(axes[1], data, vis_cfg)

    _style_figure(fig, vis_cfg, title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_rolling_metrics(
    rolling_df: pd.DataFrame,
    vis_cfg: VisualizationConfig,
    title: str,
) -> None:
    """Plot rolling accuracy + rolling recalls."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.patch.set_facecolor(vis_cfg.figure_bg)

    axes[0].plot(rolling_df["date"], rolling_df["rolling_accuracy"], color="#f4d35e", label="Rolling ACC")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title(f"Rolling accuracy ({vis_cfg.rolling_window_days}-day window)")
    _style_axis(axes[0], vis_cfg, with_grid=True, grid_alpha=vis_cfg.rolling_grid_alpha)
    _style_legend(axes[0], vis_cfg, loc="lower right")

    axes[1].plot(rolling_df["date"], rolling_df["rolling_recall_down"], color=vis_cfg.down_color, label="Recall DOWN")
    axes[1].plot(rolling_df["date"], rolling_df["rolling_recall_up"], color=vis_cfg.up_color, label="Recall UP")
    axes[1].plot(
        rolling_df["date"],
        rolling_df["rolling_recall_min"],
        color=vis_cfg.recall_min_color,
        linewidth=2.4,
        label="Recall MIN",
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Rolling class recalls")
    _style_axis(axes[1], vis_cfg, with_grid=True, grid_alpha=vis_cfg.rolling_grid_alpha)
    _style_legend(axes[1], vis_cfg, loc="lower right")

    _style_figure(fig, vis_cfg, title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_probability_diagnostics(
    y_true: np.ndarray,
    y_prob_up: np.ndarray,
    title: str,
    vis_cfg: VisualizationConfig,
) -> None:
    """
    Show probability histogram by true class.

    This helps verify that model probabilities are not degenerate around one value.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob_up = np.asarray(y_prob_up, dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    fig.patch.set_facecolor(vis_cfg.figure_bg)
    bins = int(max(32, vis_cfg.probability_bins))
    ax.hist(y_prob_up[y_true == 0], bins=bins, alpha=0.65, label="True DOWN", color=vis_cfg.down_color)
    ax.hist(y_prob_up[y_true == 1], bins=bins, alpha=0.65, label="True UP", color=vis_cfg.up_color)

    try:
        auc = float(roc_auc_score(y_true, y_prob_up))
        auc_text = f"ROC-AUC={auc:.3f}"
    except Exception:
        auc_text = "ROC-AUC=NA"

    ax.set_title(f"{title} ({auc_text})")
    ax.set_xlabel("Predicted probability of UP")
    ax.set_ylabel("Count")
    _style_axis(ax, vis_cfg, with_grid=True)
    _style_legend(ax, vis_cfg, loc="best")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
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
        plot_confusion_matrix_pair(y_true=y_true, y_pred=y_pred, title=title, vis_cfg=vis_cfg)
        plot_price_background_pair(
            df_plot=df_plot,
            y_pred=y_pred,
            y_true=y_true,
            vis_cfg=vis_cfg,
            title=title,
            metrics=metrics,
        )
        plot_rolling_metrics(rolling_df=rolling_df, vis_cfg=vis_cfg, title=title)
        if y_prob_up is not None:
            plot_probability_diagnostics(y_true=y_true, y_prob_up=y_prob_up, title=title, vis_cfg=vis_cfg)

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
