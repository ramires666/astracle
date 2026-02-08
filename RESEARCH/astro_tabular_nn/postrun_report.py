"""Default post-run diagnostics for tabular NN notebooks.

This module is designed for notebook usage right after each training/trial run.
It produces:
- a confusion matrix chart (if matrix counts are available in results),
- class-balance diagnostics (true/pred up/down shares),
- recall-up/recall-down bars across train/val/test,
- a compact metrics table for quick visual QA.
"""

from __future__ import annotations

import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:  # pragma: no cover - notebook helper fallback
    display = None


def _has_columns(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns for c in cols)


def _sorted_results(results: pd.DataFrame) -> pd.DataFrame:
    """Sort results with the same priority used in runners."""
    if results.empty:
        return results.copy()

    order = [c for c in ["test_cutoff_score", "test_recall_min", "test_mcc", "test_acc"] if c in results.columns]
    if not order:
        return results.reset_index(drop=True)

    asc = [False if c != "test_recall_gap" else True for c in order]
    return results.sort_values(by=order, ascending=asc).reset_index(drop=True)


def _extract_confusion_matrix(row: pd.Series, split_name: str) -> np.ndarray | None:
    """Rebuild confusion matrix from flattened `split_cm_ij` columns."""
    pattern = re.compile(rf"^{re.escape(split_name)}_cm_(\d)(\d)$")
    found: dict[tuple[int, int], int] = {}

    for col in row.index:
        match = pattern.match(str(col))
        if match is None:
            continue
        i = int(match.group(1))
        j = int(match.group(2))
        found[(i, j)] = int(round(float(row[col])))

    if not found:
        return None

    n_classes = max(max(i, j) for i, j in found.keys()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for (i, j), value in found.items():
        cm[i, j] = int(value)
    return cm


def _infer_n_classes(row: pd.Series) -> int:
    cm_test = _extract_confusion_matrix(row=row, split_name="test")
    if cm_test is not None:
        return int(cm_test.shape[0])

    recall_side = float(row.get("test_recall_sideways", np.nan))
    if np.isfinite(recall_side):
        return 3
    return 2


def _class_labels(n_classes: int) -> List[str]:
    if int(n_classes) >= 3:
        return ["DOWN", "SIDEWAYS", "UP"][: int(n_classes)]
    return ["DOWN", "UP"]


def _split_metrics_table(row: pd.Series, splits: List[str]) -> pd.DataFrame:
    rows: List[dict[str, float | str]] = []
    for split in splits:
        rows.append(
            {
                "split": split,
                "acc": float(row.get(f"{split}_acc", np.nan)),
                "mcc": float(row.get(f"{split}_mcc", np.nan)),
                "recall_down": float(row.get(f"{split}_recall_down", np.nan)),
                "recall_up": float(row.get(f"{split}_recall_up", np.nan)),
                "recall_min": float(row.get(f"{split}_recall_min", np.nan)),
                "recall_gap": float(row.get(f"{split}_recall_gap", np.nan)),
                "cutoff_score": float(row.get(f"{split}_cutoff_score", np.nan)),
                "segment_weighted_hit_rate": float(row.get(f"{split}_segment_weighted_hit_rate", np.nan)),
                "segment_weighted_majority_hit": float(row.get(f"{split}_segment_weighted_majority_hit", np.nan)),
                "true_down_share": float(row.get(f"{split}_true_down_share", np.nan)),
                "true_up_share": float(row.get(f"{split}_true_up_share", np.nan)),
                "pred_down_share": float(row.get(f"{split}_pred_down_share", np.nan)),
                "pred_up_share": float(row.get(f"{split}_pred_up_share", np.nan)),
                "pred_target_gap": float(row.get(f"{split}_pred_target_gap", np.nan)),
                "true_balance_gap_ud": float(row.get(f"{split}_true_balance_gap_ud", np.nan)),
                "pred_balance_gap_ud": float(row.get(f"{split}_pred_balance_gap_ud", np.nan)),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def _plot_confusion(ax: plt.Axes, cm: np.ndarray, labels: List[str], title: str) -> None:
    vmax = max(int(cm.max()), 1)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = int(cm[i, j])
            color = "white" if val > (vmax * 0.5) else "black"
            ax.text(j, i, str(val), va="center", ha="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_class_balance(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    idx = np.arange(len(metrics), dtype=np.float64)
    width = 0.18

    true_down = metrics["true_down_share"].to_numpy(dtype=np.float64)
    true_up = metrics["true_up_share"].to_numpy(dtype=np.float64)
    pred_down = metrics["pred_down_share"].to_numpy(dtype=np.float64)
    pred_up = metrics["pred_up_share"].to_numpy(dtype=np.float64)

    ax.bar(idx - 1.5 * width, true_down, width=width, label="true_down", color="#8da0cb")
    ax.bar(idx - 0.5 * width, true_up, width=width, label="true_up", color="#66c2a5")
    ax.bar(idx + 0.5 * width, pred_down, width=width, label="pred_down", color="#fc8d62")
    ax.bar(idx + 1.5 * width, pred_up, width=width, label="pred_up", color="#e78ac3")

    ax.axhline(0.5, color="#666666", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(idx)
    ax.set_xticklabels(metrics.index.tolist())
    ax.set_ylabel("Share")
    ax.set_title("Class Balance (Up/Down)")
    ax.legend(fontsize=8, loc="best")


def _plot_recalls(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    idx = np.arange(len(metrics), dtype=np.float64)
    width = 0.32

    rec_down = metrics["recall_down"].to_numpy(dtype=np.float64)
    rec_up = metrics["recall_up"].to_numpy(dtype=np.float64)

    ax.bar(idx - 0.5 * width, rec_down, width=width, label="recall_down", color="#377eb8")
    ax.bar(idx + 0.5 * width, rec_up, width=width, label="recall_up", color="#4daf4a")

    for i, split in enumerate(metrics.index.tolist()):
        rec_min = float(metrics.loc[split, "recall_min"])
        if np.isfinite(rec_min):
            ax.text(i, min(0.99, rec_min + 0.04), f"min={rec_min:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(idx)
    ax.set_xticklabels(metrics.index.tolist())
    ax.set_ylabel("Recall")
    ax.set_title("Recall Up/Down")
    ax.legend(fontsize=8, loc="best")


def render_postrun_report(
    results: pd.DataFrame,
    run_rank: int = 1,
    title: str | None = None,
    top_n_preview: int = 10,
) -> pd.DataFrame:
    """Render default post-run diagnostics and return split metrics table.

    Parameters
    ----------
    results:
        Trial/scout/grid results table from current notebook run.
    run_rank:
        1-based rank after default sorting. `1` means best row.
    title:
        Optional figure title.
    top_n_preview:
        Show top-N rows (key columns only) above the chart for quick scan.
    """
    if results.empty:
        raise ValueError("results is empty; run the training/trial cell first.")

    ranked = _sorted_results(results)
    rank_idx = int(max(1, run_rank)) - 1
    if rank_idx >= len(ranked):
        raise ValueError(f"run_rank={run_rank} is out of range for results length={len(ranked)}.")

    row = ranked.iloc[rank_idx]
    n_classes = _infer_n_classes(row)
    labels = _class_labels(n_classes)
    splits = [s for s in ["train", "val", "test"] if f"{s}_recall_down" in ranked.columns]
    metrics = _split_metrics_table(row=row, splits=splits)

    preview_cols = [
        c
        for c in [
            "run_id",
            "model",
            "model_type",
            "seed",
            "cutoff_kind",
            "cutoff_objective",
            "best_epoch",
            "best_margin",
            "best_val_score",
            "test_cutoff_score",
            "test_segment_weighted_hit_rate",
            "test_segment_weighted_majority_hit",
            "test_recall_down",
            "test_recall_up",
            "test_recall_min",
            "test_recall_gap",
            "test_acc",
            "test_mcc",
            "test_true_up_share",
            "test_pred_up_share",
            "test_true_balance_gap_ud",
            "test_pred_balance_gap_ud",
        ]
        if c in ranked.columns
    ]

    print(f"[postrun] selected rank={run_rank} / {len(ranked)}")
    print(
        "[postrun] test metrics: "
        f"acc={float(row.get('test_acc', np.nan)):.4f} "
        f"mcc={float(row.get('test_mcc', np.nan)):.4f} "
        f"recall_down={float(row.get('test_recall_down', np.nan)):.4f} "
        f"recall_up={float(row.get('test_recall_up', np.nan)):.4f} "
        f"balance_true_gap={float(row.get('test_true_balance_gap_ud', np.nan)):.4f} "
        f"balance_pred_gap={float(row.get('test_pred_balance_gap_ud', np.nan)):.4f}"
    )

    if preview_cols:
        top_preview = ranked[preview_cols].head(int(max(1, top_n_preview)))
        if display is not None:
            display(top_preview)
        else:
            print(top_preview.to_string(index=False))

    if display is not None:
        display(metrics.round(4))
    else:
        print(metrics.round(4).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    fig.suptitle(title or "Post-run diagnostics", fontsize=13, fontweight="bold")

    cm_test = _extract_confusion_matrix(row=row, split_name="test")
    if cm_test is not None:
        plot_labels = labels[: int(cm_test.shape[0])]
        _plot_confusion(axes[0], cm=cm_test, labels=plot_labels, title="Test Confusion Matrix")
    else:
        axes[0].axis("off")
        axes[0].text(
            0.5,
            0.5,
            "Confusion matrix\nis unavailable\nfor this result schema",
            ha="center",
            va="center",
            fontsize=10,
        )
        axes[0].set_title("Test Confusion Matrix")

    _plot_class_balance(axes[1], metrics=metrics)
    _plot_recalls(axes[2], metrics=metrics)

    plt.tight_layout()
    backend = str(plt.get_backend()).lower()
    if "agg" in backend:
        plt.close(fig)
    else:
        plt.show()

    return metrics
