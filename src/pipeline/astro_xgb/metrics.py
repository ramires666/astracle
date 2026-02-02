"""
Metrics helpers for astro_xgb pipeline.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    classification_report,
)


def majority_baseline_pred(y_true: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """
    Predict the majority class.
    """
    counts = [int((y_true == lbl).sum()) for lbl in labels]
    majority_label = labels[int(np.argmax(counts))]
    return np.full_like(y_true, majority_label)


def prev_label_baseline_pred(y_true: np.ndarray, fallback_label: int = 0) -> np.ndarray:
    """
    Predict previous label (naive time baseline).
    """
    if len(y_true) == 0:
        return np.array([], dtype=y_true.dtype)
    pred = np.roll(y_true, 1)
    pred[0] = fallback_label
    return pred


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> Dict[str, float]:
    """
    Compute basic classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "acc": float(acc),
        "bal_acc": float(bal_acc),
        "mcc": float(mcc),
        "f1_macro": float(f1_macro),
        "summary": float(0.5 * (bal_acc + f1_macro)),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
    n_boot: int = 200,
    seed: int = 42,
) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Bootstrap CI for metrics.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return None
    samples: Dict[str, List[float]] = {"acc": [], "bal_acc": [], "mcc": [], "f1_macro": [], "summary": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        m = calc_metrics(y_true[idx], y_pred[idx], labels)
        for k, v in m.items():
            samples[k].append(v)
    out: Dict[str, Tuple[float, float]] = {}
    for k, vals in samples.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[k] = (float(lo), float(hi))
    return out


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
    names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """
    Return sklearn classification report as dict.
    """
    return classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        target_names=list(names),
        output_dict=True,
        zero_division=0,
    )


def confusion_matrix_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int],
) -> np.ndarray:
    """
    Confusion matrix helper.
    """
    return confusion_matrix(y_true, y_pred, labels=list(labels))
