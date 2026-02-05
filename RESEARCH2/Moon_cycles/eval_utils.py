"""
Core metric helpers for Moon-cycle research.

This module intentionally excludes plotting so each file stays short and focused.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute a full binary metric set.

    Why many metrics:
    - Accuracy alone can hide failures on one class.
    - Recall_MIN tells us how good the weaker class prediction is.
    - MCC is a strong overall metric for binary classification.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["DOWN", "UP"],
        output_dict=True,
        zero_division=0,
    )

    recall_down = float(report["DOWN"]["recall"])
    recall_up = float(report["UP"]["recall"])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_down": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision_up": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_down": recall_down,
        "recall_up": recall_up,
        "recall_min": float(min(recall_down, recall_up)),
        "recall_gap": float(abs(recall_down - recall_up)),
        "support": int(len(y_true)),
    }


def make_majority_baseline(y_train: np.ndarray, size: int) -> np.ndarray:
    """
    Build pre-training baseline: always predict train-majority class.

    This is the minimum bar that the trained model should beat.
    """
    y_train = np.asarray(y_train, dtype=np.int32)
    if y_train.size == 0:
        majority_class = 0
    else:
        down_count = int((y_train == 0).sum())
        up_count = int((y_train == 1).sum())
        majority_class = 1 if up_count > down_count else 0
    return np.full(size, majority_class, dtype=np.int32)


def make_coin_flip_baseline(size: int, p_up: float = 0.5, seed: int = 42) -> np.ndarray:
    """Build random baseline predictions (coin flip)."""
    rng = np.random.default_rng(seed)
    return (rng.random(size) < p_up).astype(np.int32)


def _erf_approx(x: float) -> float:
    """Fast approximation of erf(x), used only in fallback p-value path."""
    sign = -1.0 if x < 0 else 1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * np.exp(-(x * x))
    return sign * y


def _exact_binomial_one_sided_pvalue(n: int, k: int, p_null: float = 0.5) -> float:
    """
    Compute one-sided p-value P(X >= k) for Binomial(n, p_null).

    We try scipy first (stable numerics). If unavailable, we use
    a normal approximation fallback.
    """
    if n <= 0:
        return 1.0

    try:
        from scipy.stats import binomtest  # type: ignore

        return float(binomtest(k=k, n=n, p=p_null, alternative="greater").pvalue)
    except Exception:
        mean = n * p_null
        std = sqrt(max(n * p_null * (1.0 - p_null), 1e-12))
        z = (k - 0.5 - mean) / std
        return float(0.5 * (1.0 - _erf_approx(z / sqrt(2.0))))


def wilson_confidence_interval(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a binomial proportion (95% when z=1.96)."""
    if n <= 0:
        return (0.0, 1.0)

    denom = 1.0 + (z * z / n)
    center = (p_hat + (z * z / (2.0 * n))) / denom
    radius = (
        z
        * sqrt((p_hat * (1.0 - p_hat) / n) + ((z * z) / (4.0 * n * n)))
        / denom
    )
    lo = max(0.0, center - radius)
    hi = min(1.0, center + radius)
    return (float(lo), float(hi))


def compute_statistical_significance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    random_baseline: float = 0.5,
) -> Dict[str, float]:
    """
    Test whether model accuracy is significantly better than random guess.

    Output includes:
    - one-sided p-value vs random baseline
    - Wilson 95% confidence interval for accuracy
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    n = int(len(y_true))
    k = int((y_true == y_pred).sum())
    acc = k / n if n > 0 else 0.0

    ci_lo, ci_hi = wilson_confidence_interval(acc, n)
    p_value = _exact_binomial_one_sided_pvalue(n=n, k=k, p_null=random_baseline)

    return {
        "n": n,
        "correct": k,
        "accuracy": float(acc),
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "p_value_vs_random": float(p_value),
        "null_accuracy": float(random_baseline),
    }


def compute_rolling_metrics(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int = 90,
    min_periods: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling accuracy and per-class recalls.

    This helps to see if performance is stable or only good in small intervals.
    """
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).reset_index(drop=True),
            "y_true": np.asarray(y_true, dtype=np.int32),
            "y_pred": np.asarray(y_pred, dtype=np.int32),
        }
    )

    df["correct"] = (df["y_true"] == df["y_pred"]).astype(float)
    df["is_true_down"] = (df["y_true"] == 0).astype(float)
    df["is_true_up"] = (df["y_true"] == 1).astype(float)
    df["tp_down"] = ((df["y_true"] == 0) & (df["y_pred"] == 0)).astype(float)
    df["tp_up"] = ((df["y_true"] == 1) & (df["y_pred"] == 1)).astype(float)

    roll = df.rolling(window=window, min_periods=min_periods)

    out = pd.DataFrame({"date": df["date"]})
    out["rolling_accuracy"] = roll["correct"].mean()

    down_denom = roll["is_true_down"].sum().replace(0.0, np.nan)
    up_denom = roll["is_true_up"].sum().replace(0.0, np.nan)

    out["rolling_recall_down"] = roll["tp_down"].sum() / down_denom
    out["rolling_recall_up"] = roll["tp_up"].sum() / up_denom
    out["rolling_recall_min"] = out[["rolling_recall_down", "rolling_recall_up"]].min(axis=1)

    return out
