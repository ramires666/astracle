"""Numba-accelerated cutoff utilities and directional metric helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numba import njit, prange
from sklearn.metrics import accuracy_score, matthews_corrcoef, recall_score


@dataclass(frozen=True)
class MarginSearchResult:
    """Best margin and full scan arrays for ternary prediction."""

    best_margin: float
    best_score: float
    best_idx: int
    scores: np.ndarray
    recall_min: np.ndarray
    recall_gap: np.ndarray
    accuracy: np.ndarray
    pred_up_share: np.ndarray


@dataclass(frozen=True)
class ThresholdSearchResult:
    """Best threshold and full scan arrays for binary prediction."""

    best_threshold: float
    best_score: float
    best_idx: int
    scores: np.ndarray
    recall_min: np.ndarray
    recall_gap: np.ndarray
    accuracy: np.ndarray
    pred_up_share: np.ndarray


@njit(parallel=True, cache=True)
def _scan_margin_numba(
    p_up: np.ndarray,
    p_down: np.ndarray,
    y_true: np.ndarray,
    margins: np.ndarray,
    gap_penalty: float,
    prior_penalty: float,
    true_up_share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_m = margins.shape[0]
    n = y_true.shape[0]

    scores = np.empty(n_m, dtype=np.float64)
    recall_min = np.empty(n_m, dtype=np.float64)
    recall_gap = np.empty(n_m, dtype=np.float64)
    accuracy = np.empty(n_m, dtype=np.float64)
    pred_up_share = np.empty(n_m, dtype=np.float64)

    for i in prange(n_m):
        m = margins[i]

        true_down = 0
        true_up = 0
        hit_down = 0
        hit_up = 0
        correct_all = 0
        pred_up = 0

        for j in range(n):
            delta = p_up[j] - p_down[j]
            pred = 1
            if delta > m:
                pred = 2
            elif delta < -m:
                pred = 0

            yt = y_true[j]
            if pred == yt:
                correct_all += 1

            if pred == 2:
                pred_up += 1

            if yt == 0:
                true_down += 1
                if pred == 0:
                    hit_down += 1
            elif yt == 2:
                true_up += 1
                if pred == 2:
                    hit_up += 1

        rec_down = hit_down / max(true_down, 1)
        rec_up = hit_up / max(true_up, 1)
        rec_min = rec_down if rec_down < rec_up else rec_up
        gap = rec_down - rec_up
        if gap < 0.0:
            gap = -gap

        pred_share = pred_up / max(n, 1)
        prior_gap = pred_share - true_up_share
        if prior_gap < 0.0:
            prior_gap = -prior_gap

        score = rec_min - gap_penalty * gap - prior_penalty * prior_gap

        scores[i] = score
        recall_min[i] = rec_min
        recall_gap[i] = gap
        accuracy[i] = correct_all / max(n, 1)
        pred_up_share[i] = pred_share

    return scores, recall_min, recall_gap, accuracy, pred_up_share


@njit(parallel=True, cache=True)
def _scan_threshold_binary_numba(
    p_up: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    gap_penalty: float,
    prior_penalty: float,
    true_up_share: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_t = thresholds.shape[0]
    n = y_true.shape[0]

    scores = np.empty(n_t, dtype=np.float64)
    recall_min = np.empty(n_t, dtype=np.float64)
    recall_gap = np.empty(n_t, dtype=np.float64)
    accuracy = np.empty(n_t, dtype=np.float64)
    pred_up_share = np.empty(n_t, dtype=np.float64)

    for i in prange(n_t):
        thr = thresholds[i]

        true_down = 0
        true_up = 0
        hit_down = 0
        hit_up = 0
        correct_all = 0
        pred_up = 0

        for j in range(n):
            pred = 1 if p_up[j] >= thr else 0
            yt = y_true[j]

            if pred == yt:
                correct_all += 1
            if pred == 1:
                pred_up += 1

            if yt == 0:
                true_down += 1
                if pred == 0:
                    hit_down += 1
            else:
                true_up += 1
                if pred == 1:
                    hit_up += 1

        rec_down = hit_down / max(true_down, 1)
        rec_up = hit_up / max(true_up, 1)
        rec_min = rec_down if rec_down < rec_up else rec_up
        gap = rec_down - rec_up
        if gap < 0.0:
            gap = -gap

        pred_share = pred_up / max(n, 1)
        prior_gap = pred_share - true_up_share
        if prior_gap < 0.0:
            prior_gap = -prior_gap

        score = rec_min - gap_penalty * gap - prior_penalty * prior_gap

        scores[i] = score
        recall_min[i] = rec_min
        recall_gap[i] = gap
        accuracy[i] = correct_all / max(n, 1)
        pred_up_share[i] = pred_share

    return scores, recall_min, recall_gap, accuracy, pred_up_share


def search_best_margin(
    probs: np.ndarray,
    y_true: np.ndarray,
    margins: np.ndarray,
    gap_penalty: float,
    prior_penalty: float,
) -> MarginSearchResult:
    """Find best up/down margin over ternary probabilities."""
    if probs.ndim != 2 or probs.shape[1] < 3:
        raise ValueError("probs must be 2D with at least 3 class columns")

    p_up = np.ascontiguousarray(probs[:, 2].astype(np.float64))
    p_down = np.ascontiguousarray(probs[:, 0].astype(np.float64))
    y = np.ascontiguousarray(y_true.astype(np.int64))
    margin_grid = np.ascontiguousarray(margins.astype(np.float64))

    true_up_share = float((y == 2).mean()) if len(y) > 0 else 0.5

    scores, rec_min, rec_gap, acc, up_share = _scan_margin_numba(
        p_up=p_up,
        p_down=p_down,
        y_true=y,
        margins=margin_grid,
        gap_penalty=float(gap_penalty),
        prior_penalty=float(prior_penalty),
        true_up_share=float(true_up_share),
    )

    best_idx = int(np.argmax(scores))
    return MarginSearchResult(
        best_margin=float(margin_grid[best_idx]),
        best_score=float(scores[best_idx]),
        best_idx=best_idx,
        scores=scores,
        recall_min=rec_min,
        recall_gap=rec_gap,
        accuracy=acc,
        pred_up_share=up_share,
    )


def search_best_threshold_binary(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
    gap_penalty: float,
    prior_penalty: float,
) -> ThresholdSearchResult:
    """Find best class-1 threshold for binary probabilities."""
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("probs must be 2D with at least 2 class columns")

    p_up = np.ascontiguousarray(probs[:, 1].astype(np.float64))
    y = np.ascontiguousarray(y_true.astype(np.int64))
    thr_grid = np.ascontiguousarray(thresholds.astype(np.float64))

    true_up_share = float((y == 1).mean()) if len(y) > 0 else 0.5

    scores, rec_min, rec_gap, acc, up_share = _scan_threshold_binary_numba(
        p_up=p_up,
        y_true=y,
        thresholds=thr_grid,
        gap_penalty=float(gap_penalty),
        prior_penalty=float(prior_penalty),
        true_up_share=float(true_up_share),
    )

    best_idx = int(np.argmax(scores))
    return ThresholdSearchResult(
        best_threshold=float(thr_grid[best_idx]),
        best_score=float(scores[best_idx]),
        best_idx=best_idx,
        scores=scores,
        recall_min=rec_min,
        recall_gap=rec_gap,
        accuracy=acc,
        pred_up_share=up_share,
    )


def predict_with_margin(probs: np.ndarray, margin: float) -> np.ndarray:
    """Vectorized ternary decision from margin over (up - down)."""
    delta = probs[:, 2] - probs[:, 0]
    pred = np.ones(delta.shape[0], dtype=np.int64)
    pred[delta > float(margin)] = 2
    pred[delta < -float(margin)] = 0
    return pred


def predict_with_threshold_binary(probs: np.ndarray, threshold: float) -> np.ndarray:
    """Vectorized binary decision from class-1 threshold."""
    return (probs[:, 1] >= float(threshold)).astype(np.int64)


def summarize_directional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    """Compute recall-gap family metrics for binary and ternary targets."""
    y_true_i = y_true.astype(np.int64)
    y_pred_i = y_pred.astype(np.int64)
    n_cls = int(max(2, n_classes))

    if n_cls >= 3:
        rec_all = recall_score(
            y_true=y_true_i,
            y_pred=y_pred_i,
            labels=[0, 1, 2],
            average=None,
            zero_division=0,
        )
        rec_down = float(rec_all[0])
        rec_side = float(rec_all[1])
        rec_up = float(rec_all[2])
        up_label = 2
        side_label = 1
    else:
        rec_all = recall_score(
            y_true=y_true_i,
            y_pred=y_pred_i,
            labels=[0, 1],
            average=None,
            zero_division=0,
        )
        rec_down = float(rec_all[0])
        rec_side = float("nan")
        rec_up = float(rec_all[1])
        up_label = 1
        side_label = -1

    mcc = float(matthews_corrcoef(y_true_i, y_pred_i))
    if not np.isfinite(mcc):
        mcc = 0.0

    acc = float(accuracy_score(y_true_i, y_pred_i))
    rec_min = float(min(rec_down, rec_up))
    rec_gap = float(abs(rec_down - rec_up))

    pred_up_share = float((y_pred_i == up_label).mean()) if len(y_pred_i) else 0.0
    true_up_share = float((y_true_i == up_label).mean()) if len(y_true_i) else 0.0
    pred_down_share = float((y_pred_i == 0).mean()) if len(y_pred_i) else 0.0
    true_down_share = float((y_true_i == 0).mean()) if len(y_true_i) else 0.0
    if side_label >= 0:
        pred_side_share = float((y_pred_i == side_label).mean()) if len(y_pred_i) else 0.0
        true_side_share = float((y_true_i == side_label).mean()) if len(y_true_i) else 0.0
    else:
        pred_side_share = 0.0
        true_side_share = 0.0

    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for yt, yp in zip(y_true_i, y_pred_i):
        if 0 <= yt < n_cls and 0 <= yp < n_cls:
            cm[int(yt), int(yp)] += 1

    out = {
        "acc": acc,
        "mcc": mcc,
        "recall_down": rec_down,
        "recall_sideways": rec_side,
        "recall_up": rec_up,
        "recall_min": rec_min,
        "recall_gap": rec_gap,
        "pred_up_share": pred_up_share,
        "true_up_share": true_up_share,
        "pred_down_share": pred_down_share,
        "true_down_share": true_down_share,
        "pred_sideways_share": pred_side_share,
        "true_sideways_share": true_side_share,
        "true_balance_gap_ud": float(abs(true_up_share - true_down_share)),
        "pred_balance_gap_ud": float(abs(pred_up_share - pred_down_share)),
        "pred_target_gap": float(abs(pred_up_share - true_up_share)),
    }

    for i in range(n_cls):
        true_count = int(cm[i, :].sum())
        pred_count = int(cm[:, i].sum())
        out[f"support_true_{i}"] = float(true_count)
        out[f"support_pred_{i}"] = float(pred_count)
        out[f"share_true_{i}"] = float(true_count / len(y_true_i)) if len(y_true_i) else 0.0
        out[f"share_pred_{i}"] = float(pred_count / len(y_pred_i)) if len(y_pred_i) else 0.0
        for j in range(n_cls):
            out[f"cm_{i}{j}"] = float(cm[i, j])

    return out
