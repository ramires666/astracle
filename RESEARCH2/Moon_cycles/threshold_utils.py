"""
Threshold-related helpers for Moon-cycle experiments.

This module contains small reusable utilities so search_utils.py stays focused
on protocol orchestration and keeps file size under project limits.
"""

from __future__ import annotations

import numpy as np

from .eval_utils import compute_binary_metrics


def predict_proba_up_safe(model, X: np.ndarray) -> np.ndarray:
    """
    Predict UP probability with safe fallback for constant predictors.

    In rare folds, training may have one class only. In that case model wrapper
    switches to constant class prediction; here we convert it to deterministic
    probabilities (1.0 for UP constant, 0.0 for DOWN constant).
    """
    if getattr(model, "constant_class", None) is not None:
        const_class = int(model.constant_class)
        return np.full(X.shape[0], 1.0 if const_class == 1 else 0.0, dtype=float)

    X_scaled = model.scaler.transform(X)
    return model.model.predict_proba(X_scaled)[:, 1]


def tune_threshold_with_balance(
    y_val: np.ndarray,
    proba_up: np.ndarray,
    gap_penalty: float,
    prior_penalty: float,
) -> tuple[float, float]:
    """
    Select threshold with explicit balance-aware objective.

    Objective to maximize:
        score = recall_min - gap_penalty * recall_gap - prior_penalty * prior_gap

    Terms:
    - recall_min: quality of weaker class
    - recall_gap: class recall imbalance
    - prior_gap: mismatch between predicted UP share and true UP share

    This helps avoid one-class collapse on validation.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    y_val = np.asarray(y_val, dtype=np.int32)
    true_up_share = float((y_val == 1).mean()) if len(y_val) > 0 else 0.5

    best_t = 0.5
    best_score = -1e9

    for t in thresholds:
        pred = (proba_up >= t).astype(np.int32)
        m = compute_binary_metrics(y_true=y_val, y_pred=pred)
        pred_up_share = float((pred == 1).mean()) if len(pred) > 0 else 0.5
        prior_gap = abs(pred_up_share - true_up_share)

        score = (
            float(m["recall_min"])
            - float(gap_penalty) * float(m["recall_gap"])
            - float(prior_penalty) * float(prior_gap)
        )

        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score)
