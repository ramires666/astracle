"""
Statistical significance helpers for model evaluation.

This module centralizes hypothesis-testing utilities so other evaluation modules
can reuse one consistent implementation.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict

import numpy as np


# =============================================================================
# INTERNAL MATH HELPERS
# =============================================================================

def _erf_approx(x: float) -> float:
    """
    Fast approximation of the error function erf(x).

    Why we keep this helper:
    - Exact binomial p-value is preferred and used when scipy is available.
    - If scipy is unavailable, we use a normal approximation fallback.
    - Normal CDF requires erf, so this helper keeps fallback path self-contained.
    """
    sign = -1.0 if x < 0 else 1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    y = 1.0 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * np.exp(-(x * x))
    return sign * y


def _exact_binomial_one_sided_pvalue(n: int, k: int, p_null: float = 0.5) -> float:
    """
    Compute one-sided p-value P(X >= k) for Binomial(n, p_null).

    Interpretation for classification:
    - n = number of predictions
    - k = number of correct predictions
    - p_null = random baseline accuracy (0.5 for balanced binary)
    - p-value = probability that random guessing reaches at least k correct

    Small p-value means observed accuracy is unlikely to be random luck.
    """
    if n <= 0:
        return 1.0

    try:
        from scipy.stats import binomtest  # type: ignore

        return float(binomtest(k=k, n=n, p=p_null, alternative="greater").pvalue)
    except Exception:
        # Fallback: normal approximation with continuity correction.
        mean = n * p_null
        std = sqrt(max(n * p_null * (1.0 - p_null), 1e-12))
        z = (k - 0.5 - mean) / std
        return float(0.5 * (1.0 - _erf_approx(z / sqrt(2.0))))


def _wilson_confidence_interval(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Wilson confidence interval for a binomial proportion.

    Wilson interval is usually more reliable than naive normal intervals,
    especially for finite sample sizes and proportions near edges.
    """
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


# =============================================================================
# PUBLIC FUNCTION
# =============================================================================

def compute_accuracy_significance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    random_baseline: float = 0.5,
) -> Dict[str, float]:
    """
    Compute statistical evidence that model accuracy is above random baseline.

    Returns a dictionary with:
    - `p_value_vs_random`: one-sided binomial p-value
    - `accuracy_ci95_low`, `accuracy_ci95_high`: Wilson 95% interval
    - bookkeeping fields (`n_eval`, `n_correct`, `random_baseline`)
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    n = int(len(y_true))
    k = int((y_true == y_pred).sum())
    acc = (k / n) if n > 0 else 0.0

    ci_lo, ci_hi = _wilson_confidence_interval(acc, n)
    p_val = _exact_binomial_one_sided_pvalue(n=n, k=k, p_null=random_baseline)

    return {
        "n_eval": n,
        "n_correct": k,
        "random_baseline": float(random_baseline),
        "accuracy_ci95_low": float(ci_lo),
        "accuracy_ci95_high": float(ci_hi),
        "p_value_vs_random": float(p_val),
    }
