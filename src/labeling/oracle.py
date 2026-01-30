"""
Oracle label generation using Gaussian smoothing.

IMPORTANT: the smoothed line uses future points,
so it MUST NOT be used as a feature.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d


def classify_slope(slope: np.ndarray, threshold: float) -> np.ndarray:
    """
    Classify slope into labels:
    0 = DOWN, 1 = SIDEWAYS, 2 = UP
    """
    labels = np.ones(len(slope), dtype=np.int32)
    labels[slope > threshold] = 2
    labels[slope < -threshold] = 0
    return labels


def collapse_sideways_to_trend(
    labels: np.ndarray,
    fallback: str = "up",
) -> np.ndarray:
    """
    Convert 3-class labels (0=down, 1=sideways, 2=up) to binary trend labels.
    Sideways segments are carried forward from the last known direction.
    """
    if labels.size == 0:
        return labels.astype(np.int32)

    # Map to direction: -1 (down), 0 (sideways), +1 (up)
    direction = np.zeros(len(labels), dtype=np.int8)
    direction[labels == 2] = 1
    direction[labels == 0] = -1

    series = pd.Series(direction, dtype="float64").replace(0, np.nan)
    series = series.ffill().bfill()

    if series.isna().all():
        use_down = str(fallback).strip().lower() in {"down", "bear", "short", "-1", "0"}
        series[:] = -1 if use_down else 1
    else:
        if series.isna().any():
            use_down = str(fallback).strip().lower() in {"down", "bear", "short", "-1", "0"}
            series = series.fillna(-1 if use_down else 1)

    binary = (series.values > 0).astype(np.int32)
    return binary


def _prepare_smoothed_series(
    df: pd.DataFrame,
    sigma: int,
    price_col: str,
    price_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare a smoothed series.

    Returns:
      - smoothed: smoothed series in working scale (log or raw)
      - smoothed_close: smoothed price for visualization
    """
    if price_col not in df.columns:
        raise ValueError(f"Column {price_col} not found")

    mode = (price_mode or "log").strip().lower()
    if mode in {"log", "log_price", "ln", "logarithm"}:
        # 1) Log-transform price
        base_series = np.log(df[price_col].values.astype(np.float64))
        # 2) Smooth (centered filter)
        smoothed = gaussian_filter1d(base_series, sigma=sigma, mode="nearest")
        # 3) Visualization line - back to price scale
        smoothed_close = np.exp(smoothed)
    elif mode in {"raw", "price", "linear"}:
        # 1) Use raw price
        base_series = df[price_col].values.astype(np.float64)
        # 2) Smooth (centered filter)
        smoothed = gaussian_filter1d(base_series, sigma=sigma, mode="nearest")
        # 3) Visualization line - the smoothed price itself
        smoothed_close = smoothed
    else:
        raise ValueError("price_mode must be 'log' or 'raw'")

    return smoothed, smoothed_close


def estimate_threshold_for_move_balance(
    df: pd.DataFrame,
    sigma: int,
    price_col: str = "close",
    price_mode: str = "log",
    target_move_share: float = 0.5,
    min_threshold: float = 0.0,
    max_threshold: Optional[float] = None,
) -> float:
    """
    Auto-pick threshold so MOVE share is close to target_move_share.

    MOVE is defined as |slope| > threshold.
    NO_MOVE (SIDEWAYS) is defined as |slope| <= threshold.
    """
    if not (0.0 < target_move_share < 1.0):
        raise ValueError("target_move_share must be in (0, 1)")

    smoothed, _ = _prepare_smoothed_series(
        df=df,
        sigma=sigma,
        price_col=price_col,
        price_mode=price_mode,
    )
    smooth_slope = np.diff(smoothed, prepend=smoothed[0])
    abs_slope = np.abs(smooth_slope)
    abs_slope = abs_slope[np.isfinite(abs_slope)]
    if len(abs_slope) == 0:
        raise ValueError("Failed to compute slope for auto threshold")

    # Want MOVE approx target_move_share -> abs_slope > threshold
    # Therefore threshold = quantile (1 - target_move_share).
    q = 1.0 - float(target_move_share)
    threshold = float(np.quantile(abs_slope, q))

    if min_threshold is not None:
        threshold = max(threshold, float(min_threshold))
    if max_threshold is not None:
        threshold = min(threshold, float(max_threshold))

    return float(threshold)


def create_oracle_labels(
    df: pd.DataFrame,
    sigma: int = 5,
    threshold: float = 0.001,
    price_col: str = "close",
    price_mode: str = "log",
    binary_trend: bool = False,
    binary_fallback: str = "up",
) -> pd.DataFrame:
    """
    Build oracle labels from the smoothed price.

    price_mode:
      - "log" (default) - log price
      - "raw" - raw price without log
    """
    smoothed, smoothed_close = _prepare_smoothed_series(
        df=df,
        sigma=sigma,
        price_col=price_col,
        price_mode=price_mode,
    )

    result = df.copy()
    result["smoothed_close"] = smoothed_close

    # Compute slope
    smooth_slope = np.diff(smoothed, prepend=smoothed[0])

    # Classify
    labels = classify_slope(smooth_slope, threshold)

    # Save for analysis (not for features!)
    result["smooth_slope"] = smooth_slope
    if binary_trend:
        result["target_3"] = labels
        result["target"] = collapse_sideways_to_trend(labels, fallback=binary_fallback)
    else:
        result["target"] = labels

    return result


def analyze_label_distribution(
    df: pd.DataFrame,
    sigma_range: Tuple[int, int],
    threshold_range: Tuple[float, float],
    n_steps: int = 5,
    price_mode: str = "log",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Search sigma/threshold values for class balance.
    """
    # Important: linspace + dtype=int may produce duplicates (e.g. [2, 2, 3, 4, 5]),
    # which breaks pivot in the notebook. Remove duplicates.
    sigmas = np.unique(np.linspace(sigma_range[0], sigma_range[1], n_steps, dtype=int))
    thresholds = np.unique(np.linspace(threshold_range[0], threshold_range[1], n_steps))

    rows = []
    for sigma in sigmas:
        for thr in thresholds:
            labeled = create_oracle_labels(
                df,
                sigma=int(sigma),
                threshold=float(thr),
                price_col=price_col,
                price_mode=price_mode,
            )
            counts = labeled["target"].value_counts(normalize=True)
            rows.append({
                "sigma": int(sigma),
                "threshold": float(thr),
                "down_pct": counts.get(0, 0) * 100,
                "sideways_pct": counts.get(1, 0) * 100,
                "up_pct": counts.get(2, 0) * 100,
                "imbalance": abs(counts.get(0, 0) - counts.get(2, 0)) * 100,
            })

    return pd.DataFrame(rows).sort_values("imbalance")
