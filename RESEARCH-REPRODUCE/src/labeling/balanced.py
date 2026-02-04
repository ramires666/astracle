"""
Balanced labeling helpers for binary UP/DOWN targets.

This module implements "balanced_future_return" and "balanced_detrended":
- balanced_future_return: use future return in raw/log price space
- balanced_detrended: use future return on (price - centered Gaussian smooth)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

LabelMode = Literal["balanced_future_return", "balanced_detrended"]
PriceMode = Literal["raw", "log"]


def gaussian_kernel(window: int, std: float) -> np.ndarray:
    """
    Build a centered Gaussian kernel.
    """
    if window % 2 == 0:
        raise ValueError("gauss_window must be odd")
    x = np.arange(window) - window // 2
    w = np.exp(-(x ** 2) / (2 * (std ** 2)))
    w /= w.sum()
    return w


def gaussian_smooth_centered(series: pd.Series, window: int, std: float) -> pd.Series:
    """
    Centered Gaussian smoothing with full window only.
    Edges are NaN unless the caller fills them.
    """
    weights = gaussian_kernel(window, std)
    return series.rolling(window=window, center=True, min_periods=window).apply(
        lambda x: np.dot(x, weights), raw=True
    )


def _base_series(df: pd.DataFrame, price_col: str, price_mode: PriceMode) -> pd.Series:
    """
    Select base price series (raw or log).
    """
    if price_col not in df.columns:
        raise ValueError(f"price_col not found: {price_col}")
    base = df[price_col].astype(float)
    if price_mode == "log":
        return np.log(base)
    if price_mode == "raw":
        return base
    raise ValueError(f"Unknown price_mode={price_mode}")


def _balanced_label_from_future_return(
    future_ret: pd.Series,
    move_share_total: float,
) -> pd.Series:
    """
    Select top-N UP and DOWN by magnitude from future returns.
    """
    valid = future_ret.dropna()
    total_n = len(valid)
    if total_n == 0:
        raise ValueError("No valid future returns for labeling")

    per_side = max(1, int(total_n * move_share_total / 2))
    pos = valid[valid > 0]
    neg = valid[valid < 0]

    n_up = min(per_side, len(pos))
    n_down = min(per_side, len(neg))

    up_idx = pos.nlargest(n_up).index
    down_idx = neg.nsmallest(n_down).index

    target = pd.Series(index=future_ret.index, dtype="float64")
    target.loc[up_idx] = 1
    target.loc[down_idx] = 0
    return target


def build_balanced_labels(
    df_market: pd.DataFrame,
    horizon: int,
    target_move_share: float = 0.5,
    label_mode: LabelMode = "balanced_detrended",
    price_mode: PriceMode = "raw",
    gauss_window: int = 201,
    gauss_std: float = 50.0,
    price_col: str = "close",
    edge_fill: bool = True,
) -> pd.DataFrame:
    """
    Create balanced binary labels with UP/DOWN classes.
    """
    if not (0.0 < target_move_share <= 1.0):
        raise ValueError("target_move_share must be in (0, 1]")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    base = _base_series(df_market, price_col=price_col, price_mode=price_mode)

    if label_mode == "balanced_detrended":
        smooth = gaussian_smooth_centered(base, gauss_window, gauss_std)
        if edge_fill:
            smooth = smooth.bfill().ffill()
        base_detrended = base - smooth
        future_ret = base_detrended.shift(-horizon) - base_detrended
    elif label_mode == "balanced_future_return":
        future_ret = base.shift(-horizon) - base
    else:
        raise ValueError(f"Unknown label_mode={label_mode}")

    target = _balanced_label_from_future_return(
        future_ret=future_ret,
        move_share_total=float(target_move_share),
    )

    out = df_market.copy()
    out["target"] = target
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    out["target"] = out["target"].astype(int)
    return out
