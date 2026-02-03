"""
Labeling module for RESEARCH pipeline.
Creates balanced UP/DOWN labels based on future returns.
Ensures latest data from DB is used.
"""
import numpy as np
import pandas as pd
from typing import Optional, Literal

from .config import cfg


def gaussian_kernel(window: int, std: float) -> np.ndarray:
    """
    Create a Gaussian kernel for smoothing.
    
    Args:
        window: Window size (must be odd)
        std: Standard deviation
    
    Returns:
        Normalized Gaussian weights
    """
    if window % 2 == 0:
        window += 1  # Ensure odd
    
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / std) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_smooth_centered(
    series: pd.Series, 
    window: int, 
    std: float
) -> pd.Series:
    """
    Apply centered Gaussian smoothing (no lag).
    
    Uses edge padding to avoid NaN at boundaries.
    
    Args:
        series: Input price series
        window: Window size
        std: Standard deviation
    
    Returns:
        Smoothed series
    """
    if series.empty:
        return series.copy()
    
    kernel = gaussian_kernel(window, std)
    pad = len(kernel) // 2
    
    arr = series.values.astype(float)
    left = np.full(pad, arr[0])
    right = np.full(pad, arr[-1])
    padded = np.concatenate([left, arr, right])
    
    smooth = np.convolve(padded, kernel, mode='valid')
    return pd.Series(smooth, index=series.index)


def create_balanced_labels(
    df_market: pd.DataFrame,
    horizon: Optional[int] = None,
    move_share: Optional[float] = None,
    gauss_window: Optional[int] = None,
    gauss_std: Optional[float] = None,
    price_mode: Literal["raw", "log"] = "raw",
    label_mode: Literal["balanced_detrended", "balanced_future_return"] = "balanced_detrended",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Create balanced binary labels (UP/DOWN) based on future returns.
    
    Uses detrended price (price - gaussian smooth) for label creation.
    Selects top N largest UP moves and top N largest DOWN moves.
    
    Args:
        df_market: Market DataFrame with 'date' and 'close' columns
        horizon: Prediction horizon in days (default from config)
        move_share: Total share of samples to keep (default from config)
        gauss_window: Gaussian window for detrending (default from config)
        gauss_std: Gaussian std for detrending (default from config)
        price_mode: 'raw' or 'log' price space
        label_mode: 'balanced_detrended' or 'balanced_future_return'
    
    Returns:
        DataFrame with date, close, and target columns
    """
    # Get defaults from config
    label_cfg = cfg.get_label_config()
    horizon = horizon or label_cfg["horizon"]
    move_share = move_share or label_cfg["target_move_share"]
    gauss_window = gauss_window or label_cfg["gauss_window"]
    gauss_std = gauss_std or label_cfg["gauss_std"]
    
    # Prepare base price series
    if price_mode == "log":
        base_series = np.log(df_market["close"]).astype(float)
    else:
        base_series = df_market["close"].astype(float)
    
    # Calculate future return (detrended if specified)
    if label_mode == "balanced_detrended":
        smooth = gaussian_smooth_centered(base_series, gauss_window, gauss_std)
        base = base_series - smooth
        future_ret = base.shift(-horizon) - base
    else:
        future_ret = base_series.shift(-horizon) - base_series
    
    # Select balanced labels
    valid = future_ret.dropna()
    total_n = len(valid)
    
    if total_n == 0:
        raise ValueError("No valid future returns for labeling")
    
    # Choose top-N UP and top-N DOWN
    per_side = max(1, int(total_n * move_share / 2))
    
    pos = valid[valid > 0]
    neg = valid[valid < 0]
    
    n_up = min(per_side, len(pos))
    n_down = min(per_side, len(neg))
    
    up_idx = pos.nlargest(n_up).index
    down_idx = neg.nsmallest(n_down).index  # Most negative
    
    # Create labeled DataFrame
    df_labels = df_market[["date", "close"]].copy()
    df_labels["target"] = np.nan
    df_labels.loc[up_idx, "target"] = 1
    df_labels.loc[down_idx, "target"] = 0
    
    # Keep only labeled rows
    df_labels = df_labels.dropna(subset=["target"]).reset_index(drop=True)
    df_labels["target"] = df_labels["target"].astype(int)
    
    # Stats
    n_up_final = (df_labels["target"] == 1).sum()
    n_down_final = (df_labels["target"] == 0).sum()
    
    if verbose:
        print(f"Labels created: {len(df_labels)} samples")
        print(f"  UP: {n_up_final} ({100*n_up_final/len(df_labels):.1f}%)")
        print(f"  DOWN: {n_down_final} ({100*n_down_final/len(df_labels):.1f}%)")
        print(f"  Date range: {df_labels['date'].min().date()} -> {df_labels['date'].max().date()}")
    
    return df_labels


def get_label_stats(df_labels: pd.DataFrame) -> dict:
    """Get statistics about labels."""
    total = len(df_labels)
    n_up = (df_labels["target"] == 1).sum()
    n_down = (df_labels["target"] == 0).sum()
    
    return {
        "total": total,
        "up": n_up,
        "down": n_down,
        "up_pct": 100 * n_up / total if total > 0 else 0,
        "down_pct": 100 * n_down / total if total > 0 else 0,
        "date_min": df_labels["date"].min(),
        "date_max": df_labels["date"].max(),
    }
