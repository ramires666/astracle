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


# ====================================================================================
# TERNARY CLASSIFICATION LABELS (3 CLASSES: DOWN, SIDEWAYS, UP)
# ====================================================================================

def create_ternary_labels(
    df_market: pd.DataFrame,
    horizon: Optional[int] = None,
    gauss_window: Optional[int] = None,
    gauss_std: Optional[float] = None,
    price_mode: Literal["raw", "log"] = "raw",
    sideways_threshold: Optional[float] = None,
    balance_classes: bool = True,
    verbose: bool = True,
) -> tuple:
    """
    ═══════════════════════════════════════════════════════════════════════════════════
    CREATE TERNARY LABELS: DOWN=0, SIDEWAYS=1, UP=2
    ═══════════════════════════════════════════════════════════════════════════════════
    
    This function creates 3-class labels for price movement prediction:
    
    CLASSES:
    ────────────────────────────────────────────────────────────────────────────────
    - DOWN (0):     Price goes down significantly (detrended return < -threshold)
    - SIDEWAYS (1): Price stays relatively flat (-threshold <= return <= +threshold)
    - UP (2):       Price goes up significantly (detrended return > +threshold)
    
    ALGORITHM:
    ────────────────────────────────────────────────────────────────────────────────
    1. Compute detrended price using Gaussian smoothing (same as binary labels)
    2. Calculate future return at given horizon
    3. If balance_classes=True: 
       - Auto-tune threshold so all 3 classes have approximately equal samples
       - This uses binary search to find optimal threshold
    4. If sideways_threshold is specified and balance_classes=False:
       - Use fixed threshold (useful for reproducibility)
    5. Classify each sample based on detrended return vs threshold
    
    WHY BALANCE CLASSES?
    ────────────────────────────────────────────────────────────────────────────────
    Imbalanced classes lead to biased models that always predict the majority class.
    By balancing, we ensure the model learns patterns for all 3 market states.
    
    Args:
        df_market: Market DataFrame with 'date' and 'close' columns
        horizon: Prediction horizon in days (default from config)
        gauss_window: Gaussian window for detrending (default from config)
        gauss_std: Gaussian std for detrending (default from config)
        price_mode: 'raw' or 'log' price space
        sideways_threshold: Fixed threshold for SIDEWAYS class (optional)
        balance_classes: If True, auto-tune threshold for balanced classes
        verbose: Print statistics
    
    Returns:
        Tuple of (df_labels, threshold_used):
        - df_labels: DataFrame with date, close, and target columns
        - threshold_used: The threshold that was used (for reproducibility)
    ═══════════════════════════════════════════════════════════════════════════════════
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 1: Get defaults from config
    # ─────────────────────────────────────────────────────────────────────────────
    label_cfg = cfg.get_label_config()
    horizon = horizon or label_cfg["horizon"]
    gauss_window = gauss_window or label_cfg["gauss_window"]
    gauss_std = gauss_std or label_cfg["gauss_std"]
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 2: Prepare base price series
    # ─────────────────────────────────────────────────────────────────────────────
    if price_mode == "log":
        # Log prices: good for long time series with large price changes
        base_series = np.log(df_market["close"]).astype(float)
    else:
        # Raw prices: simpler, works well for stable price ranges
        base_series = df_market["close"].astype(float)
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 3: Calculate detrended future return
    # ─────────────────────────────────────────────────────────────────────────────
    # The detrending removes the trend component, so we focus on cyclical moves
    smooth = gaussian_smooth_centered(base_series, gauss_window, gauss_std)
    detrended = base_series - smooth
    future_ret = detrended.shift(-horizon) - detrended
    
    # Drop NaN values (last 'horizon' days have no future return)
    valid = future_ret.dropna()
    total_n = len(valid)
    
    if total_n == 0:
        raise ValueError("No valid future returns for labeling")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 4: Determine threshold for SIDEWAYS class
    # ─────────────────────────────────────────────────────────────────────────────
    if sideways_threshold is not None and not balance_classes:
        # Use fixed threshold
        threshold = sideways_threshold
    else:
        # Auto-tune threshold for balanced classes
        # Goal: find threshold T such that:
        #   count(return > T) ≈ count(|return| <= T) ≈ count(return < -T)
        #
        # We use binary search: start with absolute return percentiles
        
        abs_returns = valid.abs()
        
        # Binary search for optimal threshold
        # Target: each class has ~33% of samples
        target_per_class = total_n // 3
        
        # Search range: from 0 to max absolute return
        low, high = 0.0, abs_returns.max()
        best_threshold = abs_returns.median()  # Start with median
        best_imbalance = float("inf")
        
        # 20 iterations of binary search is enough for convergence
        for _ in range(20):
            mid = (low + high) / 2
            
            # Count samples in each class with this threshold
            n_up = (valid > mid).sum()
            n_down = (valid < -mid).sum()
            n_side = total_n - n_up - n_down
            
            # Imbalance metric: max deviation from target
            imbalance = max(
                abs(n_up - target_per_class),
                abs(n_down - target_per_class),
                abs(n_side - target_per_class)
            )
            
            if imbalance < best_imbalance:
                best_imbalance = imbalance
                best_threshold = mid
            
            # Adjust search range
            # If too many SIDEWAYS, increase threshold; if too few, decrease
            if n_side > target_per_class:
                high = mid
            else:
                low = mid
        
        threshold = best_threshold
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 5: Assign labels based on threshold
    # ─────────────────────────────────────────────────────────────────────────────
    df_labels = df_market[["date", "close"]].copy()
    df_labels["detrended_return"] = future_ret
    
    # Create target column with 3 classes
    df_labels["target"] = np.nan
    df_labels.loc[valid.index[valid > threshold], "target"] = 2   # UP
    df_labels.loc[valid.index[valid < -threshold], "target"] = 0  # DOWN
    # Everything else is SIDEWAYS
    mask_sideways = valid.index[(valid >= -threshold) & (valid <= threshold)]
    df_labels.loc[mask_sideways, "target"] = 1  # SIDEWAYS
    
    # Keep only labeled rows
    df_labels = df_labels.dropna(subset=["target"]).reset_index(drop=True)
    df_labels["target"] = df_labels["target"].astype(int)
    
    # Drop helper column
    df_labels = df_labels.drop(columns=["detrended_return"])
    
    # ─────────────────────────────────────────────────────────────────────────────
    # STEP 6: Print statistics
    # ─────────────────────────────────────────────────────────────────────────────
    n_up = (df_labels["target"] == 2).sum()
    n_side = (df_labels["target"] == 1).sum()
    n_down = (df_labels["target"] == 0).sum()
    total = len(df_labels)
    
    if verbose:
        print(f"═══════════════════════════════════════════════════════════════════")
        print(f"TERNARY LABELS CREATED: {total} samples")
        print(f"───────────────────────────────────────────────────────────────────")
        print(f"  UP (2):       {n_up:5d} ({100*n_up/total:5.1f}%)")
        print(f"  SIDEWAYS (1): {n_side:5d} ({100*n_side/total:5.1f}%)")
        print(f"  DOWN (0):     {n_down:5d} ({100*n_down/total:5.1f}%)")
        print(f"───────────────────────────────────────────────────────────────────")
        print(f"  Threshold used: {threshold:.6f}")
        print(f"  Date range: {df_labels['date'].min().date()} → {df_labels['date'].max().date()}")
        print(f"═══════════════════════════════════════════════════════════════════")
    
    return df_labels, threshold


def get_ternary_label_stats(df_labels: pd.DataFrame) -> dict:
    """
    Get statistics about ternary labels.
    
    Args:
        df_labels: DataFrame with 'target' column (0=DOWN, 1=SIDEWAYS, 2=UP)
    
    Returns:
        Dictionary with counts and percentages for each class
    """
    total = len(df_labels)
    n_up = (df_labels["target"] == 2).sum()
    n_side = (df_labels["target"] == 1).sum()
    n_down = (df_labels["target"] == 0).sum()
    
    return {
        "total": total,
        "up": n_up,
        "sideways": n_side,
        "down": n_down,
        "up_pct": 100 * n_up / total if total > 0 else 0,
        "sideways_pct": 100 * n_side / total if total > 0 else 0,
        "down_pct": 100 * n_down / total if total > 0 else 0,
        "date_min": df_labels["date"].min(),
        "date_max": df_labels["date"].max(),
    }
