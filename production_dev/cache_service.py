"""
Prediction Cache Service

Manages cached predictions stored in parquet files for fast loading.
- Backtest: Past 6 months of predictions with actual results
- Forecast: Future 1 year of predictions

This avoids recalculating expensive ephemeris calculations on every request.
"""

import os
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

CACHE_DIR = PROJECT_ROOT / "data" / "prediction_cache"

CACHE_FILES = {
    "backtest": "backtest_predictions.parquet",  # Past predictions with actual
    "forecast": "forecast_predictions.parquet",  # Future predictions
}

# Time ranges
BACKTEST_DAYS = 180   # ~6 months of past predictions
FORECAST_DAYS = 365   # 1 year of future predictions


# =============================================================================
# CACHE OPERATIONS
# =============================================================================

def ensure_cache_dir() -> Path:
    """Create cache directory if it doesn't exist."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def get_cache_path(cache_type: str) -> Path:
    """Get full path for a cache file."""
    return CACHE_DIR / CACHE_FILES.get(cache_type, f"{cache_type}.parquet")


def load_cached_predictions(cache_type: str) -> Optional[pd.DataFrame]:
    """
    Load predictions from parquet cache.
    
    Args:
        cache_type: "backtest" or "forecast"
        
    Returns:
        DataFrame with predictions or None if cache doesn't exist
    """
    cache_path = get_cache_path(cache_type)
    
    if not cache_path.exists():
        print(f"Cache not found: {cache_path}")
        return None
    
    try:
        df = pd.read_parquet(cache_path)
        print(f"Loaded {len(df)} predictions from {cache_type} cache")
        return df
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None


def save_predictions_to_cache(
    predictions: List[Dict],
    cache_type: str,
    actual_prices: Optional[pd.DataFrame] = None,
) -> bool:
    """
    Save predictions to parquet cache.
    
    Args:
        predictions: List of prediction dicts
        cache_type: "backtest" or "forecast"
        actual_prices: Optional DataFrame with actual prices for backtest
        
    Returns:
        True if saved successfully
    """
    ensure_cache_dir()
    cache_path = get_cache_path(cache_type)
    
    try:
        df = pd.DataFrame(predictions)
        df["date"] = pd.to_datetime(df["date"])
        
        # For backtest, merge with actual prices to calculate accuracy
        if cache_type == "backtest" and actual_prices is not None:
            df = merge_with_actual_prices(df, actual_prices)
        
        df.to_parquet(cache_path, index=False)
        print(f"Saved {len(df)} predictions to {cache_path}")
        return True
        
    except Exception as e:
        print(f"Error saving cache: {e}")
        return False


def merge_with_actual_prices(
    predictions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge predictions with actual prices to calculate accuracy.
    
    Adds columns:
    - actual_price: Actual closing price
    - price_change: Price change from previous day
    - actual_direction: "UP" or "DOWN" based on actual movement
    - correct: Whether prediction was correct
    """
    df = predictions_df.copy()
    prices = prices_df.copy()
    
    # Ensure date columns are datetime
    df["date"] = pd.to_datetime(df["date"])
    prices["date"] = pd.to_datetime(prices["date"])
    
    # Merge with actual prices
    df = pd.merge(
        df,
        prices[["date", "close"]].rename(columns={"close": "actual_price"}),
        on="date",
        how="left"
    )
    
    # Calculate actual direction (compare to previous day)
    df = df.sort_values("date").reset_index(drop=True)
    df["price_change"] = df["actual_price"].diff()
    df["actual_direction"] = df["price_change"].apply(
        lambda x: "UP" if x > 0 else "DOWN" if x < 0 else "FLAT"
    )
    
    # Calculate correctness
    df["correct"] = (df["direction"] == df["actual_direction"])
    
    # First row has no previous day, so mark as None
    df.loc[0, "correct"] = None
    
    return df


def get_backtest_with_accuracy(
    days: int = BACKTEST_DAYS,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Get backtest predictions with accuracy statistics.
    
    Returns:
        Tuple of (DataFrame, accuracy_stats_dict)
    """
    df = load_cached_predictions("backtest")
    
    if df is None or len(df) == 0:
        return pd.DataFrame(), {"accuracy": 0, "total": 0, "correct": 0}
    
    # Filter to requested days
    cutoff_date = datetime.now() - timedelta(days=days)
    df = df[df["date"] >= cutoff_date].copy()
    
    # Calculate accuracy stats
    valid_rows = df[df["correct"].notna()].copy()
    total = len(valid_rows)
    correct = valid_rows["correct"].sum() if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    stats = {
        "total": int(total),
        "correct": int(correct),
        "accuracy": round(float(accuracy), 4),
        "up_accuracy": 0,
        "down_accuracy": 0,
    }
    
    # Calculate per-direction accuracy
    if total > 0:
        up_preds = valid_rows[valid_rows["direction"] == "UP"]
        down_preds = valid_rows[valid_rows["direction"] == "DOWN"]
        
        if len(up_preds) > 0:
            stats["up_accuracy"] = round(up_preds["correct"].mean(), 4)
        if len(down_preds) > 0:
            stats["down_accuracy"] = round(down_preds["correct"].mean(), 4)
    
    return df, stats


def get_full_predictions() -> Dict:
    """
    Get all cached predictions (backtest + forecast) for the frontend.
    
    Returns:
        Dictionary with backtest, forecast, and accuracy stats
    """
    # Load backtest
    backtest_df, accuracy_stats = get_backtest_with_accuracy()
    
    # Load forecast
    forecast_df = load_cached_predictions("forecast")
    
    # Convert to JSON-serializable format
    backtest_data = []
    if backtest_df is not None and len(backtest_df) > 0:
        for _, row in backtest_df.iterrows():
            backtest_data.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "direction": row["direction"],
                "confidence": float(row["confidence"]) if pd.notna(row.get("confidence")) else 0.5,
                "actual_price": float(row["actual_price"]) if pd.notna(row.get("actual_price")) else None,
                "actual_direction": row.get("actual_direction"),
                "correct": bool(row["correct"]) if pd.notna(row.get("correct")) else None,
            })
    
    forecast_data = []
    if forecast_df is not None and len(forecast_df) > 0:
        for _, row in forecast_df.iterrows():
            forecast_data.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "direction": row["direction"],
                "confidence": float(row["confidence"]) if pd.notna(row.get("confidence")) else 0.5,
                "simulated_price": float(row.get("simulated_price", 0)),
            })
    
    return {
        "backtest": backtest_data,
        "forecast": forecast_data,
        "accuracy": accuracy_stats,
        "cache_info": {
            "backtest_count": len(backtest_data),
            "forecast_count": len(forecast_data),
            "last_updated": datetime.now().isoformat(),
        }
    }


def is_cache_stale(cache_type: str, max_age_hours: int = 24) -> bool:
    """Check if cache is older than max_age_hours."""
    cache_path = get_cache_path(cache_type)
    
    if not cache_path.exists():
        return True
    
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    age = datetime.now() - mtime
    
    return age > timedelta(hours=max_age_hours)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing cache service...")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Backtest cache exists: {get_cache_path('backtest').exists()}")
    print(f"Forecast cache exists: {get_cache_path('forecast').exists()}")
    
    data = get_full_predictions()
    print(f"\nBacktest predictions: {len(data['backtest'])}")
    print(f"Forecast predictions: {len(data['forecast'])}")
    print(f"Accuracy: {data['accuracy']}")
