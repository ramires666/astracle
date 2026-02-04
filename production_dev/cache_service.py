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
BACKTEST_DAYS = 1095   # 3 years of past predictions
FORECAST_DAYS = 365   # 1 year of future predictions


# =============================================================================
# IN-MEMORY CACHE (loaded from parquet on startup)
# =============================================================================

class MemoryCache:
    """
    In-memory cache for instant response.
    Loaded from parquet files on startup, updated when new predictions are generated.
    """
    
    def __init__(self):
        self.backtest_data: List[Dict] = []
        self.forecast_data: List[Dict] = []
        self.accuracy_stats: Dict = {"accuracy": 0, "total": 0, "correct": 0}
        self.is_loaded: bool = False
        self.last_loaded: Optional[datetime] = None
    
    def load_from_parquet(self) -> bool:
        """Load cache from parquet files into memory."""
        try:
            # Load backtest
            backtest_df, self.accuracy_stats = get_backtest_with_accuracy_from_file()
            if backtest_df is not None and len(backtest_df) > 0:
                self.backtest_data = df_to_backtest_list(backtest_df)
            
            # Load forecast
            forecast_df = load_cached_predictions("forecast")
            if forecast_df is not None and len(forecast_df) > 0:
                self.forecast_data = df_to_forecast_list(forecast_df)
            
            self.is_loaded = True
            self.last_loaded = datetime.now()
            
            print(f"📦 Memory cache loaded: {len(self.backtest_data)} backtest, {len(self.forecast_data)} forecast")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to load memory cache: {e}")
            return False
    
    def get_all(self) -> Dict:
        """Get all cached data (instant response)."""
        return {
            "backtest": self.backtest_data,
            "forecast": self.forecast_data,
            "accuracy": self.accuracy_stats,
            "cache_info": {
                "backtest_count": len(self.backtest_data),
                "forecast_count": len(self.forecast_data),
                "last_updated": self.last_loaded.isoformat() if self.last_loaded else None,
                "is_loaded": self.is_loaded,
            }
        }


# Global in-memory cache instance
_memory_cache = MemoryCache()


def get_memory_cache() -> MemoryCache:
    """Get the global memory cache instance."""
    return _memory_cache


def init_memory_cache() -> bool:
    """Initialize memory cache from parquet files. Call on server startup."""
    return _memory_cache.load_from_parquet()


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
    
    IMPORTANT: The model predicts the direction for the NEXT day.
    So if we have a prediction for date X with direction "UP",
    it means "price on day X+1 will be higher than on day X".
    
    To calculate accuracy, we compare:
    - prediction[date] with actual_price[date+1] vs actual_price[date]
    
    Adds columns:
    - actual_price: Actual closing price on prediction date
    - next_price: Actual closing price on next day  
    - price_change: Price change from prediction date to next day
    - actual_direction: "UP" or "DOWN" based on actual movement
    - correct: Whether prediction was correct
    """
    df = predictions_df.copy()
    prices = prices_df.copy()
    
    # Ensure date columns are datetime
    df["date"] = pd.to_datetime(df["date"])
    prices["date"] = pd.to_datetime(prices["date"])
    
    # Sort prices by date for proper shifting
    prices = prices.sort_values("date").reset_index(drop=True)
    
    # Merge with actual prices for the prediction date
    df = pd.merge(
        df,
        prices[["date", "close"]].rename(columns={"close": "actual_price"}),
        on="date",
        how="left"
    )
    
    # Add next day's price (shift prices by -1 to get next day)
    prices["next_day_price"] = prices["close"].shift(-1)
    df = pd.merge(
        df,
        prices[["date", "next_day_price"]],
        on="date",
        how="left"
    )
    
    # Calculate actual direction (compare current day to next day)
    # If prediction is for date X, we compare close[X+1] vs close[X]
    df = df.sort_values("date").reset_index(drop=True)
    df["price_change"] = df["next_day_price"] - df["actual_price"]
    df["actual_direction"] = df["price_change"].apply(
        lambda x: "UP" if x > 0 else "DOWN" if x < 0 else "FLAT" if pd.notna(x) else None
    )
    
    # Calculate correctness (use object dtype to allow None)
    df["correct"] = None  # Initialize as None
    valid_mask = df["actual_direction"].notna() & (df["actual_direction"] != "FLAT")
    df.loc[valid_mask, "correct"] = (
        df.loc[valid_mask, "direction"] == df.loc[valid_mask, "actual_direction"]
    )
    
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
    
    # Default stats if 'correct' column doesn't exist
    stats = {
        "total": 0,
        "correct": 0,
        "accuracy": 0,
        "up_accuracy": 0,
        "down_accuracy": 0,
    }
    
    # Only calculate accuracy if 'correct' column exists
    if "correct" in df.columns:
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


# Alias for use by MemoryCache
get_backtest_with_accuracy_from_file = get_backtest_with_accuracy


def df_to_backtest_list(df: pd.DataFrame) -> List[Dict]:
    """Convert backtest DataFrame to list of dicts for JSON."""
    data = []
    for _, row in df.iterrows():
        item = {
            "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], 'strftime') else str(row["date"]),
            "direction": row["direction"],
            "confidence": float(row["confidence"]) if pd.notna(row.get("confidence")) else 0.5,
        }
        # Optional columns (may not exist if actual prices weren't merged)
        if "actual_price" in row.index:
            item["actual_price"] = float(row["actual_price"]) if pd.notna(row["actual_price"]) else None
        else:
            item["actual_price"] = None
        if "actual_direction" in row.index:
            item["actual_direction"] = row["actual_direction"] if pd.notna(row.get("actual_direction")) else None
        else:
            item["actual_direction"] = None
        if "correct" in row.index:
            item["correct"] = bool(row["correct"]) if pd.notna(row["correct"]) else None
        else:
            item["correct"] = None
        data.append(item)
    return data


def df_to_forecast_list(df: pd.DataFrame) -> List[Dict]:
    """Convert forecast DataFrame to list of dicts for JSON."""
    data = []
    for _, row in df.iterrows():
        data.append({
            "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], 'strftime') else str(row["date"]),
            "direction": row["direction"],
            "confidence": float(row["confidence"]) if pd.notna(row.get("confidence")) else 0.5,
            "simulated_price": float(row.get("simulated_price", 0)),
        })
    return data


def get_full_predictions() -> Dict:
    """
    Get all cached predictions (backtest + forecast) for the frontend.
    
    Uses in-memory cache for instant response. Falls back to file loading
    if memory cache is not initialized.
    
    Returns:
        Dictionary with backtest, forecast, and accuracy stats
    """
    # Try memory cache first (instant response)
    cache = get_memory_cache()
    if cache.is_loaded:
        return cache.get_all()
    
    # Fall back to loading from files (slower)
    print("⚠️ Memory cache not loaded, falling back to file-based loading")
    
    # Load backtest
    backtest_df, accuracy_stats = get_backtest_with_accuracy()
    backtest_data = df_to_backtest_list(backtest_df) if len(backtest_df) > 0 else []
    
    # Load forecast
    forecast_df = load_cached_predictions("forecast")
    forecast_data = df_to_forecast_list(forecast_df) if forecast_df is not None and len(forecast_df) > 0 else []
    
    return {
        "backtest": backtest_data,
        "forecast": forecast_data,
        "accuracy": accuracy_stats,
        "cache_info": {
            "backtest_count": len(backtest_data),
            "forecast_count": len(forecast_data),
            "last_updated": datetime.now().isoformat(),
            "is_loaded": False,  # Indicates fallback was used
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
