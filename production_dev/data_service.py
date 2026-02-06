"""
Data Service Module for Bitcoin Astro Predictor.

Storage policy (project decision):
- No database reads for the web service.
- Single source of truth: local market files under `data/market/`.
- Preferred format: parquet.
- Optional fallback: processed CSV mirror.
"""

import sys
from pathlib import Path
from datetime import date, datetime
from typing import Optional, List, Dict
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MARKET_PARQUET_PATH = PROJECT_ROOT / "data" / "market" / "processed" / "BTC_full_market_daily.parquet"
MARKET_CSV_PATH = PROJECT_ROOT / "data" / "market" / "processed" / "BTC_full_market_daily.csv"
ARCHIVE_CSV_PATH = PROJECT_ROOT / "data" / "market" / "raw" / "Bitcoin Historical Data.csv"


def _normalize_market_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize market dataframe into canonical shape: [date, close].
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["date", "close"])

    out = df.copy()
    if "close" not in out.columns:
        raise ValueError("Market dataset has no 'close' column.")
    if "date" not in out.columns:
        raise ValueError("Market dataset has no 'date' column.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"])[["date", "close"]]
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return out


def _load_market_dataframe() -> tuple[pd.DataFrame, str]:
    """
    Load market history from local files only.

    Priority:
    1) processed parquet
    2) processed csv mirror
    3) raw archive csv (legacy fallback)
    """
    if MARKET_PARQUET_PATH.exists():
        df = pd.read_parquet(MARKET_PARQUET_PATH)
        return _normalize_market_df(df), "processed_parquet"

    if MARKET_CSV_PATH.exists():
        df = pd.read_csv(MARKET_CSV_PATH)
        return _normalize_market_df(df), "processed_csv"

    if ARCHIVE_CSV_PATH.exists():
        from src.market.archive_csv import load_bitcoin_historical_csv

        df = load_bitcoin_historical_csv(ARCHIVE_CSV_PATH, progress=False, verbose=False)
        return _normalize_market_df(df), "archive_csv"

    raise FileNotFoundError(
        "No local market data found. Expected one of:\n"
        f"- {MARKET_PARQUET_PATH}\n"
        f"- {MARKET_CSV_PATH}\n"
        f"- {ARCHIVE_CSV_PATH}"
    )


def export_market_csv() -> Optional[Path]:
    """
    Export processed market parquet into CSV mirror for optional CSV-only workflows.
    """
    if not MARKET_PARQUET_PATH.exists():
        return None
    df = pd.read_parquet(MARKET_PARQUET_PATH)
    df = _normalize_market_df(df)
    MARKET_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MARKET_CSV_PATH, index=False)
    return MARKET_CSV_PATH


def load_historical_prices(
    days: int = 30,
    subject_id: Optional[str] = None,
) -> List[Dict]:
    """
    Load historical BTC prices from local market files.

    Args:
        days: Number of days back to return (inclusive-style window).
        subject_id: Kept for API compatibility; ignored in file-only mode.
    """
    try:
        safe_days = max(1, int(days))
        # Inclusive window by default (N full day transitions needs N+1 points).
        window = safe_days + 1
        df, source = _load_market_dataframe()
        df = df.tail(window).copy()
        print(f"Loaded {len(df)} rows from local market store (source={source})")
        return [
            {"date": row["date"].strftime("%Y-%m-%d"), "price": float(row["close"])}
            for _, row in df.iterrows()
        ]
    except Exception as e:
        print(f"Error loading historical prices from local store: {e}")
        return []


def get_current_price(subject_id: Optional[str] = None) -> float:
    """
    Get the latest close from local market files.

    Args:
        subject_id: Kept for API compatibility; ignored in file-only mode.
    """
    try:
        df, _source = _load_market_dataframe()
        if len(df) > 0:
            return float(df.iloc[-1]["close"])
    except Exception as e:
        print(f"Error getting current price from local store: {e}")

    # Final fallback (UI still works, just less accurate)
    return 100000.0


def refresh_market_data(subject_id: Optional[str] = None) -> Dict:
    """
    Refresh local market parquet/csv from external source.

    Args:
        subject_id: Kept for API compatibility; ignored in file-only mode.
    """
    try:
        from production_dev.market_update import update_full_market_data

        result = update_full_market_data(progress=False, verbose=True)
        csv_path = export_market_csv()
        result["csv_exported"] = str(csv_path) if csv_path else None
        return result
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to refresh local market files",
        }


def get_data_summary(subject_id: Optional[str] = None) -> Dict:
    """
    Get summary of local market history.

    Args:
        subject_id: Kept for API compatibility; ignored in file-only mode.
    """
    try:
        df, source = _load_market_dataframe()
        if len(df) == 0:
            raise ValueError("Local market dataset is empty")

        if len(df) > 1:
            prev_close = float(df.iloc[-2]["close"])
            last_close = float(df.iloc[-1]["close"])
            daily_change = (last_close - prev_close) / prev_close * 100.0
        else:
            daily_change = 0.0

        return {
            "subject_id": subject_id,
            "total_rows": int(len(df)),
            "start_date": df["date"].min().strftime("%Y-%m-%d"),
            "end_date": df["date"].max().strftime("%Y-%m-%d"),
            "current_price": float(df.iloc[-1]["close"]),
            "price_change_1d": float(daily_change),
            "source": source,
        }
    except Exception as e:
        return {"error": str(e), "message": "Failed to load local data summary"}


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Data Service...")
    
    # Test historical prices
    print("\n📈 Historical Prices (last 7 days):")
    prices = load_historical_prices(days=7)
    for p in prices[-7:]:
        print(f"  {p['date']}: ${p['price']:,.2f}")
    
    # Test current price
    print(f"\n💰 Current Price: ${get_current_price():,.2f}")
    
    # Test data summary
    print("\n📊 Data Summary:")
    summary = get_data_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
