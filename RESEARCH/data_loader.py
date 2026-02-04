"""
Data loader module for RESEARCH pipeline.
Loads market data from local parquet (DB is no longer used).
"""
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Optional

from .config import cfg, PROJECT_ROOT


def load_market_data(
    subject_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load market daily data from local parquet.
    
    Args:
        subject_id: Subject ID to load (defaults to active subject).
            We keep this argument for API compatibility, but parquet is global.
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        DataFrame with columns: date, close
    """
    subject_id = subject_id or cfg.active_subject_id
    parquet_path = _resolve_market_parquet_path(subject_id)

    df = pd.read_parquet(parquet_path)
    
    if df.empty:
        raise ValueError(f"No market data found in parquet: {parquet_path}")
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows from parquet for subject={subject_id}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    
    return df


def get_latest_date(subject_id: Optional[str] = None) -> Optional[datetime]:
    """
    Get the latest date available in the parquet file.
    
    Args:
        subject_id: Subject ID to check (defaults to active subject)
    
    Returns:
        Latest date or None if no data
    """
    subject_id = subject_id or cfg.active_subject_id
    try:
        parquet_path = _resolve_market_parquet_path(subject_id)
        df = pd.read_parquet(parquet_path, columns=["date"])
        if df.empty:
            return None
        return pd.to_datetime(df["date"]).max()
    except Exception:
        return None


def get_data_paths() -> dict:
    """Get commonly used data paths."""
    return {
        "project_root": PROJECT_ROOT,
        "data_root": cfg.data_root,
        "processed_dir": cfg.processed_dir,
        "reports_dir": cfg.reports_dir,
    }


def _resolve_market_parquet_path(subject_id: str) -> Path:
    """
    Pick the best available parquet file for market data.

    Priority order (most complete first):
    1) BTC_full_market_daily.parquet (archive + Binance merged)
    2) {SYMBOL}_market_daily.parquet (Binance-only daily)
    3) btc_market_daily.parquet (legacy name)
    4) BTC_archive_market_daily.parquet (archive-only)
    """
    processed = cfg.processed_dir
    symbol = (cfg.subject or {}).get("symbol", "")

    candidates = [
        processed / "BTC_full_market_daily.parquet",
        processed / f"{symbol}_market_daily.parquet" if symbol else None,
        processed / "btc_market_daily.parquet",
        processed / "BTC_archive_market_daily.parquet",
    ]

    for path in candidates:
        if path and path.exists():
            return path

    raise FileNotFoundError(
        f"No market parquet found in {processed}. "
        "Expected BTC_full_market_daily.parquet or a symbol-specific parquet."
    )
