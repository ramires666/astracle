"""
Data loader module for RESEARCH pipeline.
Loads market data directly from PostgreSQL database.
"""
import pandas as pd
import psycopg2
from datetime import datetime
from typing import Optional

from .config import cfg, PROJECT_ROOT


def load_market_data(
    subject_id: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load market daily data from PostgreSQL database.
    
    Args:
        subject_id: Subject ID to load (defaults to active subject)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        DataFrame with columns: date, close
    """
    subject_id = subject_id or cfg.active_subject_id
    db_url = cfg.db_url
    
    if not db_url:
        raise ValueError("Database URL not configured in configs/db.yaml")
    
    # Build query (market_daily only has date and close)
    query = """
        SELECT date, close 
        FROM market_daily 
        WHERE subject_id = %s
    """
    params = [subject_id]
    
    if start_date:
        query += " AND date >= %s"
        params.append(start_date)
    
    if end_date:
        query += " AND date <= %s"
        params.append(end_date)
    
    query += " ORDER BY date"
    
    try:
        conn = psycopg2.connect(db_url)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
    except Exception as e:
        raise RuntimeError(f"Failed to load market data from DB: {e}")
    
    if df.empty:
        raise ValueError(f"No market data found for subject_id={subject_id}")
    
    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    print(f"Loaded {len(df)} rows from DB for subject={subject_id}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    
    return df


def get_latest_date(subject_id: Optional[str] = None) -> Optional[datetime]:
    """
    Get the latest date available for a subject in the database.
    
    Args:
        subject_id: Subject ID to check (defaults to active subject)
    
    Returns:
        Latest date or None if no data
    """
    subject_id = subject_id or cfg.active_subject_id
    db_url = cfg.db_url
    
    if not db_url:
        return None
    
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MAX(date) FROM market_daily WHERE subject_id = %s",
            (subject_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return pd.Timestamp(result[0])
    except Exception:
        pass
    
    return None


def get_data_paths() -> dict:
    """Get commonly used data paths."""
    return {
        "project_root": PROJECT_ROOT,
        "data_root": cfg.data_root,
        "processed_dir": cfg.processed_dir,
        "reports_dir": cfg.reports_dir,
    }
