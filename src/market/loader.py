"""
Load daily market data to DB or local files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_market_daily_parquet(path: Path) -> pd.DataFrame:
    """
    Load parquet with market_daily (date, close).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_parquet(path)
    return df


def save_market_daily_csv(df: pd.DataFrame, path: Path) -> Path:
    """
    Save market_daily to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_market_daily_parquet(df: pd.DataFrame, path: Path) -> Path:
    """
    Save market_daily to parquet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def insert_market_daily_psql(
    df: pd.DataFrame,
    subject_id: str,
    db_url: str,
    table: str = "market_daily",
) -> None:
    """
    Load data into Postgres.
    NOTE: requires psycopg2.
    """
    try:
        import psycopg2
    except ImportError as e:
        raise RuntimeError("psycopg2 is not installed. Install it and retry.") from e

    # Prepare data
    rows = [(subject_id, row["date"], float(row["close"])) for _, row in df.iterrows()]

    conn = psycopg2.connect(db_url)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.executemany(
                    f"""
                    INSERT INTO {table} (subject_id, date, close)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (subject_id, date) DO UPDATE
                      SET close = EXCLUDED.close
                    """,
                    rows,
                )
    finally:
        conn.close()


def load_market_daily_psql(
    subject_id: str,
    db_url: str,
    table: str = "market_daily",
) -> pd.DataFrame:
    """
    Load market_daily data from Postgres.
    NOTE: requires psycopg2.
    """
    try:
        import psycopg2
    except ImportError as e:
        raise RuntimeError("psycopg2 is not installed. Install it and retry.") from e

    conn = psycopg2.connect(db_url)
    try:
        query = f"""
            SELECT date, close
            FROM {table}
            WHERE subject_id = %s
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn, params=[subject_id])
        return df
    finally:
        conn.close()
