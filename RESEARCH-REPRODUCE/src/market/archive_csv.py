"""
Parser for archived BTC historical CSV (Investing.com format).
Provides full daily OHLCV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _parse_float(value: str) -> Optional[float]:
    """
    Parse a number string like "67,211.9" to float.
    Returns None if empty.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s or s == "-":
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_volume(value: str) -> Optional[float]:
    """
    Parse volume like "65.59K"/"1.23M"/"0.45B" to float.
    """
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not s or s == "-":
        return None

    multiplier = 1.0
    if s.endswith("K"):
        multiplier = 1e3
        s = s[:-1]
    elif s.endswith("M"):
        multiplier = 1e6
        s = s[:-1]
    elif s.endswith("B"):
        multiplier = 1e9
        s = s[:-1]

    try:
        return float(s) * multiplier
    except ValueError:
        return None


def _parse_change_pct(value: str) -> Optional[float]:
    """
    Parse percent string like "4.96%" to float.
    """
    if value is None:
        return None
    s = str(value).strip().replace("%", "")
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_bitcoin_historical_csv(
    path: Path,
    progress: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load archived CSV and return DataFrame with columns:
    date, open, high, low, close, volume, change_pct.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if verbose:
        print(f"[ARCH] Reading CSV: {path}")

    df = pd.read_csv(path)

    # Expected columns: Date, Price, Open, High, Low, Vol., Change %
    rename_map = {
        "Date": "date",
        "Price": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Vol.": "volume_raw",
        "Change %": "change_raw",
    }
    df = df.rename(columns=rename_map)

    # Parse date
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce").dt.date

    # Vectorized number parsing
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(str).map(_parse_float)

    if progress:
        try:
            from tqdm import tqdm

            tqdm.pandas()
            df["volume"] = df["volume_raw"].progress_apply(_parse_volume)
            df["change_pct"] = df["change_raw"].progress_apply(_parse_change_pct)
        except Exception:
            if verbose:
                print("[WARN] tqdm is unavailable, running without progress.")
            df["volume"] = df["volume_raw"].map(_parse_volume)
            df["change_pct"] = df["change_raw"].map(_parse_change_pct)
    else:
        df["volume"] = df["volume_raw"].map(_parse_volume)
        df["change_pct"] = df["change_raw"].map(_parse_change_pct)

    # Drop raw columns
    df = df.drop(columns=["volume_raw", "change_raw"])

    # Cleanup and sort
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if verbose:
        if not df.empty:
            print(f"[ARCH] Period: {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
            print(f"[ARCH] Rows: {len(df)}")
        else:
            print("[ARCH] Empty DataFrame after processing.")

    return df
