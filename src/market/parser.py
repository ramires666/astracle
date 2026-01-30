"""
Parser for daily (1d) klines from Binance Vision ZIP archives.

In MVP we only keep date and close.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
import zipfile


def _iter_with_progress(items: Iterable, enabled: bool, desc: str):
    """
    Progress wrapper. If tqdm is not installed, return items as-is.
    """
    if not enabled:
        return items

    try:
        from tqdm import tqdm
    except ImportError:
        print("[WARN] tqdm is not installed, progress disabled.")
        return items

    return tqdm(items, desc=desc, unit="file")


def _detect_header(first_line: str) -> bool:
    """
    Detect if CSV has a header.
    """
    lower = first_line.lower()
    if "open_time" in lower or "close" in lower:
        return True
    # If first column is not a digit, it's a header
    first = first_line.split(",")[0].strip()
    return not first.isdigit()


def read_klines_zip_1d(path: Path) -> pd.DataFrame:
    """
    Read a ZIP with daily candles.
    Returns DataFrame with columns: date, close.
    """
    path = Path(path)

    with zipfile.ZipFile(path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV in archive: {path}")

        member = csv_members[0]
        with zf.open(member) as f:
            first_line = f.readline().decode("utf-8").strip()
            has_header = _detect_header(first_line)

        with zf.open(member) as f:
            if has_header:
                df = pd.read_csv(f)
            else:
                df = pd.read_csv(f, header=None)

    # Expected Binance column order (if no header)
    expected_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"
    ]
    if not has_header:
        df.columns = expected_cols[:len(df.columns)]

    # Convert open_time to date
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["date"] = df["open_time"].dt.date
    else:
        raise RuntimeError("Column open_time not found")

    # Keep only date and close
    df = df[["date", "close"]].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    return df


def read_klines_zip_1d_full(path: Path) -> pd.DataFrame:
    """
    Read ZIP with daily candles and return full OHLCV.
    Columns: date, open, high, low, close, volume, quote_volume,
             taker_buy_volume, taker_buy_quote_volume, count
    """
    path = Path(path)

    with zipfile.ZipFile(path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError(f"No CSV in archive: {path}")

        member = csv_members[0]
        with zf.open(member) as f:
            first_line = f.readline().decode("utf-8").strip()
            has_header = _detect_header(first_line)

        with zf.open(member) as f:
            if has_header:
                df = pd.read_csv(f)
            else:
                df = pd.read_csv(f, header=None)

    expected_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"
    ]
    if not has_header:
        df.columns = expected_cols[:len(df.columns)]

    # Convert open_time to date
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["date"] = df["open_time"].dt.date
    else:
        raise RuntimeError("Column open_time not found")

    # Convert numeric fields
    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_volume", "taker_buy_quote_volume", "count"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = ["date"] + [c for c in numeric_cols if c in df.columns]
    df = df[keep_cols].copy()

    return df


def parse_klines_folder_1d(
    input_dir: Path,
    output_path: Path,
    progress: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Read all ZIP files in a folder and save combined result.
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.zip"))
    if not files:
        raise FileNotFoundError(f"ZIP files not found: {input_dir}")

    if verbose:
        print(f"[INFO] ZIP files found: {len(files)}")

    dfs: List[pd.DataFrame] = []
    for fp in _iter_with_progress(files, progress, desc="parse 1d"):
        try:
            dfs.append(read_klines_zip_1d(fp))
        except Exception as e:
            if verbose:
                print(f"[WARN] Parse error {fp}: {e}")

    if not dfs:
        raise RuntimeError("Failed to parse any files.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["date"])
    df_all = df_all.sort_values("date").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(output_path, index=False)
    if verbose:
        print(f"[OK] Saved: {output_path} ({len(df_all)} rows)")
    return output_path


def parse_klines_folder_1d_ohlcv(
    input_dir: Path,
    output_path: Path,
    progress: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Read all ZIP files with daily candles and save full OHLCV.
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.zip"))
    if not files:
        raise FileNotFoundError(f"ZIP files not found: {input_dir}")

    if verbose:
        print(f"[INFO] ZIP files found: {len(files)}")

    dfs: List[pd.DataFrame] = []
    for fp in _iter_with_progress(files, progress, desc="parse ohlcv"):
        try:
            dfs.append(read_klines_zip_1d_full(fp))
        except Exception as e:
            if verbose:
                print(f"[WARN] Parse error {fp}: {e}")

    if not dfs:
        raise RuntimeError("Failed to parse any files.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["date"])
    df_all = df_all.sort_values("date").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(output_path, index=False)

    if verbose:
        print(f"[OK] Saved: {output_path} ({len(df_all)} rows)")
    return output_path
