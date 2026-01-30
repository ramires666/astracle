"""
CLI: merge archived BTC quotes (CSV) with fresh Binance 1d into a single OHLCV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Add project root to sys.path so src.* imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.market.archive_csv import load_bitcoin_historical_csv
from src.market.parser import parse_klines_folder_1d_ohlcv


def _load_binance_ohlcv(
    raw_dir: Path,
    parquet_path: Path,
    progress: bool,
    verbose: bool,
) -> pd.DataFrame:
    """
    Load OHLCV from Binance. If parquet does not exist, create it from ZIPs.
    """
    if parquet_path.exists():
        if verbose:
            print(f"[MERGE] Found Binance parquet: {parquet_path}")
        return pd.read_parquet(parquet_path)

    if verbose:
        print(f"[MERGE] Binance parquet not found, parsing ZIPs: {raw_dir}")

    parse_klines_folder_1d_ohlcv(
        input_dir=raw_dir,
        output_path=parquet_path,
        progress=progress,
        verbose=verbose,
    )
    return pd.read_parquet(parquet_path)


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Add missing columns as NaN.
    """
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA
    return df


def main() -> int:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Merge archived BTC quotes and Binance 1d into one OHLCV",
    )
    parser.add_argument(
        "--archive-csv",
        default="data/market/raw/Bitcoin Historical Data.csv",
        help="Path to archived CSV",
    )
    parser.add_argument(
        "--binance-raw-dir",
        default="data/market/raw/klines_1d",
        help="Folder with Binance Vision ZIPs (1d)",
    )
    parser.add_argument(
        "--binance-parquet",
        default="data/market/processed/BTCUSDT_ohlcv_daily.parquet",
        help="Parquet for Binance OHLCV (created if missing)",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/market/processed/BTC_full_ohlcv_daily.parquet",
        help="Output parquet (archive + Binance)",
    )
    parser.add_argument(
        "--output-market-daily",
        default="data/market/processed/BTC_full_market_daily.parquet",
        help="Output parquet with date/close only",
    )
    parser.add_argument(
        "--prefer-binance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If dates overlap, prefer Binance",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress",
    )

    args = parser.parse_args()

    archive_csv = Path(args.archive_csv)
    binance_raw_dir = Path(args.binance_raw_dir)
    binance_parquet = Path(args.binance_parquet)
    output_parquet = Path(args.output_parquet)
    output_market_daily = Path(args.output_market_daily)

    print("[MERGE] Run params:")
    print(f"[MERGE]  archive_csv         = {archive_csv}")
    print(f"[MERGE]  binance_raw_dir     = {binance_raw_dir}")
    print(f"[MERGE]  binance_parquet     = {binance_parquet}")
    print(f"[MERGE]  output_parquet      = {output_parquet}")
    print(f"[MERGE]  output_market_daily = {output_market_daily}")
    print(f"[MERGE]  prefer_binance      = {args.prefer_binance}")
    print(f"[MERGE]  progress            = {args.progress}")

    # --- Archive ---
    df_arch = load_bitcoin_historical_csv(archive_csv, progress=args.progress, verbose=True)
    df_arch["source"] = "archive"

    # --- Binance ---
    df_bin = _load_binance_ohlcv(
        raw_dir=binance_raw_dir,
        parquet_path=binance_parquet,
        progress=args.progress,
        verbose=True,
    )
    df_bin["source"] = "binance"

    # --- Column unification ---
    all_cols = [
        "date", "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_volume", "taker_buy_quote_volume", "count",
        "change_pct", "source",
    ]
    df_arch = _ensure_columns(df_arch, all_cols)
    df_bin = _ensure_columns(df_bin, all_cols)

    df_arch = df_arch[all_cols].copy()
    df_bin = df_bin[all_cols].copy()

    # --- Merge ---
    df_all = pd.concat([df_arch, df_bin], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date

    # Priority
    if args.prefer_binance:
        priority = {"archive": 0, "binance": 1}
    else:
        priority = {"archive": 1, "binance": 0}
    df_all["priority"] = df_all["source"].map(priority).fillna(0)

    # Keep one record per date by priority
    df_all = df_all.sort_values(["date", "priority"]).drop_duplicates(subset=["date"], keep="last")
    df_all = df_all.sort_values("date").reset_index(drop=True)
    df_all = df_all.drop(columns=["priority"])

    # --- Save ---
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(output_parquet, index=False)
    df_all[["date", "close"]].to_parquet(output_market_daily, index=False)

    print(f"[MERGE] Final period: {df_all['date'].iloc[0]} -> {df_all['date'].iloc[-1]}")
    print(f"[MERGE] Final rows: {len(df_all)}")
    print(f"[MERGE] Saved: {output_parquet}")
    print(f"[MERGE] Saved: {output_market_daily}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
