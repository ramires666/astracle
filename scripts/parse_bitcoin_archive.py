"""
CLI: parse archived CSV and save full daily OHLCV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Add project root to sys.path so src.* imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.market.archive_csv import load_bitcoin_historical_csv


def main() -> int:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Parse archived Bitcoin Historical Data.csv to parquet",
    )
    parser.add_argument(
        "--input",
        default="data/market/raw/Bitcoin Historical Data.csv",
        help="Path to archived CSV",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/market/processed/BTC_archive_ohlcv_daily.parquet",
        help="Where to save parquet",
    )
    parser.add_argument(
        "--output-csv",
        default="data/market/processed/BTC_archive_ohlcv_daily.csv",
        help="Where to save CSV (for checks)",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress",
    )
    parser.add_argument(
        "--save-market-daily",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save separate date/close file (for DB load)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_parquet = Path(args.output_parquet)
    output_csv = Path(args.output_csv)

    print("[ARCH] Run params:")
    print(f"[ARCH]  input          = {input_path}")
    print(f"[ARCH]  output_parquet = {output_parquet}")
    print(f"[ARCH]  output_csv     = {output_csv}")
    print(f"[ARCH]  progress       = {args.progress}")
    print(f"[ARCH]  save_market_daily = {args.save_market_daily}")

    df = load_bitcoin_historical_csv(input_path, progress=args.progress, verbose=True)

    # Save full OHLCV
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"[ARCH] Saved parquet: {output_parquet}")

    # CSV for quick checks
    df.to_csv(output_csv, index=False)
    print(f"[ARCH] Saved CSV: {output_csv}")

    # Optional date/close file
    if args.save_market_daily:
        market_daily_path = output_parquet.parent / "BTC_archive_market_daily.parquet"
        df[["date", "close"]].to_parquet(market_daily_path, index=False)
        print(f"[ARCH] Saved market_daily parquet: {market_daily_path}")

    print("[ARCH] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
