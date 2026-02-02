"""
CLI: download Binance Vision 1d klines (monthly archives).

This script only downloads ZIP files into data_root/raw/klines_1d.
Use parse_klines_folder_1d to build parquet afterwards.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import sys

# Add project root to sys.path so src.* imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_yaml, load_subjects
from src.market.downloader import download_1d_auto, download_monthly_1d
from src.market.parser import parse_klines_folder_1d


def _resolve_path(value: str | Path) -> Path:
    """
    Resolve relative paths against project root.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def main() -> int:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Download Binance Vision 1d klines (monthly ZIPs)",
    )
    parser.add_argument(
        "--config-market",
        default="configs/market.yaml",
        help="Path to market.yaml",
    )
    parser.add_argument(
        "--config-subjects",
        default="configs/subjects.yaml",
        help="Path to subjects.yaml (for optional subject_id)",
    )
    parser.add_argument("--symbol", default=None, help="Override symbol (e.g. BTCUSDT)")
    parser.add_argument("--market-type", default=None, help="Override market type (spot or futures)")
    parser.add_argument("--start-year", type=int, default=None, help="Start year override")
    parser.add_argument("--start-month", type=int, default=None, help="Start month override")
    parser.add_argument("--end-year", type=int, default=None, help="End year override")
    parser.add_argument("--end-month", type=int, default=None, help="End month override")
    parser.add_argument(
        "--auto-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-detect the first available month in archive",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress",
    )
    parser.add_argument(
        "--print-urls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print URLs while downloading",
    )
    parser.add_argument(
        "--daily-tail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If monthly archive is missing, download daily archives instead",
    )
    parser.add_argument(
        "--parse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Parse ZIPs into market_daily parquet after download",
    )
    parser.add_argument(
        "--check-db",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check local DB cache before downloading",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force download even if DB cache exists",
    )
    parser.add_argument(
        "--subject-id",
        default=None,
        help="Subject ID for output filename (optional)",
    )

    args = parser.parse_args()

    cfg_market = load_yaml(args.config_market)
    market_cfg = cfg_market["market"]

    subjects, active_id = load_subjects(args.config_subjects)
    subject_id = args.subject_id or active_id
    subject = subjects.get(subject_id)

    symbol = args.symbol or market_cfg["symbol"]
    market_type = args.market_type or market_cfg["market_type"]

    data_root = _resolve_path(market_cfg["data_root"])
    raw_dir = data_root / "raw" / "klines_1d"
    processed_dir = data_root / "processed"

    # Defaults from config (if overrides not provided)
    start_year = args.start_year or market_cfg.get("start_year", 2010)
    start_month = args.start_month or market_cfg.get("start_month", 1)

    now = datetime.utcnow()
    end_year = args.end_year or market_cfg.get("end_year") or now.year
    end_month = args.end_month or market_cfg.get("end_month") or now.month

    print("[DL] Run params:")
    print(f"[DL]  symbol      = {symbol}")
    print(f"[DL]  market_type = {market_type}")
    print(f"[DL]  data_root   = {data_root}")
    print(f"[DL]  auto_start  = {args.auto_start}")
    print(f"[DL]  start       = {start_year}-{start_month:02d}")
    print(f"[DL]  end         = {end_year}-{end_month:02d}")
    print(f"[DL]  progress    = {args.progress}")
    print(f"[DL]  print_urls  = {args.print_urls}")
    print(f"[DL]  parse       = {args.parse}")
    print(f"[DL]  daily_tail  = {args.daily_tail}")
    print(f"[DL]  check_db   = {args.check_db}")
    print(f"[DL]  force      = {args.force}")

    # DB cache check (local first)
    if args.check_db and not args.force:
        try:
            cfg_db = load_yaml("configs/db.yaml")
            db_url = cfg_db.get("db", {}).get("url", "")
            if db_url:
                from src.market.loader import load_market_daily_psql

                df_db = load_market_daily_psql(subject.subject_id, db_url)
                if df_db is not None and not df_db.empty:
                    last_date = pd.to_datetime(df_db['date']).max()
                    print(f"[DL] DB cache found. Last date: {last_date.date()}")
                    
                    # If data is very fresh (today or yesterday), we can skip
                    if last_date.date() >= (datetime.utcnow().date() - timedelta(days=1)):
                        print("[DL] Data is up to date. Skipping download.")
                        return 0
                    
                    # Otherwise, start downloading from the next month
                    # We might have a partial month, so it's safer to start from the month of the last date
                    # The downloader handles existing files with [SKIP], so overlapping is fine and safer
                    start_year = last_date.year
                    start_month = last_date.month
                    args.auto_start = False # Disable auto-search, we know where to start
                    print(f"[DL] Resuming download from {start_year}-{start_month:02d}")
        except Exception as e:
            print(f"[WARN] DB cache check failed: {e}")

    if args.auto_start:
        download_1d_auto(
            symbol=symbol,
            market_type=market_type,
            data_root=data_root,
            auto_start=True,
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
            progress=args.progress,
            verbose=True,
            print_urls=args.print_urls,
            daily_tail=args.daily_tail,
        )
    else:
        download_monthly_1d(
            symbol=symbol,
            market_type=market_type,
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
            dest_folder=raw_dir,
            progress=args.progress,
            verbose=True,
            print_urls=args.print_urls,
        )

    if args.parse:
        if subject is not None:
            out_name = f"{subject.subject_id}_market_daily.parquet"
        else:
            out_name = f"{symbol}_market_daily.parquet"
        output_path = processed_dir / out_name
        parse_klines_folder_1d(raw_dir, output_path, progress=args.progress, verbose=True)
        print(f"[DL] Parsed parquet: {output_path}")

        # Insert into DB if subject is valid
        if subject is not None and args.check_db:
            try:
                from src.market.loader import insert_market_daily_psql, load_market_daily_parquet
                cfg_db = load_yaml("configs/db.yaml")
                db_url = cfg_db.get("db", {}).get("url", "")
                if db_url:
                    df_parsed = load_market_daily_parquet(output_path)
                    print(f"[DL] Inserting {len(df_parsed)} rows into DB (upsert)...")
                    insert_market_daily_psql(df_parsed, subject.subject_id, db_url)
                    print("[DL] DB insertion complete.")
            except Exception as e:
                print(f"[WARN] Failed to insert into DB: {e}")

    else:
        print("[DL] Parse step skipped. Use parse_klines_folder_1d to build parquet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
