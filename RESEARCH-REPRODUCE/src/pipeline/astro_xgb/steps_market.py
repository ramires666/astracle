"""
Market data step: download/parse/merge daily data and produce market_daily parquet.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.pipeline.astro_xgb.context import PipelineContext
from src.market.archive_csv import load_bitcoin_historical_csv
from src.market.downloader import download_1d_auto
from src.market.loader import load_market_daily_parquet, load_market_daily_csv
from src.market.loader import save_market_daily_parquet
from src.market.loader import load_market_daily_psql
from src.market.parser import parse_klines_folder_1d_ohlcv


def _resolve_existing_market_file(processed_dir: Path, subject_id: str, symbol: str) -> Optional[Path]:
    """
    Try common market_daily file names and return the first existing one.
    """
    candidates = [
        processed_dir / f"{subject_id}_market_daily.parquet",
        processed_dir / f"{subject_id}_market_daily.csv",
        processed_dir / f"{symbol}_market_daily.parquet",
        processed_dir / f"{symbol}_market_daily.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _merge_archive_and_binance(
    df_arch: pd.DataFrame,
    df_bin: pd.DataFrame,
    prefer_binance: bool = True,
) -> pd.DataFrame:
    """
    Merge archived CSV (Investing) with Binance OHLCV by date.
    """
    df_arch = df_arch.copy()
    df_bin = df_bin.copy()
    df_arch["source"] = "archive"
    df_bin["source"] = "binance"

    all_cols = [
        "date", "open", "high", "low", "close", "volume",
        "quote_volume", "taker_buy_volume", "taker_buy_quote_volume", "count",
        "change_pct", "source",
    ]
    for col in all_cols:
        if col not in df_arch.columns:
            df_arch[col] = pd.NA
        if col not in df_bin.columns:
            df_bin[col] = pd.NA

    df_arch = df_arch[all_cols].copy()
    df_bin = df_bin[all_cols].copy()

    df_all = pd.concat([df_arch, df_bin], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date

    if prefer_binance:
        priority = {"archive": 0, "binance": 1}
    else:
        priority = {"archive": 1, "binance": 0}
    df_all["priority"] = df_all["source"].map(priority).fillna(0)

    df_all = df_all.sort_values(["date", "priority"]).drop_duplicates(subset=["date"], keep="last")
    df_all = df_all.sort_values("date").reset_index(drop=True)
    df_all = df_all.drop(columns=["priority"])
    return df_all


def run_market_step(
    ctx: PipelineContext,
    force: bool = False,
    update: bool = False,
    prefer_binance: bool = True,
    data_start: Optional[str] = None,
    save_db: Optional[bool] = None,
) -> Path:
    """
    Ensure market_daily parquet exists and return its path.
    """
    market_cfg = ctx.cfg_market.get("market", {})
    symbol = market_cfg.get("symbol") or ctx.subject.symbol
    market_type = market_cfg.get("market_type", "futures_um")
    write_db = bool(market_cfg.get("write_db", True) if save_db is None else save_db)

    ctx.processed_dir.mkdir(parents=True, exist_ok=True)
    ctx.raw_dir.mkdir(parents=True, exist_ok=True)

    out_market_daily = ctx.processed_dir / f"{ctx.subject.subject_id}_market_daily.parquet"
    out_ohlcv = ctx.processed_dir / f"{ctx.subject.subject_id}_ohlcv_daily.parquet"

    if out_market_daily.exists() and not force and not update:
        try:
            df_existing = pd.read_parquet(out_market_daily)
            if "date" in df_existing.columns:
                df_existing["date"] = pd.to_datetime(df_existing["date"])
            print("[MARKET] Using cached file:", out_market_daily)
            if not df_existing.empty:
                print(f"[MARKET] Rows: {len(df_existing)} | "
                      f"Range: {df_existing['date'].min().date()} -> {df_existing['date'].max().date()}")
        except Exception:
            print("[MARKET] Using cached file:", out_market_daily)
        return out_market_daily

    # Try existing processed file if update is not required
    if not update:
        existing = _resolve_existing_market_file(ctx.processed_dir, ctx.subject.subject_id, symbol)
        if existing is not None and not force:
            if existing.suffix == ".parquet":
                df_market = load_market_daily_parquet(existing)
            else:
                df_market = load_market_daily_csv(existing)
            df_market = df_market.copy()
            df_market["date"] = pd.to_datetime(df_market["date"])
            if data_start:
                df_market = df_market[df_market["date"] >= pd.Timestamp(data_start)]
            save_market_daily_parquet(df_market, out_market_daily)
            print("[MARKET] Loaded from existing file:", existing)
            if not df_market.empty:
                print(f"[MARKET] Rows: {len(df_market)} | "
                      f"Range: {df_market['date'].min().date()} -> {df_market['date'].max().date()}")
            print("[MARKET] Saved market_daily:", out_market_daily)
            return out_market_daily

    # Check DB first (local cache)
    df_db = None
    if bool(market_cfg.get("use_db_cache", False)):
        db_url = (ctx.cfg_db.get("db") or {}).get("url", "")
        if db_url:
            try:
                df_db = load_market_daily_psql(ctx.subject.subject_id, db_url)
                if df_db is not None and not df_db.empty:
                    df_db = df_db.copy()
                    df_db["date"] = pd.to_datetime(df_db["date"])
                    if data_start:
                        df_db = df_db[df_db["date"] >= pd.Timestamp(data_start)]

                    # If not updating, prefer DB and skip download.
                    if not update and not force:
                        save_market_daily_parquet(df_db, out_market_daily)
                        print("[MARKET] Loaded from DB cache (no update).")
                        print(f"[MARKET] Rows: {len(df_db)} | "
                              f"Range: {df_db['date'].min().date()} -> {df_db['date'].max().date()}")
                        print("[MARKET] Saved market_daily:", out_market_daily)
                        return out_market_daily

                    # If DB is up-to-date, no download needed.
                    last_db_date = df_db["date"].max().date()
                    yesterday = datetime.utcnow().date() - timedelta(days=1)
                    if last_db_date >= yesterday and not force:
                        save_market_daily_parquet(df_db, out_market_daily)
                        print("[MARKET] DB cache is up-to-date (no download).")
                        print(f"[MARKET] Rows: {len(df_db)} | "
                              f"Range: {df_db['date'].min().date()} -> {df_db['date'].max().date()}")
                        print("[MARKET] Saved market_daily:", out_market_daily)
                        return out_market_daily

                    # Otherwise download only from DB tail month onward.
                    download_1d_auto(
                        symbol=symbol,
                        market_type=market_type,
                        data_root=ctx.data_root,
                        auto_start=False,
                        start_year=last_db_date.year,
                        start_month=last_db_date.month,
                        end_year=market_cfg.get("end_year"),
                        end_month=market_cfg.get("end_month"),
                        progress=True,
                        verbose=True,
                        print_urls=False,
                        daily_tail=True,
                    )
            except Exception as e:
                print(f"[WARN] DB cache failed: {e}")

    # Download/update raw archives (monthly + daily tail)
    if df_db is None or df_db.empty:
        download_1d_auto(
            symbol=symbol,
            market_type=market_type,
            data_root=ctx.data_root,
            auto_start=bool(market_cfg.get("auto_start", True)),
            start_year=int(market_cfg.get("start_year", 2010)),
            start_month=int(market_cfg.get("start_month", 1)),
            end_year=market_cfg.get("end_year"),
            end_month=market_cfg.get("end_month"),
            progress=True,
            verbose=True,
            print_urls=False,
            daily_tail=True,
        )

    # Parse raw ZIPs into OHLCV parquet
    parse_klines_folder_1d_ohlcv(
        input_dir=ctx.raw_dir,
        output_path=out_ohlcv,
        progress=True,
        verbose=True,
    )
    df_bin = pd.read_parquet(out_ohlcv)

    # Merge with archived CSV if it exists
    archive_csv = ctx.data_root / "raw" / "Bitcoin Historical Data.csv"
    if archive_csv.exists():
        df_arch = load_bitcoin_historical_csv(archive_csv, progress=True, verbose=True)
        df_all = _merge_archive_and_binance(df_arch, df_bin, prefer_binance=prefer_binance)
    else:
        df_all = df_bin.copy()

    df_all["date"] = pd.to_datetime(df_all["date"])
    df_market = df_all[["date", "close"]].copy()
    df_market = df_market.sort_values("date").reset_index(drop=True)

    if data_start:
        df_market = df_market[df_market["date"] >= pd.Timestamp(data_start)].reset_index(drop=True)

    # Merge DB cache if configured (DB wins on duplicates)
    if df_db is not None and not df_db.empty:
        df_market = pd.concat([df_market, df_db], ignore_index=True)
        df_market = df_market.drop_duplicates(subset=["date"], keep="last")
        df_market = df_market.sort_values("date").reset_index(drop=True)

    # Save to DB (upsert) if enabled
    if write_db and bool(market_cfg.get("use_db_cache", False)):
        db_url = (ctx.cfg_db.get("db") or {}).get("url", "")
        if db_url:
            try:
                df_db_max = None
                if df_db is not None and not df_db.empty:
                    df_db_max = pd.to_datetime(df_db["date"]).max().date()

                df_to_save = df_market.copy()
                df_to_save["date"] = pd.to_datetime(df_to_save["date"]).dt.date
                if df_db_max is not None:
                    df_to_save = df_to_save[df_to_save["date"] > df_db_max]

                if not df_to_save.empty:
                    from src.market.loader import insert_market_daily_psql

                    insert_market_daily_psql(df_to_save, ctx.subject.subject_id, db_url)
                    print(f"[DB] Upserted {len(df_to_save)} rows into market_daily")
                else:
                    print("[DB] No new rows to upsert")
            except Exception as e:
                print(f"[WARN] DB save failed: {e}")

    save_market_daily_parquet(df_market, out_market_daily)
    print("[MARKET] Built market_daily from files.")
    if not df_market.empty:
        print(f"[MARKET] Rows: {len(df_market)} | "
              f"Range: {df_market['date'].min().date()} -> {df_market['date'].max().date()}")
    print("[MARKET] Saved market_daily:", out_market_daily)
    return out_market_daily
