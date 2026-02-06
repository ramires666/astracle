"""
Market data updater for production automation.

This module updates the local parquet source of truth used by the frontend and
model retraining:
- data/market/processed/BTC_full_ohlcv_daily.parquet
- data/market/processed/BTC_full_market_daily.parquet

Update priority:
1) CoinGecko daily prices (primary)
2) Binance archive pipeline (fallback only)
"""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from production_dev.coingecko_client import fetch_btc_daily_prices_usd
from src.common.config import load_yaml, load_subjects
from src.market.archive_csv import load_bitcoin_historical_csv
from src.market.downloader import download_1d_auto
from src.market.parser import parse_klines_folder_1d_ohlcv


PROJECT_ROOT = Path(__file__).parent.parent


def _resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _load_last_date(path: Path) -> Optional[date]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["date"])
        if df.empty:
            return None
        return pd.to_datetime(df["date"]).max().date()
    except Exception:
        return None


def _merge_archive_and_source(
    df_archive: pd.DataFrame,
    df_source: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    """
    Merge archive + source and prefer source values on overlapping dates.
    """
    df_archive = df_archive.copy()
    df_source = df_source.copy()
    df_archive["source"] = "archive"
    df_source["source"] = source_name

    all_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "count",
        "change_pct",
        "source",
    ]
    for col in all_cols:
        if col not in df_archive.columns:
            df_archive[col] = pd.NA
        if col not in df_source.columns:
            df_source[col] = pd.NA

    df_archive = df_archive[all_cols].copy()
    df_source = df_source[all_cols].copy()

    merged = pd.concat([df_archive, df_source], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"]).dt.date

    merged["priority"] = merged["source"].map({"archive": 0, source_name: 1}).fillna(0)
    merged = (
        merged.sort_values(["date", "priority"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
        .drop(columns=["priority"])
    )
    return merged


def _merge_preserve_existing_history(
    df_existing: pd.DataFrame,
    df_updated: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge two full OHLCV tables and prefer rows from `df_updated` on overlap.

    Why this is needed:
    - Incremental upstream fetches (especially API-limited providers) may cover
      only a recent window.
    - Without this merge, a partial update can accidentally drop older periods.
    """
    all_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "count",
        "change_pct",
        "source",
    ]

    old = df_existing.copy()
    new = df_updated.copy()

    for col in all_cols:
        if col not in old.columns:
            old[col] = pd.NA
        if col not in new.columns:
            new[col] = pd.NA

    old = old[all_cols].copy()
    new = new[all_cols].copy()
    old["date"] = pd.to_datetime(old["date"]).dt.date
    new["date"] = pd.to_datetime(new["date"]).dt.date
    old["_priority"] = 0
    new["_priority"] = 1

    merged = pd.concat([old, new], ignore_index=True)
    merged = (
        merged.sort_values(["date", "_priority"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
        .drop(columns=["_priority"])
    )
    return merged


def _load_coingecko_daily_close_frame(
    previous_max_date: Optional[date],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load daily close series from CoinGecko.
    """
    today = datetime.utcnow().date()
    if previous_max_date is None:
        days_param: int | str = "max"
    else:
        gap_days = max(1, (today - previous_max_date).days)
        # Keep a margin so late corrections on recent days are captured too.
        days_param = max(30, min(365, gap_days + 7))

    if verbose:
        print(f"[MARKET-UPDATE] Fetching CoinGecko daily prices (days={days_param})...")

    series = fetch_btc_daily_prices_usd(days=days_param, timeout=20)
    if not series:
        raise ValueError("CoinGecko returned empty daily series")

    df = pd.DataFrame(series, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).copy()
    # Exclude "today" because it is usually an incomplete day candle.
    # Keeping only closed days makes history stable and reproducible.
    today = datetime.utcnow().date()
    df = df[df["date"] < today].copy()

    # Keep OHLCV schema compatible for parquet consumers.
    df["open"] = pd.NA
    df["high"] = pd.NA
    df["low"] = pd.NA
    df["volume"] = pd.NA
    df["quote_volume"] = pd.NA
    df["taker_buy_volume"] = pd.NA
    df["taker_buy_quote_volume"] = pd.NA
    df["count"] = pd.NA
    df["change_pct"] = pd.NA
    return df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def _load_binance_fallback_frame(
    symbol: str,
    market_type: str,
    data_root: Path,
    raw_dir: Path,
    processed_dir: Path,
    archive_csv: Path,
    start_year: int,
    start_month: int,
    now: datetime,
    cfg_market: Dict[str, Any],
    progress: bool,
    verbose: bool,
) -> tuple[pd.DataFrame, int]:
    """
    Legacy fallback path via Binance archives.
    """
    binance_ohlcv_path = processed_dir / f"{symbol}_ohlcv_daily.parquet"

    if verbose:
        print("[MARKET-UPDATE][FALLBACK] Downloading Binance 1d archives...")
    touched_archives = download_1d_auto(
        symbol=symbol,
        market_type=market_type,
        data_root=data_root,
        auto_start=False,
        start_year=start_year,
        start_month=start_month,
        end_year=cfg_market.get("end_year") or now.year,
        end_month=cfg_market.get("end_month") or now.month,
        progress=progress,
        verbose=verbose,
        print_urls=False,
        daily_tail=True,
    )

    if verbose:
        print("[MARKET-UPDATE][FALLBACK] Parsing Binance OHLCV...")
    parse_klines_folder_1d_ohlcv(
        input_dir=raw_dir,
        output_path=binance_ohlcv_path,
        progress=progress,
        verbose=verbose,
    )
    df_binance = pd.read_parquet(binance_ohlcv_path)

    if archive_csv.exists():
        if verbose:
            print("[MARKET-UPDATE][FALLBACK] Merging archive + Binance...")
        df_archive = load_bitcoin_historical_csv(
            archive_csv,
            progress=progress,
            verbose=verbose,
        )
        df_full = _merge_archive_and_source(df_archive, df_binance, source_name="binance")
    else:
        if verbose:
            print("[MARKET-UPDATE][FALLBACK] Archive CSV not found, using Binance only.")
        df_full = df_binance.copy()
        df_full["date"] = pd.to_datetime(df_full["date"]).dt.date
        df_full = df_full.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    return df_full, int(len(touched_archives))


def update_full_market_data(
    project_root: Path = PROJECT_ROOT,
    progress: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Update local full market parquets used by production.

    Returns:
        Dict with status, previous/new max date and row counts.
    """
    cfg_market = load_yaml(project_root / "configs" / "market.yaml").get("market", {})
    subjects, active_subject_id = load_subjects(project_root / "configs" / "subjects.yaml")
    active_subject = subjects.get(active_subject_id)

    symbol = cfg_market.get("symbol") or (active_subject.symbol if active_subject else "BTCUSDT")
    market_type = cfg_market.get("market_type", "futures_um")
    data_root = _resolve_path(project_root, cfg_market.get("data_root", "data/market"))

    raw_dir = data_root / "raw" / "klines_1d"
    processed_dir = data_root / "processed"
    archive_csv = data_root / "raw" / "Bitcoin Historical Data.csv"

    full_ohlcv_path = processed_dir / "BTC_full_ohlcv_daily.parquet"
    full_market_path = processed_dir / "BTC_full_market_daily.parquet"
    full_market_csv_path = processed_dir / "BTC_full_market_daily.csv"

    previous_max_date = _load_last_date(full_market_path)

    now = datetime.utcnow()
    if previous_max_date is not None:
        start_year = previous_max_date.year
        start_month = previous_max_date.month
    else:
        start_year = int(cfg_market.get("start_year", 2010))
        start_month = int(cfg_market.get("start_month", 1))

    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    allow_binance_fallback = os.getenv("MARKET_UPDATE_ALLOW_BINANCE_FALLBACK", "0").strip().lower() not in {
        "0",
        "false",
        "no",
    }

    update_source = "coingecko"
    fallback_used = False
    coingecko_error: Optional[str] = None
    touched_archives = 0

    try:
        df_source = _load_coingecko_daily_close_frame(previous_max_date=previous_max_date, verbose=verbose)
    except Exception as e:
        coingecko_error = str(e)
        if not allow_binance_fallback:
            raise RuntimeError(f"CoinGecko update failed and Binance fallback is disabled: {e}") from e
        fallback_used = True
        update_source = "binance"
        if verbose:
            print(f"[MARKET-UPDATE] CoinGecko failed, switching to Binance fallback: {e}")
        df_source, touched_archives = _load_binance_fallback_frame(
            symbol=symbol,
            market_type=market_type,
            data_root=data_root,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            archive_csv=archive_csv,
            start_year=start_year,
            start_month=start_month,
            now=now,
            cfg_market=cfg_market,
            progress=progress,
            verbose=verbose,
        )

    if archive_csv.exists():
        if verbose:
            print(f"[MARKET-UPDATE] Merging archive + {update_source}...")
        df_archive = load_bitcoin_historical_csv(
            archive_csv,
            progress=progress,
            verbose=verbose,
        )
        df_full = _merge_archive_and_source(df_archive, df_source, source_name=update_source)
    else:
        if verbose:
            print(f"[MARKET-UPDATE] Archive CSV not found, using {update_source} only.")
        df_full = df_source.copy()
        df_full["date"] = pd.to_datetime(df_full["date"]).dt.date
        df_full = df_full.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    if "close" not in df_full.columns:
        raise ValueError("Merged market dataset has no 'close' column.")

    # Safety net: never lose already saved historical days when update source is partial.
    if full_ohlcv_path.exists():
        try:
            df_existing_full = pd.read_parquet(full_ohlcv_path)
            before_rows = int(len(df_full))
            df_full = _merge_preserve_existing_history(df_existing_full, df_full)
            if verbose:
                print(
                    "[MARKET-UPDATE] Preserved existing history:",
                    f"rows_before={before_rows}",
                    f"rows_after={len(df_full)}",
                )
        except Exception as e:
            if verbose:
                print(f"[MARKET-UPDATE] Warning: failed to merge existing history safety net: {e}")

    # Keep only closed daily candles in final storage.
    today = datetime.utcnow().date()
    df_full["date"] = pd.to_datetime(df_full["date"]).dt.date
    df_full = df_full[df_full["date"] < today].copy()

    full_ohlcv_path.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(full_ohlcv_path, index=False)
    df_full[["date", "close"]].to_parquet(full_market_path, index=False)
    # Keep an optional CSV mirror so service can run in CSV-only mode too.
    df_full[["date", "close"]].to_csv(full_market_csv_path, index=False)

    new_max_date = _load_last_date(full_market_path)
    status = "up_to_date"
    rows_added = 0
    if previous_max_date is None and new_max_date is not None:
        status = "updated"
        rows_added = int(len(df_full))
    elif previous_max_date is not None and new_max_date is not None and new_max_date > previous_max_date:
        status = "updated"
        rows_added = int((new_max_date - previous_max_date).days)

    return {
        "status": status,
        "symbol": symbol,
        "market_type": market_type,
        "previous_max_date": previous_max_date.isoformat() if previous_max_date else None,
        "new_max_date": new_max_date.isoformat() if new_max_date else None,
        "rows_total": int(len(df_full)),
        "rows_added_estimate": rows_added,
        "update_source": update_source,
        "fallback_used": fallback_used,
        "coingecko_error": coingecko_error,
        "archives_touched": int(touched_archives),
        "full_market_path": str(full_market_path),
        "full_market_csv_path": str(full_market_csv_path),
        "full_ohlcv_path": str(full_ohlcv_path),
    }
