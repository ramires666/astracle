"""
MVP pipeline: market (1d) -> astro data -> features -> labels.

NOTE: this is a prototype. DB can be plugged in later.
"""

from __future__ import annotations

from datetime import datetime, time
from pathlib import Path
import sys

# Add project root to sys.path so this file can be run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Iterable, List

import pandas as pd

from src.common.config import load_yaml, load_subjects
from src.market.downloader import download_1d_auto
from src.market.parser import parse_klines_folder_1d
from src.market.loader import load_market_daily_parquet, load_market_daily_psql
from src.astro.engine.settings import AstroSettings
from src.astro.engine.calculator import set_ephe_path, calculate_bodies, calculate_daily_bodies
from src.astro.engine.aspects import calculate_aspects, calculate_transit_aspects
from src.features.builder import build_features_daily
from src.labeling.oracle import create_oracle_labels, estimate_threshold_for_move_balance
from src.visualization.dxcharts_report import write_dxcharts_report


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

    return tqdm(items, desc=desc, unit="step")


def _parse_time_utc(value: str) -> time:
    """
    Convert HH:MM:SS string to time.
    """
    return datetime.strptime(value, "%H:%M:%S").time()


def _parse_birth_dt_utc(value: str) -> datetime:
    """
    Convert ISO datetime string to UTC datetime.
    Supports suffix "Z".
    """
    from datetime import timezone

    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def run_mvp():
    """
    MVP entry point.
    """
    # --- Configs ---
    cfg_market = load_yaml("configs/market.yaml")
    cfg_astro = load_yaml("configs/astro.yaml")
    cfg_labels = load_yaml("configs/labels.yaml")

    subjects, active_id = load_subjects("configs/subjects.yaml")
    subject = subjects[active_id]

    # --- Market: download and parse ---
    market_cfg = cfg_market["market"]
    data_root = Path(market_cfg["data_root"])

    raw_dir = data_root / "raw" / "klines_1d"
    processed_dir = data_root / "processed"
    market_parquet = processed_dir / f"{subject.subject_id}_market_daily.parquet"

    df_market = None
    use_db_cache = bool(market_cfg.get("use_db_cache", False))
    if use_db_cache:
        try:
            cfg_db = load_yaml("configs/db.yaml")
            db_url = cfg_db.get("db", {}).get("url", "")
            if db_url:
                df_market = load_market_daily_psql(subject.subject_id, db_url)
                if df_market.empty:
                    df_market = None
                else:
                    print(f"[MVP] Using market data from DB: {len(df_market)} rows")
            else:
                print("[WARN] configs/db.yaml missing db.url, DB cache disabled.")
        except Exception as e:
            print(f"[WARN] DB cache failed, fallback to files. Reason: {e}")
            df_market = None

    use_parquet_cache = bool(market_cfg.get("use_parquet_cache", True))
    if df_market is None and use_parquet_cache and market_parquet.exists():
        print(f"[MVP] Using cached parquet: {market_parquet}")
        df_market = load_market_daily_parquet(market_parquet)

    if df_market is None:
        print("[MVP] Downloading 1d data...")
        download_1d_auto(
            symbol=market_cfg["symbol"],
            market_type=market_cfg["market_type"],
            data_root=data_root,
            auto_start=market_cfg["auto_start"],
            start_year=market_cfg["start_year"],
            start_month=market_cfg["start_month"],
            end_year=market_cfg["end_year"],
            end_month=market_cfg["end_month"],
            progress=True,
            verbose=True,
            print_urls=False,
        )

        print("[MVP] Parsing ZIP -> parquet...")
        parse_klines_folder_1d(raw_dir, market_parquet, progress=True, verbose=True)

        df_market = load_market_daily_parquet(market_parquet)

    # --- Astro settings ---
    astro_cfg = cfg_astro["astro"]
    set_ephe_path(astro_cfg["ephe_path"])
    settings = AstroSettings(
        bodies_path=Path(astro_cfg["bodies_path"]),
        aspects_path=Path(astro_cfg["aspects_path"]),
    )
    time_utc = _parse_time_utc(astro_cfg["daily_time_utc"])
    center = astro_cfg.get("center", "geo")

    birth_dt = _parse_birth_dt_utc(subject.birth_dt_utc)
    natal_bodies = calculate_bodies(birth_dt, settings.bodies, center=center)

    # --- Bodies and aspects ---
    print("[MVP] Computing astro data...")
    bodies_rows: List[dict] = []
    aspects_rows: List[dict] = []
    transit_rows: List[dict] = []

    dates = pd.to_datetime(df_market["date"]).dt.date
    print(f"[MVP] Days in sample: {len(dates)}")
    for d in _iter_with_progress(dates, True, desc="astro days"):
        bodies = calculate_daily_bodies(d, time_utc, settings.bodies, center=center)
        aspects = calculate_aspects(bodies, settings.aspects)
        transit_hits = calculate_transit_aspects(bodies, natal_bodies, settings.aspects)

        for b in bodies:
            bodies_rows.append({
                "date": b.date,
                "body": b.body,
                "lon": b.lon,
                "lat": b.lat,
                "speed": b.speed,
                "is_retro": b.is_retro,
                "sign": b.sign,
                "declination": b.declination,
            })

        for a in aspects:
            aspects_rows.append({
                "date": a.date,
                "p1": a.p1,
                "p2": a.p2,
                "aspect": a.aspect,
                "orb": a.orb,
                "is_exact": a.is_exact,
                "is_applying": a.is_applying,
            })

        for h in transit_hits:
            transit_rows.append({
                "date": h.date,
                "transit_body": h.transit_body,
                "natal_body": h.natal_body,
                "aspect": h.aspect,
                "orb": h.orb,
                "is_exact": h.is_exact,
                "is_applying": h.is_applying,
            })

    df_bodies = pd.DataFrame(bodies_rows)
    df_aspects = pd.DataFrame(aspects_rows)
    df_transits = pd.DataFrame(transit_rows)

    astro_dir = data_root / "processed"
    df_bodies.to_parquet(astro_dir / f"{subject.subject_id}_astro_bodies.parquet", index=False)
    df_aspects.to_parquet(astro_dir / f"{subject.subject_id}_astro_aspects.parquet", index=False)
    df_transits.to_parquet(astro_dir / f"{subject.subject_id}_transit_aspects.parquet", index=False)

    # --- Features ---
    print("[MVP] Building features...")
    df_features = build_features_daily(df_bodies, df_aspects, df_transits)
    df_features.to_parquet(astro_dir / f"{subject.subject_id}_features.parquet", index=False)

    # --- Labels ---
    print("[MVP] Generating oracle labels...")
    labels_cfg = cfg_labels["labels"]
    price_mode = labels_cfg.get("price_mode", "log")
    sigma = int(labels_cfg["sigma"])
    threshold = float(labels_cfg.get("threshold", 0.001))
    binary_trend = bool(labels_cfg.get("binary_trend", False))
    binary_fallback = str(labels_cfg.get("binary_fallback", "up"))
    threshold_mode = str(labels_cfg.get("threshold_mode", "fixed")).strip().lower()
    target_move_share = float(labels_cfg.get("target_move_share", 0.5))
    threshold_min = labels_cfg.get("threshold_min", 0.0)
    if threshold_min is None:
        threshold_min = 0.0
    threshold_min = float(threshold_min)
    threshold_max = labels_cfg.get("threshold_max", None)
    if threshold_max is not None:
        threshold_max = float(threshold_max)

    if threshold_mode == "auto":
        threshold = estimate_threshold_for_move_balance(
            df_market,
            sigma=sigma,
            price_col="close",
            price_mode=price_mode,
            target_move_share=target_move_share,
            min_threshold=threshold_min,
            max_threshold=threshold_max,
        )
        print(f"[MVP] Auto threshold={threshold:.8f} (target_move_share={target_move_share:.2f})")

    df_labels = create_oracle_labels(
        df_market,
        sigma=sigma,
        threshold=threshold,
        price_col="close",
        price_mode=price_mode,
        binary_trend=binary_trend,
        binary_fallback=binary_fallback,
    )
    df_labels.to_parquet(astro_dir / f"{subject.subject_id}_labels.parquet", index=False)

    # --- Visual report ---
    print("[MVP] Generating dxcharts report...")
    report_path = data_root / "reports" / f"{subject.subject_id}_oracle.html"
    write_dxcharts_report(df_labels, report_path, title=f"Oracle Report: {subject.subject_id}")

    print("[MVP] Done.")


if __name__ == "__main__":
    run_mvp()
