"""
CLI: load market data and astro data into Postgres.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Optional

import pandas as pd

# Add project root to sys.path so src.* imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_yaml, load_subjects
from src.market.loader import load_market_daily_parquet, insert_market_daily_psql
from src.astro.engine.settings import AstroSettings
from src.astro.engine.calculator import set_ephe_path, calculate_daily_bodies, calculate_bodies
from src.astro.engine.aspects import calculate_aspects, calculate_transit_aspects
from src.db.loader import (
    upsert_subject_psql,
    insert_astro_bodies_psql,
    insert_astro_aspects_psql,
    insert_natal_bodies_psql,
    insert_natal_aspects_psql,
    insert_transit_aspects_psql,
)


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

    return tqdm(items, desc=desc, unit="day")


def _read_db_url(path: Path) -> str:
    """
    Read db.yaml and return the connection URL.
    """
    cfg = load_yaml(path)
    if "db" not in cfg or "url" not in cfg["db"]:
        raise KeyError("configs/db.yaml must define db.url")
    return cfg["db"]["url"]


def _resolve_market_parquet(data_root: Path, subject_id: str, symbol: str) -> Path:
    """
    Find market parquet file in processed folder.
    """
    processed = data_root / "processed"
    p1 = processed / f"{subject_id}_market_daily.parquet"
    p2 = processed / f"{symbol}_market_daily.parquet"

    if p1.exists():
        return p1
    if p2.exists():
        return p2

    raise FileNotFoundError(f"Market parquet not found. Expected {p1} or {p2}")


def _parse_time_utc(value: str):
    """
    Convert HH:MM:SS string to time.
    """
    from datetime import datetime

    return datetime.strptime(value, "%H:%M:%S").time()


def _parse_birth_dt_utc(value: str):
    """
    Convert ISO datetime string to UTC datetime.
    Supports suffix "Z".
    """
    from datetime import datetime, timezone

    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _build_astro_daily(
    df_market: pd.DataFrame,
    settings: AstroSettings,
    time_utc,
    center: str,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute astro bodies and aspects for each market date.
    """
    bodies_rows = []
    aspects_rows = []

    dates = pd.to_datetime(df_market["date"]).dt.date
    print(f"[DB] Days in sample: {len(dates)}")

    for d in _iter_with_progress(dates, progress, desc="astro days"):
        bodies = calculate_daily_bodies(d, time_utc, settings.bodies, center=center)
        aspects = calculate_aspects(bodies, settings.aspects)

        for b in bodies:
            bodies_rows.append(
                {
                    "date": b.date,
                    "body": b.body,
                    "lon": b.lon,
                    "lat": b.lat,
                    "speed": b.speed,
                    "is_retro": b.is_retro,
                    "sign": b.sign,
                    "declination": b.declination,
                }
            )

        for a in aspects:
            aspects_rows.append(
                {
                    "date": a.date,
                    "p1": a.p1,
                    "p2": a.p2,
                    "aspect": a.aspect,
                    "orb": a.orb,
                    "is_exact": a.is_exact,
                    "is_applying": a.is_applying,
                }
            )

    df_bodies = pd.DataFrame(bodies_rows)
    df_aspects = pd.DataFrame(aspects_rows)

    return df_bodies, df_aspects


def _build_natal(
    subject,
    settings: AstroSettings,
    center: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Compute natal bodies and aspects for subject birth datetime.
    """
    birth_dt = _parse_birth_dt_utc(subject.birth_dt_utc)
    print(f"[DB] Natal datetime: {birth_dt.isoformat()}")

    natal_bodies = calculate_bodies(birth_dt, settings.bodies, center=center)
    natal_aspects = calculate_aspects(natal_bodies, settings.aspects)

    bodies_rows = []
    for b in natal_bodies:
        bodies_rows.append(
            {
                "body": b.body,
                "lon": b.lon,
                "lat": b.lat,
                "speed": b.speed,
                "is_retro": b.is_retro,
                "sign": b.sign,
                "declination": b.declination,
            }
        )

    aspects_rows = []
    for a in natal_aspects:
        aspects_rows.append(
            {
                "p1": a.p1,
                "p2": a.p2,
                "aspect": a.aspect,
                "orb": a.orb,
            }
        )

    df_bodies = pd.DataFrame(bodies_rows)
    df_aspects = pd.DataFrame(aspects_rows)

    return df_bodies, df_aspects, natal_bodies


def _build_transit_aspects(
    df_market: pd.DataFrame,
    settings: AstroSettings,
    time_utc,
    center: str,
    natal_bodies: list,
    progress: bool,
) -> pd.DataFrame:
    """
    Compute transit aspects (transit -> natal) for each date.
    """
    rows = []
    dates = pd.to_datetime(df_market["date"]).dt.date
    print(f"[DB] Transit days: {len(dates)}")

    for d in _iter_with_progress(dates, progress, desc="transit days"):
        transit_bodies = calculate_daily_bodies(d, time_utc, settings.bodies, center=center)
        hits = calculate_transit_aspects(transit_bodies, natal_bodies, settings.aspects)

        for h in hits:
            rows.append(
                {
                    "date": h.date,
                    "transit_body": h.transit_body,
                    "natal_body": h.natal_body,
                    "aspect": h.aspect,
                    "orb": h.orb,
                    "is_exact": h.is_exact,
                    "is_applying": h.is_applying,
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(
        description="Load market + astro data into Postgres",
    )
    parser.add_argument("--config-db", default="configs/db.yaml", help="Path to db.yaml")
    parser.add_argument("--config-market", default="configs/market.yaml", help="Path to market.yaml")
    parser.add_argument("--config-astro", default="configs/astro.yaml", help="Path to astro.yaml")
    parser.add_argument("--config-subjects", default="configs/subjects.yaml", help="Path to subjects.yaml")
    parser.add_argument("--subject-id", default=None, help="Subject ID (defaults to active)")
    parser.add_argument("--market-parquet", default=None, help="Explicit parquet path")
    parser.add_argument("--skip-astro", action="store_true", help="Skip astro calculations and load")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress",
    )
    parser.add_argument(
        "--natal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute and load natal data",
    )
    parser.add_argument(
        "--transits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute and load transit aspects to natal",
    )

    args = parser.parse_args()

    db_url = _read_db_url(Path(args.config_db))
    cfg_market = load_yaml(args.config_market)
    cfg_astro = load_yaml(args.config_astro)
    subjects, active_id = load_subjects(args.config_subjects)
    subject_id = args.subject_id or active_id

    if subject_id not in subjects:
        raise ValueError(f"subject_id={subject_id} not found in configs/subjects.yaml")
    subject = subjects[subject_id]

    market_cfg = cfg_market["market"]
    data_root = Path(market_cfg["data_root"])

    if args.market_parquet:
        market_parquet = Path(args.market_parquet)
    else:
        market_parquet = _resolve_market_parquet(
            data_root=data_root,
            subject_id=subject.subject_id,
            symbol=subject.symbol,
        )

    print("[DB] Run params:")
    print(f"[DB]  db_url         = {db_url}")
    print(f"[DB]  subject_id     = {subject.subject_id}")
    print(f"[DB]  symbol         = {subject.symbol}")
    print(f"[DB]  market_parquet = {market_parquet}")
    print(f"[DB]  skip_astro     = {args.skip_astro}")
    print(f"[DB]  natal          = {args.natal}")
    print(f"[DB]  transits       = {args.transits}")
    print(f"[DB]  progress       = {args.progress}")

    # --- subjects ---
    print("[DB] Upserting subjects...")
    upsert_subject_psql(subject, db_url)

    # --- market_daily ---
    print("[DB] Loading market_daily...")
    df_market = load_market_daily_parquet(market_parquet)
    insert_market_daily_psql(df_market, subject.subject_id, db_url)
    print(f"[DB] market_daily: loaded {len(df_market)} rows")

    if args.skip_astro:
        print("[DB] Astro data skipped.")
        return 0

    # --- astro settings ---
    astro_cfg = cfg_astro["astro"]
    set_ephe_path(astro_cfg["ephe_path"])
    settings = AstroSettings(
        bodies_path=Path(astro_cfg["bodies_path"]),
        aspects_path=Path(astro_cfg["aspects_path"]),
    )
    time_utc = _parse_time_utc(astro_cfg["daily_time_utc"])
    center = astro_cfg.get("center", "geo")

    # --- astro ---
    print("[DB] Computing astro data...")
    df_bodies, df_aspects = _build_astro_daily(
        df_market=df_market,
        settings=settings,
        time_utc=time_utc,
        center=center,
        progress=args.progress,
    )

    print("[DB] Loading astro_bodies_daily...")
    insert_astro_bodies_psql(df_bodies, subject.subject_id, db_url)
    print(f"[DB] astro_bodies_daily: loaded {len(df_bodies)} rows")

    print("[DB] Loading astro_aspects_daily...")
    insert_astro_aspects_psql(df_aspects, subject.subject_id, db_url)
    print(f"[DB] astro_aspects_daily: loaded {len(df_aspects)} rows")

    # --- natal ---
    natal_bodies = []
    if args.natal:
        print("[DB] Computing natal data...")
        df_natal_bodies, df_natal_aspects, natal_bodies = _build_natal(
            subject, settings, center
        )

        print("[DB] Loading natal_bodies...")
        insert_natal_bodies_psql(df_natal_bodies, subject.subject_id, db_url)
        print(f"[DB] natal_bodies: loaded {len(df_natal_bodies)} rows")

        print("[DB] Loading natal_aspects...")
        insert_natal_aspects_psql(df_natal_aspects, subject.subject_id, db_url)
        print(f"[DB] natal_aspects: loaded {len(df_natal_aspects)} rows")

    # --- transits ---
    if args.transits:
        if not natal_bodies:
            print("[DB] Transits require natal bodies. Enable --natal.")
        else:
            print("[DB] Computing transit aspects...")
            df_transits = _build_transit_aspects(
                df_market=df_market,
                settings=settings,
                time_utc=time_utc,
                center=center,
                natal_bodies=natal_bodies,
                progress=args.progress,
            )

            print("[DB] Loading transit_aspects_daily...")
            insert_transit_aspects_psql(df_transits, subject.subject_id, db_url)
            print(f"[DB] transit_aspects_daily: loaded {len(df_transits)} rows")

    print("[DB] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
