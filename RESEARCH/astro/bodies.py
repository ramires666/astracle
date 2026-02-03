"""
Body position calculations for the astro package.

This module handles:
- Ephemeris initialization
- Planet position calculations (geocentric/heliocentric)
- Multi-coordinate mode support
"""
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timezone
from typing import List, Optional, Tuple
from tqdm import tqdm

from src.astro.engine.settings import AstroSettings
from src.astro.engine.calculator import set_ephe_path, calculate_daily_bodies
from src.astro.engine.models import BodyPosition

from ..config import cfg


def init_ephemeris() -> AstroSettings:
    """
    Initialize Swiss Ephemeris and return AstroSettings.
    
    Returns:
        AstroSettings with bodies and aspects configurations
    """
    astro_cfg = cfg.get_astro_config()
    set_ephe_path(str(astro_cfg["ephe_path"]))
    
    settings = AstroSettings(
        bodies_path=astro_cfg["bodies_path"],
        aspects_path=astro_cfg["aspects_path"],
    )
    return settings


def parse_birth_dt_utc(value: str) -> datetime:
    """Parse birth datetime string to UTC datetime."""
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_time_utc(value: str) -> time:
    """Parse time string (HH:MM:SS) to time object."""
    return datetime.strptime(value, "%H:%M:%S").time()


def calculate_bodies_for_dates(
    dates: pd.Series,
    settings: AstroSettings,
    time_utc: Optional[time] = None,
    center: str = "geo",
    progress: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    Calculate planet positions for a range of dates.
    
    Args:
        dates: Series of dates to calculate for
        settings: AstroSettings with body configurations
        time_utc: Time of day for calculations (default from config)
        center: Coordinate center ('geo' or 'helio')
        progress: Show progress bar
    
    Returns:
        Tuple of (df_bodies DataFrame, bodies_by_date dict)
    """
    # Get time from config
    astro_cfg = cfg.get_astro_config()
    time_utc = time_utc or parse_time_utc(astro_cfg["daily_time_utc"])
    
    bodies_rows = []
    bodies_by_date = {}
    
    # Convert dates to date objects
    date_list = pd.to_datetime(pd.Series(dates)).dt.date
    iterator = tqdm(date_list, desc="Calculating bodies") if progress else date_list
    
    for d in iterator:
        bodies = calculate_daily_bodies(d, time_utc, settings.bodies, center=center)
        bodies_by_date[d] = bodies
        
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
    
    df_bodies = pd.DataFrame(bodies_rows)
    return df_bodies, bodies_by_date


def calculate_bodies_for_dates_multi(
    dates: pd.Series,
    settings: AstroSettings,
    coord_mode: str = "geo",
    time_utc: Optional[time] = None,
    progress: bool = True,
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    Calculate planet positions in different coordinate systems.
    
    Modes:
        - "geo": Geocentric only (Earth at center)
        - "helio": Heliocentric only (Sun at center)
        - "both": Both systems combined
    
    Args:
        dates: Series of dates
        settings: AstroSettings
        coord_mode: 'geo', 'helio', or 'both'
        time_utc: Time of day (optional)
        progress: Show progress bar
        
    Returns:
        Tuple of (df_bodies, geo_bodies_by_date, helio_bodies_by_date)
    """
    valid_modes = ["geo", "helio", "both"]
    if coord_mode not in valid_modes:
        raise ValueError(f"coord_mode must be one of {valid_modes}, got '{coord_mode}'")
    
    geo_bodies_by_date = {}
    helio_bodies_by_date = {}
    all_dfs = []
    
    # Geocentric
    if coord_mode in ["geo", "both"]:
        if progress:
            print("📍 Расчёт ГЕОЦЕНТРИЧЕСКИХ координат (Земля в центре)...")
        df_geo, geo_bodies_by_date = calculate_bodies_for_dates(
            dates, settings, time_utc=time_utc, center="geo", progress=progress
        )
        if coord_mode == "both":
            df_geo["body"] = "geo_" + df_geo["body"].astype(str)
        all_dfs.append(df_geo)
    
    # Heliocentric
    if coord_mode in ["helio", "both"]:
        if progress:
            print("☀️ Расчёт ГЕЛИОЦЕНТРИЧЕСКИХ координат (Солнце в центре)...")
        df_helio, helio_bodies_by_date = calculate_bodies_for_dates(
            dates, settings, time_utc=time_utc, center="helio", progress=progress
        )
        df_helio["body"] = "helio_" + df_helio["body"].astype(str)
        all_dfs.append(df_helio)
    
    # Combine
    if len(all_dfs) == 1:
        df_bodies = all_dfs[0]
    else:
        df_bodies = pd.concat(all_dfs, ignore_index=True)
        if progress:
            print(f"✅ Объединено: {len(df_bodies)} записей из {len(all_dfs)} систем координат")
    
    return df_bodies, geo_bodies_by_date, helio_bodies_by_date
