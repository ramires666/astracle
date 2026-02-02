"""
Astro engine module for RESEARCH pipeline.
Calculates planetary positions, aspects, and transits on-the-fly.
No caching to DB - calculated fast each time.

TODO: Save best grid search results (orb, birth dates, etc.)
TODO: Add moon phases and other planet phases
TODO: Add houses when doing birth date grid search
"""
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timezone
from typing import List, Optional
from tqdm import tqdm

# Import from existing src modules
from src.astro.engine.settings import AstroSettings
from src.astro.engine.calculator import set_ephe_path, calculate_daily_bodies, calculate_bodies
from src.astro.engine.aspects import calculate_aspects, calculate_transit_aspects
from src.astro.engine.models import AspectConfig, BodyPosition

from .config import cfg, resolve_path


def init_ephemeris() -> AstroSettings:
    """
    Initialize Swiss Ephemeris and return AstroSettings.
    
    Returns:
        AstroSettings with bodies and aspects configurations
    """
    astro_cfg = cfg.get_astro_config()
    
    # Set ephemeris path
    set_ephe_path(str(astro_cfg["ephe_path"]))
    
    # Create settings
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


def scale_aspects(aspects: List[AspectConfig], orb_mult: float) -> List[AspectConfig]:
    """Scale aspect orbs by multiplier."""
    return [
        AspectConfig(name=a.name, degree=a.degree, orb=float(a.orb) * orb_mult)
        for a in aspects
    ]


def calculate_bodies_for_dates(
    dates: pd.Series,
    settings: AstroSettings,
    time_utc: Optional[time] = None,
    center: str = "geo",
    progress: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Calculate body positions for a range of dates.
    
    Args:
        dates: Series of dates to calculate for
        settings: AstroSettings with body configurations
        time_utc: Time of day for calculations (default from config)
        center: Coordinate center ('geo' or 'helio')
        progress: Show progress bar
    
    Returns:
        Tuple of (df_bodies DataFrame, bodies_by_date dict for aspect calculations)
    """
    astro_cfg = cfg.get_astro_config()
    time_utc = time_utc or parse_time_utc(astro_cfg["daily_time_utc"])
    
    bodies_rows = []
    bodies_by_date = {}
    
    date_list = pd.to_datetime(dates).dt.date
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


def calculate_aspects_for_dates(
    bodies_by_date: dict,
    settings: AstroSettings,
    orb_mult: float = 1.0,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Calculate aspects for all dates.
    
    Args:
        bodies_by_date: Dict mapping date -> list of BodyPosition
        settings: AstroSettings with aspect configurations
        orb_mult: Orb multiplier (1.0 = default orbs)
        progress: Show progress bar
    
    Returns:
        DataFrame with aspect data
    """
    aspects_cfg = scale_aspects(settings.aspects, orb_mult)
    aspects_rows = []
    
    iterator = tqdm(bodies_by_date.items(), desc=f"Calculating aspects (orb={orb_mult})") if progress else bodies_by_date.items()
    
    for d, bodies in iterator:
        aspects = calculate_aspects(bodies, aspects_cfg)
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
    
    return pd.DataFrame(aspects_rows)


def calculate_transits_for_dates(
    bodies_by_date: dict,
    natal_bodies: List[BodyPosition],
    settings: AstroSettings,
    orb_mult: float = 1.0,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Calculate transit-to-natal aspects for all dates.
    
    Args:
        bodies_by_date: Dict mapping date -> list of BodyPosition
        natal_bodies: Natal chart body positions
        settings: AstroSettings with aspect configurations
        orb_mult: Orb multiplier
        progress: Show progress bar
    
    Returns:
        DataFrame with transit aspect data
    """
    aspects_cfg = scale_aspects(settings.aspects, orb_mult)
    transit_rows = []
    
    iterator = tqdm(bodies_by_date.items(), desc=f"Calculating transits (orb={orb_mult})") if progress else bodies_by_date.items()
    
    for d, bodies in iterator:
        hits = calculate_transit_aspects(bodies, natal_bodies, aspects_cfg)
        for h in hits:
            transit_rows.append({
                "date": h.date,
                "transit_body": h.transit_body,
                "natal_body": h.natal_body,
                "aspect": h.aspect,
                "orb": h.orb,
                "is_exact": h.is_exact,
                "is_applying": h.is_applying,
            })
    
    return pd.DataFrame(transit_rows)


def get_natal_bodies(
    birth_dt_str: str,
    settings: AstroSettings,
    center: str = "geo",
) -> List[BodyPosition]:
    """
    Calculate natal chart body positions.
    
    Args:
        birth_dt_str: Birth datetime string (ISO format)
        settings: AstroSettings with body configurations
        center: Coordinate center
    
    Returns:
        List of natal body positions
    """
    birth_dt = parse_birth_dt_utc(birth_dt_str)
    return calculate_bodies(birth_dt, settings.bodies, center=center)


# TODO: Add moon phase calculation
def calculate_moon_phases(dates: pd.Series) -> pd.Series:
    """
    Calculate moon phases for given dates.
    
    Returns:
        Series with moon phase values (0-1 representing phase)
    
    NOTE: This is a placeholder. Needs Swiss Ephemeris implementation.
    """
    # Placeholder - to be implemented
    return pd.Series(np.nan, index=dates.index)
