"""
Transit calculations for the astro package.

This module handles:
- Transit-to-natal aspect calculations
- Natal chart body positions
"""
import pandas as pd
from typing import List
from tqdm import tqdm

from src.astro.engine.settings import AstroSettings
from src.astro.engine.calculator import calculate_bodies
from src.astro.engine.aspects import calculate_transit_aspects
from src.astro.engine.models import AspectConfig, BodyPosition

from .aspects import scale_aspects
from .bodies import parse_birth_dt_utc


def calculate_transits_for_dates(
    bodies_by_date: dict,
    natal_bodies: List[BodyPosition],
    settings: AstroSettings,
    orb_mult: float = 1.0,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Calculate transit-to-natal aspects for all dates.
    
    Transits show how current planet positions interact with 
    the natal chart positions.
    
    Args:
        bodies_by_date: Dict {date -> [BodyPosition]}
        natal_bodies: Natal chart body positions
        settings: AstroSettings with aspect configs
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
        birth_dt_str: Birth datetime string (ISO format, e.g., "2009-10-10T12:00:00")
        settings: AstroSettings with body configurations
        center: Coordinate center ('geo' or 'helio')
    
    Returns:
        List of natal body positions
    """
    birth_dt = parse_birth_dt_utc(birth_dt_str)
    return calculate_bodies(birth_dt, settings.bodies, center=center)
