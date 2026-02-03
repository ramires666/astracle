"""
Aspect calculations for the astro package.

This module handles:
- Angle precomputation for optimization
- Aspect calculation between planets
- Multi-coordinate aspect support
"""
import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm

from src.astro.engine.settings import AstroSettings
from src.astro.engine.aspects import calculate_aspects
from src.astro.engine.models import AspectConfig, BodyPosition

from ..numba_utils import (
    compute_pairwise_angles, 
    filter_aspects_by_orb, 
    is_aspect_applying,
)


def scale_aspects(aspects: List[AspectConfig], orb_mult: float) -> List[AspectConfig]:
    """Scale aspect orbs by multiplier."""
    return [
        AspectConfig(name=a.name, degree=a.degree, orb=float(a.orb) * orb_mult)
        for a in aspects
    ]


def precompute_angles_for_dates(
    bodies_by_date: dict,
    progress: bool = True,
) -> dict:
    """
    Precompute all pairwise angles between planets.
    
    This allows reusing angles for different orb values during grid search,
    providing ~3-5x speedup.
    
    Args:
        bodies_by_date: Dict {date: [BodyPosition, ...]}
        progress: Show progress bar
        
    Returns:
        Dict {date: {'body_names', 'longitudes', 'speeds', 'angles'}}
    """
    angles_cache = {}
    
    iterator = tqdm(bodies_by_date.items(), desc="Precomputing angles") if progress else bodies_by_date.items()
    
    for d, bodies in iterator:
        body_names = [b.body for b in bodies]
        longitudes = np.array([b.lon for b in bodies], dtype=np.float64)
        speeds = np.array([b.speed for b in bodies], dtype=np.float64)
        
        angles_matrix = compute_pairwise_angles(longitudes)
        
        angles_cache[d] = {
            'date': d,
            'body_names': body_names,
            'longitudes': longitudes,
            'speeds': speeds,
            'angles': angles_matrix,
        }
    
    return angles_cache


def calculate_aspects_from_cache(
    angles_cache: dict,
    settings: AstroSettings,
    orb_mult: float = 1.0,
    prefix: str = "",
    progress: bool = False,
) -> pd.DataFrame:
    """
    Fast aspect calculation from precomputed angles.
    
    Args:
        angles_cache: Result from precompute_angles_for_dates()
        settings: AstroSettings with aspect configs
        orb_mult: Orb multiplier
        prefix: Prefix for planet names
        progress: Show progress bar
        
    Returns:
        DataFrame with aspects
    """
    from src.astro.engine.aspects import EXACT_EPS
    
    aspect_degrees = np.array([a.degree for a in settings.aspects], dtype=np.float64)
    aspect_orbs = np.array([a.orb * orb_mult for a in settings.aspects], dtype=np.float64)
    aspect_names = [a.name for a in settings.aspects]
    
    aspects_rows = []
    iterator = tqdm(angles_cache.items(), desc=f"Filtering aspects (orb×{orb_mult})") if progress else angles_cache.items()
    
    for d, cache_entry in iterator:
        body_names = cache_entry['body_names']
        longitudes = cache_entry['longitudes']
        speeds = cache_entry['speeds']
        angles_matrix = cache_entry['angles']
        
        i_idx, j_idx, asp_idx, orb_vals = filter_aspects_by_orb(
            angles_matrix, aspect_degrees, aspect_orbs
        )
        
        for k in range(len(i_idx)):
            i, j = i_idx[k], j_idx[k]
            a_idx = asp_idx[k]
            orb = orb_vals[k]
            
            is_exact = orb <= EXACT_EPS
            applying = is_aspect_applying(
                longitudes[i], longitudes[j],
                speeds[i], speeds[j],
                aspect_degrees[a_idx]
            )
            
            aspects_rows.append({
                "date": d,
                "p1": prefix + body_names[i],
                "p2": prefix + body_names[j],
                "aspect": aspect_names[a_idx],
                "orb": orb,
                "is_exact": is_exact,
                "is_applying": applying,
            })
    
    return pd.DataFrame(aspects_rows)


def calculate_aspects_for_dates(
    bodies_by_date: dict,
    settings: AstroSettings,
    orb_mult: float = 1.0,
    progress: bool = True,
    prefix: str = "",
) -> pd.DataFrame:
    """
    Calculate aspects between planets for all dates.
    
    Aspects are angular distances between planets:
    - Conjunction (0°): planets together
    - Sextile (60°): harmonious
    - Square (90°): tense
    - Trine (120°): harmonious
    - Opposition (180°): tense
    
    Args:
        bodies_by_date: Dict {date -> [BodyPosition]}
        settings: AstroSettings with aspect configs
        orb_mult: Orb multiplier (0.5=tight, 1.0=standard, 1.5=wide)
        progress: Show progress bar
        prefix: Prefix for planet names (e.g., "helio_")
    
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
                "p1": prefix + a.p1,
                "p2": prefix + a.p2,
                "aspect": a.aspect,
                "orb": a.orb,
                "is_exact": a.is_exact,
                "is_applying": a.is_applying,
            })
    
    return pd.DataFrame(aspects_rows)


def calculate_aspects_for_dates_multi(
    geo_bodies_by_date: dict,
    helio_bodies_by_date: dict,
    settings: AstroSettings,
    coord_mode: str = "geo",
    orb_mult: float = 1.0,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Calculate aspects for different coordinate systems.
    
    Modes:
        - "geo": Geocentric aspects only
        - "helio": Heliocentric aspects only
        - "both": Both systems combined
    
    Args:
        geo_bodies_by_date: Geocentric body positions
        helio_bodies_by_date: Heliocentric body positions
        settings: AstroSettings
        coord_mode: 'geo', 'helio', or 'both'
        orb_mult: Orb multiplier
        progress: Show progress bar
    
    Returns:
        DataFrame with aspects
    """
    all_dfs = []
    
    if coord_mode in ["geo", "both"] and geo_bodies_by_date:
        prefix = "geo_" if coord_mode == "both" else ""
        if progress:
            print(f"📐 Расчёт GEO-аспектов (orb×{orb_mult})...")
        df_geo_aspects = calculate_aspects_for_dates(
            geo_bodies_by_date, settings, orb_mult, progress, prefix=prefix
        )
        all_dfs.append(df_geo_aspects)
    
    if coord_mode in ["helio", "both"] and helio_bodies_by_date:
        prefix = "helio_"
        if progress:
            print(f"☀️ Расчёт HELIO-аспектов (orb×{orb_mult})...")
        df_helio_aspects = calculate_aspects_for_dates(
            helio_bodies_by_date, settings, orb_mult, progress, prefix=prefix
        )
        all_dfs.append(df_helio_aspects)
    
    if not all_dfs:
        return pd.DataFrame()
    
    if len(all_dfs) == 1:
        return all_dfs[0]
    
    result = pd.concat(all_dfs, ignore_index=True)
    if progress:
        print(f"✅ Объединено {len(result)} аспектов из {len(all_dfs)} систем")
    
    return result
