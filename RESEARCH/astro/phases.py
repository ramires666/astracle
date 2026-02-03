"""
Moon phases and planet elongation calculations for the astro package.

This module handles:
- Moon phase calculations
- Planet elongation from Sun
- Illumination and lunar day calculations
"""
import math
import pandas as pd
import numpy as np
from typing import Dict
from tqdm import tqdm


def calculate_moon_phase(sun_lon: float, moon_lon: float) -> Dict:
    """
    Calculate moon phase from Sun and Moon longitudes.
    
    Phase angles:
    - 0°: New Moon (invisible)
    - 90°: First Quarter (half illuminated)
    - 180°: Full Moon (fully illuminated)
    - 270°: Last Quarter (other half illuminated)
    
    Args:
        sun_lon: Sun longitude (0-360°)
        moon_lon: Moon longitude (0-360°)
    
    Returns:
        Dict with:
        - phase_angle: 0-360°
        - phase_ratio: 0.0-1.0 (0=new, 0.5=full)
        - illumination: 0.0-1.0 (fraction illuminated)
        - lunar_day: 1-29.5 (lunar day number)
        - phase_name: Phase name in Russian
    """
    # Phase angle (Moon-Sun distance)
    phase_angle = (moon_lon - sun_lon) % 360.0
    phase_ratio = phase_angle / 360.0
    
    # Illumination fraction
    illumination = (1 - math.cos(math.radians(phase_angle))) / 2.0
    
    # Lunar day (1-29.5)
    SYNODIC_MONTH = 29.53059
    lunar_day = (phase_angle / 360.0) * SYNODIC_MONTH + 1.0
    
    # Phase name
    if phase_angle < 22.5 or phase_angle >= 337.5:
        phase_name = "Новолуние"
    elif phase_angle < 67.5:
        phase_name = "Молодая Луна"
    elif phase_angle < 112.5:
        phase_name = "Первая четверть"
    elif phase_angle < 157.5:
        phase_name = "Прибывающая Луна"
    elif phase_angle < 202.5:
        phase_name = "Полнолуние"
    elif phase_angle < 247.5:
        phase_name = "Убывающая Луна"
    elif phase_angle < 292.5:
        phase_name = "Последняя четверть"
    else:
        phase_name = "Старая Луна"
    
    return {
        "phase_angle": phase_angle,
        "phase_ratio": phase_ratio,
        "illumination": illumination,
        "lunar_day": lunar_day,
        "phase_name": phase_name,
    }


def calculate_planet_elongation(sun_lon: float, planet_lon: float) -> Dict:
    """
    Calculate planet elongation from Sun.
    
    Elongation is the angular distance from Sun as seen from Earth.
    - Positive: Evening star (east of Sun)
    - Negative: Morning star (west of Sun)
    
    Args:
        sun_lon: Sun longitude (0-360°)
        planet_lon: Planet longitude (0-360°)
    
    Returns:
        Dict with:
        - elongation: -180 to +180°
        - elongation_abs: 0-180° (absolute)
        - position: 'morning', 'evening', or 'conjunction'
    """
    diff = (planet_lon - sun_lon) % 360.0
    
    # Convert to -180 to +180 range
    if diff > 180:
        diff = diff - 360
    
    elongation = diff
    elongation_abs = abs(diff)
    
    # Position
    if elongation > 0:
        position = "evening"
    elif elongation < 0:
        position = "morning"
    else:
        position = "conjunction"
    
    return {
        "elongation": elongation,
        "elongation_abs": elongation_abs,
        "position": position,
    }


def calculate_phases_for_dates(
    bodies_by_date: dict,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Calculate moon phases and planet elongations for all dates.
    
    For each date calculates:
    - Moon phase (angle, ratio, illumination, lunar day)
    - Elongations of all planets from Sun
    
    Args:
        bodies_by_date: Dict {date: [BodyPosition, ...]}
        progress: Show progress bar
    
    Returns:
        DataFrame with phase/elongation features per date
    """
    rows = []
    iterator = tqdm(bodies_by_date.items(), desc="Calculating phases & elongations") if progress else bodies_by_date.items()
    
    for d, bodies in iterator:
        row = {"date": d}
        
        # Find Sun and Moon
        sun_lon = None
        moon_lon = None
        planet_lons = {}
        
        for b in bodies:
            if b.body == "Sun":
                sun_lon = b.lon
            elif b.body == "Moon":
                moon_lon = b.lon
            else:
                planet_lons[b.body] = b.lon
        
        # Moon phase
        if sun_lon is not None and moon_lon is not None:
            moon_phase = calculate_moon_phase(sun_lon, moon_lon)
            row["moon_phase_angle"] = moon_phase["phase_angle"]
            row["moon_phase_ratio"] = moon_phase["phase_ratio"]
            row["moon_illumination"] = moon_phase["illumination"]
            row["lunar_day"] = moon_phase["lunar_day"]
        
        # Planet elongations
        if sun_lon is not None:
            for planet, lon in planet_lons.items():
                elong = calculate_planet_elongation(sun_lon, lon)
                row[f"{planet}_elongation"] = elong["elongation"]
                row[f"{planet}_elongation_abs"] = elong["elongation_abs"]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if progress:
        print(f"✅ Рассчитано {len(df)} дней: фаза Луны + элонгации планет")
    
    return df


# Legacy function for backwards compatibility
def calculate_moon_phases(dates: pd.Series) -> pd.Series:
    """
    DEPRECATED: Use calculate_phases_for_dates instead.
    """
    return pd.Series(np.nan, index=dates.index)
