"""
Astro engine module for RESEARCH pipeline.

NOTICE: This module has been refactored into the RESEARCH.astro package.
This file is kept for backwards compatibility and simply re-exports
all functions from the new modular structure.

For new code, prefer importing directly from RESEARCH.astro:
    from RESEARCH.astro import init_ephemeris, calculate_bodies_for_dates

Package structure:
    RESEARCH/astro/
    ├── bodies.py    - Body position calculations
    ├── aspects.py   - Aspect calculations
    ├── transits.py  - Transit-to-natal calculations
    └── phases.py    - Moon phases and elongations
"""

# Re-export everything from the new package structure
from .astro import (
    # Bodies
    init_ephemeris,
    parse_birth_dt_utc,
    parse_time_utc,
    calculate_bodies_for_dates,
    calculate_bodies_for_dates_multi,
    # Aspects
    scale_aspects,
    precompute_angles_for_dates,
    calculate_aspects_from_cache,
    calculate_aspects_for_dates,
    calculate_aspects_for_dates_multi,
    # Transits
    calculate_transits_for_dates,
    get_natal_bodies,
    # Phases
    calculate_moon_phase,
    calculate_planet_elongation,
    calculate_phases_for_dates,
)

# Legacy alias
calculate_moon_phases = calculate_phases_for_dates

__all__ = [
    "init_ephemeris",
    "parse_birth_dt_utc",
    "parse_time_utc",
    "calculate_bodies_for_dates",
    "calculate_bodies_for_dates_multi",
    "scale_aspects",
    "precompute_angles_for_dates",
    "calculate_aspects_from_cache",
    "calculate_aspects_for_dates",
    "calculate_aspects_for_dates_multi",
    "calculate_transits_for_dates",
    "get_natal_bodies",
    "calculate_moon_phase",
    "calculate_planet_elongation",
    "calculate_phases_for_dates",
    "calculate_moon_phases",  # Legacy
]
