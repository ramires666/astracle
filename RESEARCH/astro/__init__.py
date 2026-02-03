"""
RESEARCH.astro - Astrological calculation package.

This package provides functions for calculating:
- Planetary positions (bodies)
- Aspects between planets
- Transit-to-natal aspects
- Moon phases and planet elongations

Usage:
    from RESEARCH.astro import (
        init_ephemeris,
        calculate_bodies_for_dates,
        calculate_bodies_for_dates_multi,
        calculate_aspects_for_dates,
        calculate_transits_for_dates,
        calculate_phases_for_dates,
        get_natal_bodies,
    )
"""

# Body calculations
from .bodies import (
    init_ephemeris,
    parse_birth_dt_utc,
    parse_time_utc,
    calculate_bodies_for_dates,
    calculate_bodies_for_dates_multi,
)

# Aspect calculations
from .aspects import (
    scale_aspects,
    precompute_angles_for_dates,
    calculate_aspects_from_cache,
    calculate_aspects_for_dates,
    calculate_aspects_for_dates_multi,
)

# Transit calculations
from .transits import (
    calculate_transits_for_dates,
    get_natal_bodies,
)

# Phase calculations
from .phases import (
    calculate_moon_phase,
    calculate_planet_elongation,
    calculate_phases_for_dates,
)

__all__ = [
    # Bodies
    "init_ephemeris",
    "parse_birth_dt_utc",
    "parse_time_utc",
    "calculate_bodies_for_dates",
    "calculate_bodies_for_dates_multi",
    # Aspects
    "scale_aspects",
    "precompute_angles_for_dates",
    "calculate_aspects_from_cache",
    "calculate_aspects_for_dates",
    "calculate_aspects_for_dates_multi",
    # Transits
    "calculate_transits_for_dates",
    "get_natal_bodies",
    # Phases
    "calculate_moon_phase",
    "calculate_planet_elongation",
    "calculate_phases_for_dates",
]
