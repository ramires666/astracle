"""
Calculate body positions (Swiss Ephemeris).
"""

from __future__ import annotations

from datetime import datetime, date, time
from typing import List

import swisseph as swe

from .models import BodyConfig, BodyPosition
from .settings import ZODIAC_SIGNS


def set_ephe_path(ephe_path: str) -> None:
    """
    Set path to Swiss Ephemeris files.
    """
    swe.set_ephe_path(ephe_path)


def get_julian_day(dt_utc: datetime) -> float:
    """
    Convert UTC datetime to Julian day.
    """
    return swe.utc_to_jd(
        dt_utc.year, dt_utc.month, dt_utc.day,
        dt_utc.hour, dt_utc.minute, dt_utc.second, 1
    )[1]


def _map_swe_id(swe_id: int | str) -> int:
    """
    Map string identifiers to swe constants.
    """
    if isinstance(swe_id, int):
        return swe_id

    s = str(swe_id).lower().strip()
    if s == "true_node":
        return swe.TRUE_NODE
    if s == "mean_node":
        return swe.MEAN_NODE
    if s == "lilith":
        return swe.MEAN_APOG

    raise ValueError(f"Unknown swe_id: {swe_id}")


def _sign_from_lon(lon: float) -> str:
    """
    Zodiac sign by longitude.
    """
    idx = int(lon / 30) % 12
    return ZODIAC_SIGNS[idx]


def _center_to_flag(center: str) -> int:
    """
    Map coordinate center to Swiss Ephemeris flags.
    """
    s = str(center).strip().lower()
    if s in ("geo", "geocentric"):
        return 0
    if s in ("helio", "heliocentric"):
        return swe.FLG_HELCTR
    raise ValueError(f"Unknown center: {center}. Use 'geo' or 'helio'.")


def calculate_bodies(
    dt_utc: datetime,
    bodies: List[BodyConfig],
    center: str = "geo",
) -> List[BodyPosition]:
    """
    Calculate body positions for the given time.
    """
    jd = get_julian_day(dt_utc)
    result: List[BodyPosition] = []
    flags = swe.FLG_SWIEPH | swe.FLG_SPEED | _center_to_flag(center)

    for body in bodies:
        swe_id = _map_swe_id(body.swe_id)
        pos, flag = swe.calc_ut(jd, swe_id, flags)
        if flag < 0:
            # Skip body if error occurred
            continue

        lon = pos[0]
        lat = pos[1]
        speed = pos[3]
        is_retro = speed < 0
        sign = _sign_from_lon(lon)

        # Declination = lat (simplification)
        declination = lat

        result.append(
            BodyPosition(
                date=dt_utc.date(),
                body=body.name,
                lon=lon,
                lat=lat,
                speed=speed,
                is_retro=is_retro,
                sign=sign,
                declination=declination,
            )
        )

    return result


def calculate_daily_bodies(
    day: date,
    time_utc: time,
    bodies: List[BodyConfig],
    center: str = "geo",
) -> List[BodyPosition]:
    """
    Calculate bodies for the given date using a fixed UTC time.
    """
    dt = datetime.combine(day, time_utc)
    return calculate_bodies(dt, bodies, center=center)
