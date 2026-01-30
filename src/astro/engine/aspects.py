"""
Aspect calculations between bodies.
"""

from __future__ import annotations

from typing import List

from .models import AspectConfig, AspectHit, BodyPosition, TransitAspectHit


EXACT_EPS = 0.1  # orb for "exact" aspect (degrees)


def _diff_0_180(lon1: float, lon2: float) -> float:
    """
    Longitude difference normalized to [0..180].
    """
    diff = abs(lon1 - lon2) % 360
    return diff if diff <= 180 else 360 - diff


def _is_applying(
    lon1: float,
    lon2: float,
    speed1: float,
    speed2: float,
    aspect_degree: float,
    dt_days: float = 1.0,
) -> bool:
    """
    Determine applying/separating by approach trend.
    Simplification: compare current orb vs orb after dt_days.
    """
    diff_now = _diff_0_180(lon1, lon2)
    orb_now = abs(diff_now - aspect_degree)

    lon1_next = (lon1 + speed1 * dt_days) % 360
    lon2_next = (lon2 + speed2 * dt_days) % 360
    diff_next = _diff_0_180(lon1_next, lon2_next)
    orb_next = abs(diff_next - aspect_degree)

    return orb_next < orb_now


def calculate_aspects(
    bodies: List[BodyPosition],
    aspects: List[AspectConfig],
) -> List[AspectHit]:
    """
    Find aspects between all body pairs.
    """
    hits: List[AspectHit] = []

    for i, b1 in enumerate(bodies):
        for b2 in bodies[i + 1:]:
            diff = _diff_0_180(b1.lon, b2.lon)

            for asp in aspects:
                orb = abs(diff - asp.degree)
                if orb <= asp.orb:
                    is_exact = orb <= EXACT_EPS
                    is_applying = _is_applying(
                        b1.lon, b2.lon, b1.speed, b2.speed, asp.degree
                    )
                    hits.append(
                        AspectHit(
                            date=b1.date,
                            p1=b1.body,
                            p2=b2.body,
                            aspect=asp.name,
                            orb=orb,
                            is_exact=is_exact,
                            is_applying=is_applying,
                        )
                    )
                    break

    return hits


def calculate_transit_aspects(
    transit_bodies: List[BodyPosition],
    natal_bodies: List[BodyPosition],
    aspects: List[AspectConfig],
) -> List[TransitAspectHit]:
    """
    Find aspects between transit bodies and natal bodies.
    """
    hits: List[TransitAspectHit] = []

    for tb in transit_bodies:
        for nb in natal_bodies:
            diff = _diff_0_180(tb.lon, nb.lon)

            for asp in aspects:
                orb = abs(diff - asp.degree)
                if orb <= asp.orb:
                    is_exact = orb <= EXACT_EPS
                    is_applying = _is_applying(
                        tb.lon, nb.lon, tb.speed, nb.speed, asp.degree
                    )
                    hits.append(
                        TransitAspectHit(
                            date=tb.date,
                            transit_body=tb.body,
                            natal_body=nb.body,
                            aspect=asp.name,
                            orb=orb,
                            is_exact=is_exact,
                            is_applying=is_applying,
                        )
                    )
                    break

    return hits
