"""
Data models for astro calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class BodyConfig:
    """
    Celestial body configuration.
    """
    id: int
    name: str
    swe_id: int | str
    visible: bool = True


@dataclass
class AspectConfig:
    """
    Aspect configuration.
    """
    name: str
    degree: float
    orb: float


@dataclass
class BodyPosition:
    """
    Body position at a specific date.
    """
    date: date
    body: str
    lon: float
    lat: float
    speed: float
    is_retro: bool
    sign: str
    declination: float


@dataclass
class AspectHit:
    """
    Aspect found between two bodies.
    """
    date: date
    p1: str
    p2: str
    aspect: str
    orb: float
    is_exact: bool
    is_applying: bool


@dataclass
class TransitAspectHit:
    """
    Transit aspect (transit body -> natal body).
    """
    date: date
    transit_body: str
    natal_body: str
    aspect: str
    orb: float
    is_exact: bool
    is_applying: bool
