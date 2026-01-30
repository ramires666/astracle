"""
Load astro calculation settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .models import BodyConfig, AspectConfig


ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]


class AstroSettings:
    """
    Collect bodies and aspects settings.
    """

    def __init__(self, bodies_path: Path, aspects_path: Path):
        self.bodies: List[BodyConfig] = self._load_bodies(bodies_path)
        self.aspects: List[AspectConfig] = self._load_aspects(aspects_path)
        self.body_map: Dict[str, BodyConfig] = {b.name: b for b in self.bodies}

    def _load_bodies(self, path: Path) -> List[BodyConfig]:
        data = _load_yaml(path)
        raw_bodies = data.get("bodies", [])
        return [
            BodyConfig(
                id=int(b["id"]),
                name=str(b["name"]),
                swe_id=b["swe_id"],
                visible=bool(b.get("visible", True)),
            )
            for b in raw_bodies
        ]

    def _load_aspects(self, path: Path) -> List[AspectConfig]:
        data = _load_yaml(path)
        raw_aspects = data.get("aspects", [])
        return [
            AspectConfig(
                name=str(a["name"]),
                degree=float(a["degree"]),
                orb=float(a["orb"]),
            )
            for a in raw_aspects
        ]


def _load_yaml(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
