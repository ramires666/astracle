"""
Utilities for loading YAML configs.
Keep it simple and transparent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file into a dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def require_keys(cfg: Dict[str, Any], keys: List[str], ctx: str) -> None:
    """
    Validate required keys in a config.
    """
    missing = [k for k in keys if k not in cfg]
    if missing:
        raise KeyError(f"Missing keys in {ctx} config: {missing}")


@dataclass
class Subject:
    """
    Subject description used for prediction.
    """
    subject_id: str
    symbol: str
    exchange: str
    birth_dt_utc: str
    birth_lat: Optional[float]
    birth_lon: Optional[float]


def load_subjects(path: str | Path) -> tuple[Dict[str, Subject], str]:
    """
    Load subjects and return:
    - dict {subject_id: Subject}
    - active_subject_id
    """
    data = load_yaml(path)
    require_keys(data, ["active_subject_id", "subjects"], "subjects.yaml")

    subjects: Dict[str, Subject] = {}
    for raw in data["subjects"]:
        subject = Subject(
            subject_id=raw["subject_id"],
            symbol=raw["symbol"],
            exchange=raw["exchange"],
            birth_dt_utc=raw["birth_dt_utc"],
            birth_lat=raw.get("birth_lat"),
            birth_lon=raw.get("birth_lon"),
        )
        subjects[subject.subject_id] = subject

    active_id = data["active_subject_id"]
    if active_id not in subjects:
        raise ValueError(f"active_subject_id={active_id} not found in subjects list")

    return subjects, active_id
