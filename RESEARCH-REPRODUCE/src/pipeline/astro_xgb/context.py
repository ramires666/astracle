"""
Pipeline context for astro_xgb steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.common.config import Subject, load_subjects, load_yaml


def _resolve_path(root: Path, value: str | Path) -> Path:
    """
    Resolve path relative to project root.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


@dataclass(frozen=True)
class PipelineContext:
    """
    Shared context for pipeline steps.
    """
    root: Path
    cfg_market: Dict[str, Any]
    cfg_astro: Dict[str, Any]
    cfg_labels: Dict[str, Any]
    cfg_train: Dict[str, Any]
    cfg_db: Dict[str, Any]
    subject: Subject
    data_root: Path
    processed_dir: Path
    raw_dir: Path
    reports_dir: Path


def build_context(project_root: Path, subject_id: Optional[str] = None) -> PipelineContext:
    """
    Build pipeline context from configs.
    """
    project_root = Path(project_root).resolve()

    cfg_market = load_yaml(project_root / "configs" / "market.yaml")
    cfg_astro = load_yaml(project_root / "configs" / "astro.yaml")
    cfg_labels = load_yaml(project_root / "configs" / "labels.yaml")
    cfg_train = load_yaml(project_root / "configs" / "training.yaml")
    try:
        cfg_db = load_yaml(project_root / "configs" / "db.yaml")
    except Exception:
        cfg_db = {"db": {}}

    subjects, active_id = load_subjects(project_root / "configs" / "subjects.yaml")
    use_id = subject_id or active_id
    if use_id not in subjects:
        raise ValueError(f"subject_id={use_id} not found in configs/subjects.yaml")

    subject = subjects[use_id]

    market_cfg = cfg_market.get("market", {})
    data_root = _resolve_path(project_root, market_cfg.get("data_root", "data/market"))
    processed_dir = data_root / "processed"
    raw_dir = data_root / "raw" / "klines_1d"
    reports_dir = data_root / "reports"

    return PipelineContext(
        root=project_root,
        cfg_market=cfg_market,
        cfg_astro=cfg_astro,
        cfg_labels=cfg_labels,
        cfg_train=cfg_train,
        cfg_db=cfg_db,
        subject=subject,
        data_root=data_root,
        processed_dir=processed_dir,
        raw_dir=raw_dir,
        reports_dir=reports_dir,
    )
