"""
Configuration module for RESEARCH pipeline.
Handles project root detection, config loading, and path resolution.
"""
import sys
from pathlib import Path
from typing import Any

import yaml


def find_project_root() -> Path:
    """Find the project root by looking for configs/market.yaml."""
    current = Path.cwd().resolve()
    
    # Check current and parent directories
    for check in [current, *current.parents]:
        if (check / "configs" / "market.yaml").exists():
            return check
    
    # Fallback: look relative to this file
    module_path = Path(__file__).resolve().parent
    for check in [module_path, module_path.parent]:
        if (check / "configs" / "market.yaml").exists():
            return check
    
    raise FileNotFoundError(
        "Project root not found. Ensure configs/market.yaml exists."
    )


PROJECT_ROOT = find_project_root()

# Ensure project root is in sys.path for imports
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(value: str | Path) -> Path:
    """Resolve a path relative to PROJECT_ROOT if not absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


class Config:
    """Container for all configs."""
    
    def __init__(self):
        self.market = load_yaml(PROJECT_ROOT / "configs/market.yaml")
        self.astro = load_yaml(PROJECT_ROOT / "configs/astro.yaml")
        self.labels = load_yaml(PROJECT_ROOT / "configs/labels.yaml")
        self.db = load_yaml(PROJECT_ROOT / "configs/db.yaml")
        self.training = load_yaml(PROJECT_ROOT / "configs/training.yaml")
        self.subjects = load_yaml(PROJECT_ROOT / "configs/subjects.yaml")
        
        # Derived paths
        market_cfg = self.market.get("market", {})
        self.data_root = resolve_path(market_cfg.get("data_root", "data"))
        self.processed_dir = self.data_root / "processed"
        self.reports_dir = self.data_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Active subject
        self.active_subject_id = self.subjects.get("active_subject_id", "")
        subjects_list = self.subjects.get("subjects", {})
        self.subject = subjects_list.get(self.active_subject_id, {})
        
        # Database URL
        self.db_url = (self.db.get("db") or {}).get("url", "")
    
    def get_astro_config(self) -> dict:
        """Get astro configuration with resolved paths."""
        astro_cfg = self.astro.get("astro", {})
        return {
            "ephe_path": resolve_path(astro_cfg.get("ephe_path", "")),
            "bodies_path": resolve_path(astro_cfg.get("bodies_path", "")),
            "aspects_path": resolve_path(astro_cfg.get("aspects_path", "")),
            "daily_time_utc": astro_cfg.get("daily_time_utc", "00:00:00"),
            "center": astro_cfg.get("center", "geo"),
            "include_pair_aspects": astro_cfg.get("include_pair_aspects", True),
            "include_transit_aspects": astro_cfg.get("include_transit_aspects", False),
        }
    
    def get_label_config(self) -> dict:
        """Get labeling configuration."""
        label_cfg = self.labels.get("labels", {})
        return {
            "horizon": int(label_cfg.get("horizon", 1)),
            "target_move_share": float(label_cfg.get("target_move_share", 0.5)),
            "gauss_window": int(label_cfg.get("gauss_window", 201)),
            "gauss_std": float(label_cfg.get("gauss_std", 50.0)),
            "price_mode": label_cfg.get("price_mode", "log"),
        }


# Global config instance
cfg = Config()
