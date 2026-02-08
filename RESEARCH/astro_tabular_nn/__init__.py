"""Astro tabular neural research package."""

from .config import DatasetConfig, SplitConfig, ModelConfig, TrainConfig, ScoutConfig
from .best_grid_dataset import ensure_best_grid_dataset_path, build_best_grid_labeled_dataset
from .data_utils import load_tabular_dataset, build_time_split
from .experiments import run_quick_scout, default_scout_model_grid
from .grid_search import run_broad_grid_trial, GridSearchSpace
from .postrun_report import render_postrun_report
from .presets import recommended_trial_preset

__all__ = [
    "DatasetConfig",
    "SplitConfig",
    "ModelConfig",
    "TrainConfig",
    "ScoutConfig",
    "ensure_best_grid_dataset_path",
    "build_best_grid_labeled_dataset",
    "load_tabular_dataset",
    "build_time_split",
    "run_quick_scout",
    "default_scout_model_grid",
    "run_broad_grid_trial",
    "GridSearchSpace",
    "render_postrun_report",
    "recommended_trial_preset",
]
