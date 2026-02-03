"""
Grid search configuration module.

Defines GridSearchConfig class for parameter search.
"""
from typing import Dict, List, Optional


class GridSearchConfig:
    """
    Configuration for grid search hyperparameter optimization.
    
    Parameters:
        orb_multipliers: Orb scaling factors for aspects
        gauss_windows: Gaussian smoothing window sizes (days)
        gauss_stds: Gaussian standard deviations
        coord_modes: Coordinate systems ('geo', 'helio', 'both')
        max_exclude: Max bodies to exclude in ablation (0 = no ablation)
        max_combos: Limit number of combinations (for testing)
        model_params: XGBoost model parameters
    """
    
    def __init__(
        self,
        orb_multipliers: List[float] = [0.8, 1.0, 1.2],
        gauss_windows: List[int] = [101, 151, 201],
        gauss_stds: List[float] = [30.0, 50.0, 70.0],
        coord_modes: List[str] = ["geo"],
        max_exclude: int = 0,
        max_combos: Optional[int] = None,
        model_params: Optional[Dict] = None,
    ):
        self.orb_multipliers = orb_multipliers
        self.gauss_windows = gauss_windows
        self.gauss_stds = gauss_stds
        self.coord_modes = coord_modes
        self.max_exclude = max_exclude
        self.max_combos = max_combos
        self.model_params = model_params or {
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }


# Default model parameters for quick experiments
DEFAULT_MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

# Fast test configuration
FAST_CONFIG = GridSearchConfig(
    orb_multipliers=[1.0],
    gauss_windows=[201],
    gauss_stds=[50.0],
    coord_modes=["geo"],
    max_combos=1,
)

# Standard search configuration
STANDARD_CONFIG = GridSearchConfig(
    orb_multipliers=[0.5, 0.8, 1.0, 1.2],
    gauss_windows=[101, 151, 201],
    gauss_stds=[30.0, 50.0, 70.0],
    coord_modes=["geo"],
)
