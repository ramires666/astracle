"""
RESEARCH.grid_search_new - Modular grid search package.

This package provides:
- GridSearchConfig: Configuration for parameter search
- evaluate_combo: Evaluate single parameter combination
- run_grid_search: Main grid search runner

Usage:
    from RESEARCH.grid_search_new import (
        GridSearchConfig,
        evaluate_combo,
        run_grid_search,
    )

For full functionality, the original grid_search module is still available
until the migration is complete.
"""

# Configuration
from .config import (
    GridSearchConfig,
    DEFAULT_MODEL_PARAMS,
    FAST_CONFIG,
    STANDARD_CONFIG,
)

# Evaluation
from .evaluate import evaluate_combo

# Re-export from old module for now (large functions)
from ..grid_search import (
    run_grid_search,
    run_body_ablation_search,
    run_comprehensive_search,
    run_full_grid_search,
    save_grid_search_results,
    get_best_params,
    save_best_params,
    evaluate_and_plot_best,
)

__all__ = [
    # Config
    "GridSearchConfig",
    "DEFAULT_MODEL_PARAMS",
    "FAST_CONFIG",
    "STANDARD_CONFIG",
    # Evaluate
    "evaluate_combo",
    # Runners
    "run_grid_search",
    "run_body_ablation_search",
    "run_comprehensive_search",
    "run_full_grid_search",
    # Results
    "save_grid_search_results",
    "get_best_params",
    "save_best_params",
    "evaluate_and_plot_best",
]
