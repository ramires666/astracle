"""
Grid Search Package.

Refactored grid search functionality split into logical modules:
- config.py: GridSearchConfig class
- evaluate.py: evaluate_combo function
- results.py: save/load results functions
- runner.py: run_grid_search and related (TODO: migrate)

For backward compatibility, all main functions are re-exported here.
"""

# Re-export from submodules for backward compatibility
from .config import GridSearchConfig
from .evaluate import evaluate_combo
from .results import save_grid_search_results, get_best_params, save_best_params

# Note: run_grid_search still in main grid_search.py for now
# Will be migrated in future refactoring pass

__all__ = [
    "GridSearchConfig",
    "evaluate_combo",
    "save_grid_search_results",
    "get_best_params",
    "save_best_params",
]
