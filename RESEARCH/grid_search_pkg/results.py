"""
Results handling for Grid Search.

Functions for saving, loading, and processing grid search results.
"""
import pandas as pd
from pathlib import Path
from typing import Dict
from datetime import datetime

from ..config import cfg


def save_grid_search_results(results_df: pd.DataFrame) -> Path:
    """Save grid search results to reports directory."""
    reports_dir = cfg.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"grid_search_{timestamp}.csv"
    
    results_df.to_csv(path, index=False)
    print(f"\nResults saved to: {path}")
    
    return path


def get_best_params(results_df: pd.DataFrame) -> Dict:
    """Extract best parameters from grid search results."""
    if results_df.empty:
        return {}
    
    best = results_df.iloc[0].to_dict()
    return {
        "orb_mult": float(best.get("orb_mult", 1.0)),
        "gauss_window": int(best.get("gauss_window", 201)),
        "gauss_std": float(best.get("gauss_std", 50.0)),
        "threshold": float(best.get("threshold", 0.5)),
    }


def save_best_params(params: Dict, name: str = "best") -> Path:
    """Save best parameters to YAML file."""
    import yaml
    
    reports_dir = cfg.reports_dir
    path = reports_dir / f"{name}_params.yaml"
    
    with open(path, "w") as f:
        yaml.dump(params, f) 
    
    print(f"Best params saved to: {path}")
    return path
