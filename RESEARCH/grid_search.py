"""
Grid search module for RESEARCH pipeline.
Hyperparameter optimization for orb multiplier, gaussian params, etc.

Saves best results to disk for later use.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .config import cfg
from .data_loader import load_market_data
from .labeling import create_balanced_labels, gaussian_smooth_centered
from .astro_engine import (
    init_ephemeris,
    calculate_bodies_for_dates,
    calculate_aspects_for_dates,
    calculate_transits_for_dates,
    get_natal_bodies,
)
from .features import build_full_features, merge_features_with_labels
from .model_training import (
    split_dataset,
    prepare_xy,
    train_xgb_model,
    tune_threshold,
    predict_with_threshold,
    calc_metrics,
    check_cuda_available,
)


class GridSearchConfig:
    """Configuration for grid search."""
    
    def __init__(
        self,
        orb_multipliers: List[float] = [0.8, 1.0, 1.2],
        gauss_windows: List[int] = [101, 151, 201],
        gauss_stds: List[float] = [30.0, 50.0, 70.0],
        max_combos: Optional[int] = None,
        model_params: Optional[Dict] = None,
    ):
        self.orb_multipliers = orb_multipliers
        self.gauss_windows = gauss_windows
        self.gauss_stds = gauss_stds
        self.max_combos = max_combos
        self.model_params = model_params or {
            "n_estimators": 500,
            "max_depth": 3,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }


def evaluate_combo(
    df_market: pd.DataFrame,
    df_bodies: pd.DataFrame,
    bodies_by_date: dict,
    settings: Any,
    orb_mult: float,
    gauss_window: int,
    gauss_std: float,
    device: str = "cpu",
    model_params: Optional[Dict] = None,
) -> Dict:
    """
    Evaluate a single hyperparameter combination.
    
    Returns:
        Dictionary with combo params and metrics
    """
    # Create labels with this gauss config
    df_labels = create_balanced_labels(
        df_market,
        gauss_window=gauss_window,
        gauss_std=gauss_std,
    )
    
    # Calculate aspects with this orb
    df_aspects = calculate_aspects_for_dates(
        bodies_by_date, settings, orb_mult=orb_mult, progress=False
    )
    
    # Build features
    df_features = build_full_features(df_bodies, df_aspects)
    
    # Merge with labels
    df_dataset = merge_features_with_labels(df_features, df_labels)
    
    if len(df_dataset) < 100:
        return {"error": "Too few samples"}
    
    # Split
    train_df, val_df, test_df = split_dataset(df_dataset)
    
    feature_cols = [c for c in df_dataset.columns if c not in ["date", "target"]]
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val, y_val = prepare_xy(val_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)
    
    # Train
    params = model_params or {}
    model = train_xgb_model(
        X_train, y_train, X_val, y_val,
        feature_cols, n_classes=2, device=device,
        **params
    )
    
    # Tune threshold
    best_t, _ = tune_threshold(model, X_val, y_val, metric="bal_acc")
    
    # Predict on test
    y_pred = predict_with_threshold(model, X_test, threshold=best_t)
    
    # Metrics
    metrics = calc_metrics(y_test, y_pred, [0, 1])
    
    # Per-class F1
    from sklearn.metrics import classification_report
    report = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["DOWN", "UP"], output_dict=True, zero_division=0
    )
    
    return {
        "orb_mult": orb_mult,
        "gauss_window": gauss_window,
        "gauss_std": gauss_std,
        "threshold": best_t,
        "f1_down": report["DOWN"]["f1-score"],
        "f1_up": report["UP"]["f1-score"],
        "f1_min": min(report["DOWN"]["f1-score"], report["UP"]["f1-score"]),
        "f1_macro": metrics["f1_macro"],
        "bal_acc": metrics["bal_acc"],
        "summary": metrics["summary"],
    }


def run_grid_search(
    df_market: pd.DataFrame,
    config: Optional[GridSearchConfig] = None,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run full grid search over hyperparameters.
    
    Args:
        df_market: Market data DataFrame
        config: GridSearchConfig (uses defaults if None)
        save_results: Save results to reports directory
    
    Returns:
        DataFrame with all results sorted by balance
    """
    config = config or GridSearchConfig()
    
    print("=" * 60)
    print("GRID SEARCH: ORB + GAUSSIAN PARAMETERS")
    print("=" * 60)
    
    # Check CUDA
    _, device = check_cuda_available()
    print(f"Device: {device}")
    
    # Initialize astro
    settings = init_ephemeris()
    
    # Calculate bodies once
    print("\nCalculating body positions...")
    df_bodies, bodies_by_date = calculate_bodies_for_dates(
        df_market["date"], settings, progress=True
    )
    
    # Generate all combos
    combos = []
    for orb in config.orb_multipliers:
        for gw in config.gauss_windows:
            for gs in config.gauss_stds:
                combos.append((orb, gw, gs))
    
    if config.max_combos and len(combos) > config.max_combos:
        combos = combos[:config.max_combos]
    
    print(f"\nTotal combinations: {len(combos)}")
    
    # Run grid search
    results = []
    for i, (orb, gw, gs) in enumerate(combos):
        print(f"\n[{i+1}/{len(combos)}] orb={orb}, gauss_window={gw}, gauss_std={gs}")
        
        try:
            res = evaluate_combo(
                df_market, df_bodies, bodies_by_date, settings,
                orb, gw, gs,
                device=device,
                model_params=config.model_params,
            )
            results.append(res)
            
            if "error" not in res:
                print(f"  → bal_acc={res['bal_acc']:.3f}, f1_min={res['f1_min']:.3f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "orb_mult": orb, "gauss_window": gw, "gauss_std": gs,
                "error": str(e)
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by balance (minimize f1 gap, then maximize f1_min)
    if "f1_down" in results_df.columns and "f1_up" in results_df.columns:
        results_df["f1_gap"] = (results_df["f1_down"] - results_df["f1_up"]).abs()
        results_df = results_df.sort_values(
            ["f1_gap", "f1_min", "bal_acc"],
            ascending=[True, False, False]
        ).reset_index(drop=True)
    
    # Save results
    if save_results:
        save_grid_search_results(results_df)
    
    # Print best
    print("\n" + "=" * 60)
    print("TOP 5 COMBOS BY BALANCE:")
    print(results_df.head(5).to_string(index=False))
    
    return results_df


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
