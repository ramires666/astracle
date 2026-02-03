"""
Body Ablation Research Script
=============================

This script performs:
1. Baseline model evaluation
2. Full grid search: gauss params × coord modes × body exclusions
3. Best model evaluation with full metrics and visualizations

Run with: python -m RESEARCH.body_ablation_research
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

# RESEARCH imports
from RESEARCH.data_loader import load_market_data
from RESEARCH.labeling import create_balanced_labels
from RESEARCH.astro_engine import (
    init_ephemeris,
    calculate_bodies_for_dates_multi,
    calculate_aspects_for_dates,
    calculate_phases_for_dates,
)
from RESEARCH.features import build_full_features, merge_features_with_labels
from RESEARCH.model_training import (
    split_dataset,
    prepare_xy,
    train_xgb_model,
    tune_threshold,
    predict_with_threshold,
    check_cuda_available,
)
from RESEARCH.evaluation import evaluate_model_full, compare_models


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model params (fixed)
MODEL_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.03,
    'colsample_bytree': 0.6,
    'subsample': 0.8,
}

# GRID SEARCH PARAMETERS
GRID_PARAMS = {
    # Gaussian params to try
    'gauss_windows': [150, 200, 250],
    'gauss_stds': [50.0, 70.0, 90.0],
    
    # Coordinate modes
    'coord_modes': ['geo', 'both'],
    
    # Orb multiplier
    'orb_mults': [0.1],
}

# Best bodies to exclude (from single-body ablation study)
# Top 5: MeanNode, Pluto, Saturn, Venus, Neptune
ABLATION_BODIES = [
    [],  # Baseline
    # Single exclusions (top 5)
    ['MeanNode'],
    ['Pluto'],
    ['Saturn'],
    ['Venus'],
    ['Neptune'],
    # Pairs of best performers
    ['MeanNode', 'Pluto'],
    ['MeanNode', 'Saturn'],
    ['MeanNode', 'Venus'],
    ['Pluto', 'Saturn'],
    ['Pluto', 'Venus'],
]


def train_and_evaluate(
    df_market: pd.DataFrame,
    df_bodies: pd.DataFrame,
    geo_by_date: dict,
    settings,
    gauss_window: int,
    gauss_std: float,
    orb_mult: float,
    exclude_bodies: list = None,
    device: str = 'cpu',
    show_plots: bool = False,
    verbose: bool = True,
):
    """
    Train model with specific params and return evaluation.
    """
    # 1. Create labels
    df_labels = create_balanced_labels(
        df_market,
        gauss_window=gauss_window,
        gauss_std=gauss_std,
    )
    
    # 2. Calculate aspects
    df_aspects = calculate_aspects_for_dates(
        geo_by_date, settings, orb_mult=orb_mult, progress=False
    )
    
    # 3. Calculate phases
    df_phases = calculate_phases_for_dates(geo_by_date, progress=False)
    
    # 4. Build features
    df_features = build_full_features(
        df_bodies, df_aspects,
        df_phases=df_phases,
        exclude_bodies=exclude_bodies
    )
    
    # 5. Merge with labels
    df_dataset = merge_features_with_labels(df_features, df_labels)
    
    if len(df_dataset) < 100:
        return None
    
    # 6. Split
    train_df, val_df, test_df = split_dataset(df_dataset)
    
    feature_cols = [c for c in df_dataset.columns if c not in ["date", "target"]]
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val, y_val = prepare_xy(val_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)
    
    # 7. Train
    model = train_xgb_model(
        X_train, y_train, X_val, y_val,
        feature_cols, n_classes=2, device=device,
        **MODEL_PARAMS
    )
    
    # 8. Tune threshold
    best_t, _ = tune_threshold(model, X_val, y_val, metric="recall_min", verbose=verbose)
    
    # 9. Predict
    y_pred = predict_with_threshold(model, X_test, threshold=best_t)
    
    # 10. Metrics
    from sklearn.metrics import classification_report, balanced_accuracy_score, matthews_corrcoef
    report = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["DOWN", "UP"], output_dict=True, zero_division=0
    )
    
    recall_down = report["DOWN"]["recall"]
    recall_up = report["UP"]["recall"]
    
    metrics = {
        'recall_min': min(recall_down, recall_up),
        'recall_gap': abs(recall_down - recall_up),
        'recall_down': recall_down,
        'recall_up': recall_up,
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'f1_macro': report['macro avg']['f1-score'],
    }
    
    return {
        'model': model,
        'threshold': best_t,
        'metrics': metrics,
        'n_features': len(feature_cols),
        'y_test': y_test,
        'y_pred': y_pred,
        'test_dates': test_df['date'].reset_index(drop=True),
    }


def run_full_grid_search(df_market, settings, device='cpu'):
    """
    Run full grid search over gauss params, coord modes, and body exclusions.
    """
    # Generate all combinations
    combos = list(product(
        GRID_PARAMS['coord_modes'],
        GRID_PARAMS['gauss_windows'],
        GRID_PARAMS['gauss_stds'],
        GRID_PARAMS['orb_mults'],
        ABLATION_BODIES,
    ))
    
    print(f"\n📊 Total combinations: {len(combos)}")
    print(f"   Coord modes: {GRID_PARAMS['coord_modes']}")
    print(f"   Gauss windows: {GRID_PARAMS['gauss_windows']}")
    print(f"   Gauss stds: {GRID_PARAMS['gauss_stds']}")
    print(f"   Body exclusions: {len(ABLATION_BODIES)}")
    
    # Pre-calculate bodies for each coord mode
    cached_bodies = {}
    for coord_mode in GRID_PARAMS['coord_modes']:
        print(f"\n📍 Pre-calculating bodies for {coord_mode}...")
        df_bodies, geo_by_date, helio_by_date = calculate_bodies_for_dates_multi(
            df_market['date'], settings, coord_mode=coord_mode, progress=True
        )
        cached_bodies[coord_mode] = (df_bodies, geo_by_date, helio_by_date)
    
    # Run grid search
    results = []
    
    print("\n" + "=" * 70)
    print("🚀 STARTING GRID SEARCH")
    print("=" * 70)
    
    for i, (coord, gw, gs, orb, excl) in enumerate(combos):
        excl_str = ','.join(excl) if excl else 'none'
        
        # Get cached bodies
        df_bodies, geo_by_date, _ = cached_bodies[coord]
        
        # Train and evaluate
        result = train_and_evaluate(
            df_market, df_bodies, geo_by_date, settings,
            gauss_window=gw,
            gauss_std=gs,
            orb_mult=orb,
            exclude_bodies=excl if excl else None,
            device=device,
            show_plots=False,
            verbose=False,
        )
        
        if result is None:
            continue
        
        m = result['metrics']
        results.append({
            'coord_mode': coord,
            'gauss_window': gw,
            'gauss_std': gs,
            'orb_mult': orb,
            'exclude_bodies': excl_str,
            'n_features': result['n_features'],
            'recall_min': m['recall_min'],
            'recall_gap': m['recall_gap'],
            'balanced_accuracy': m['balanced_accuracy'],
            'mcc': m['mcc'],
            'f1_macro': m['f1_macro'],
            'threshold': result['threshold'],
        })
        
        # Print progress
        print(f"[{i+1:3d}/{len(combos)}] {coord:5s} W={gw} S={gs:.0f} O={orb} excl={excl_str:20s} → R_MIN={m['recall_min']:.3f} MCC={m['mcc']:.3f}")
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("🔬 BODY ABLATION RESEARCH - FULL GRID SEARCH")
    print("=" * 70)
    
    # Check device
    _, device = check_cuda_available()
    print(f"Device: {device}")
    
    # 1. Load data
    print("\n📥 Loading market data...")
    df_market = load_market_data()
    df_market = df_market[df_market['date'] >= '2017-11-01'].reset_index(drop=True)
    print(f"Market data: {len(df_market)} rows")
    
    # 2. Initialize ephemeris
    settings = init_ephemeris()
    
    # 3. Run full grid search
    results_df = run_full_grid_search(df_market, settings, device=device)
    
    # 4. Sort and display results
    results_df = results_df.sort_values('recall_min', ascending=False)
    
    print("\n" + "=" * 70)
    print("📊 TOP 20 RESULTS (by R_MIN)")
    print("=" * 70)
    print(results_df.head(20).to_string(index=False))
    
    # 5. Best result
    best = results_df.iloc[0]
    print("\n" + "=" * 70)
    print("🏆 BEST CONFIGURATION")
    print("=" * 70)
    print(f"Coord mode:     {best['coord_mode']}")
    print(f"Gauss window:   {best['gauss_window']}")
    print(f"Gauss std:      {best['gauss_std']}")
    print(f"Orb mult:       {best['orb_mult']}")
    print(f"Exclude bodies: {best['exclude_bodies']}")
    print(f"Features:       {best['n_features']}")
    print("-" * 40)
    print(f"R_MIN:          {best['recall_min']:.4f}")
    print(f"BAL_ACC:        {best['balanced_accuracy']:.4f}")
    print(f"MCC:            {best['mcc']:.4f}")
    
    # 6. Full evaluation of best model
    print("\n" + "=" * 70)
    print("📈 FULL EVALUATION OF BEST MODEL")
    print("=" * 70)
    
    coord_mode = best['coord_mode']
    df_bodies, geo_by_date, _ = calculate_bodies_for_dates_multi(
        df_market['date'], settings, coord_mode=coord_mode, progress=False
    )
    
    excl = best['exclude_bodies'].split(',') if best['exclude_bodies'] != 'none' else None
    
    result = train_and_evaluate(
        df_market, df_bodies, geo_by_date, settings,
        gauss_window=int(best['gauss_window']),
        gauss_std=float(best['gauss_std']),
        orb_mult=float(best['orb_mult']),
        exclude_bodies=excl,
        device=device,
        show_plots=True,
        verbose=True,
    )
    
    # Full evaluation with plots
    evaluate_model_full(
        result['y_test'], result['y_pred'],
        dates=result['test_dates'],
        title=f"Best Model: {best['exclude_bodies']} ({best['coord_mode']})",
        show_plot=True,
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"RESEARCH/reports/grid_search_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    return results_df, result


if __name__ == "__main__":
    results_df, best_result = main()
