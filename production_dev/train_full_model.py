
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import date
from xgboost import XGBClassifier

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from RESEARCH.data_loader import load_market_data
from RESEARCH.labeling import create_balanced_labels
from RESEARCH.astro_engine import (
    init_ephemeris,
    calculate_bodies_for_dates_multi,
    calculate_aspects_for_dates,
    calculate_transits_for_dates,
    calculate_phases_for_dates,
    get_natal_bodies,
)
from RESEARCH.features import build_full_features, merge_features_with_labels
from RESEARCH.model_training import prepare_xy, check_cuda_available

# Best Parameter Config (from existing model)
BEST_CONFIG = {
    # Astro
    "birth_date": date(2009, 10, 10),
    "coord_mode": "both",
    "orb_mult": 0.1,
    "gauss_window": 200,
    "gauss_std": 70.0,
    "exclude_bodies": None,
    
    # Model
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "colsample_bytree": 0.6,
    "subsample": 0.8,
}

def train_final_model():
    print("🚀 Training Final Production Model on FULL Dataset")
    print("="*60)
    
    # 1. Load Data
    print("Loading market data...")
    df_market = load_market_data()
    # Use all data available from 2017 (stable market)
    df_market = df_market[df_market["date"] >= "2017-01-01"].reset_index(drop=True)
    
    print(f"Data range: {df_market['date'].min()} -> {df_market['date'].max()}")
    print(f"Total days: {len(df_market)}")

    # 2. Astro Calculations
    print("\nCalculating astro features...")
    settings = init_ephemeris()
    
    # Bodies
    df_bodies, geo_dict, helio_dict = calculate_bodies_for_dates_multi(
        df_market["date"], settings, BEST_CONFIG["coord_mode"], progress=True
    )
    
    # Phases
    df_phases = calculate_phases_for_dates(geo_dict, progress=False)
    
    # Natal 
    natal_dt = f"{BEST_CONFIG['birth_date'].isoformat()}T12:00:00"
    natal_bodies = get_natal_bodies(natal_dt, settings)
    
    # Transits
    df_transits = calculate_transits_for_dates(
        geo_dict, natal_bodies, settings, 
        orb_mult=BEST_CONFIG["orb_mult"], progress=False
    )
    
    # Aspects
    df_aspects = calculate_aspects_for_dates(
        geo_dict, settings, 
        orb_mult=BEST_CONFIG["orb_mult"], progress=False
    )
    
    # 3. Features & Labels
    print("\nBuilding features...")
    # Labels (Target)
    df_labels = create_balanced_labels(
        df_market, 
        gauss_window=BEST_CONFIG["gauss_window"], 
        gauss_std=BEST_CONFIG["gauss_std"]
    )
    
    # Features
    df_features = build_full_features(
        df_bodies, df_aspects, df_transits=df_transits, df_phases=df_phases,
        include_pair_aspects=True, include_transit_aspects=True,
        exclude_bodies=BEST_CONFIG["exclude_bodies"]
    )
    
    # Merge
    df_dataset = merge_features_with_labels(df_features, df_labels)
    print(f"Dataset shape: {df_dataset.shape}")
    
    # 4. Train Model
    print("\nTraining XGBoost model...")
    feat_cols = [c for c in df_dataset.columns if c not in ["date", "target"]]
    X, y = prepare_xy(df_dataset, feat_cols)
    
    _, device = check_cuda_available()
    
    model = XGBClassifier(
        n_estimators=BEST_CONFIG["n_estimators"],
        max_depth=BEST_CONFIG["max_depth"],
        learning_rate=BEST_CONFIG["learning_rate"],
        colsample_bytree=BEST_CONFIG["colsample_bytree"],
        subsample=BEST_CONFIG["subsample"],
        device=device,
        random_state=42
    )
    
    model.fit(X, y)
    print("✅ Training complete.")
    
    # 5. Save Model
    output_path = "models_artifacts/btc_astro_predictor_full.joblib"
    
    # Combine model with metadata
    artifact = {
        "model": model,
        "config": BEST_CONFIG,
        "feature_names": feat_cols,
        "training_date_range": (df_market["date"].min(), df_market["date"].max()),
        "train_samples": len(df_market)
    }
    
    joblib.dump(artifact, output_path)
    print(f"\n💾 Model saved to: {output_path}")
    print(f"Training Range: {artifact['training_date_range']}")

if __name__ == "__main__":
    train_final_model()
