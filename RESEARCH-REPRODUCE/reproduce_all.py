# ============================================================================
# REPRODUCE_ALL.PY - Master Script to Reproduce Research Journey to 60.3%
# ============================================================================
# 
# This script runs all 5 steps of the research in sequence and verifies
# that we can reproduce the 60.3% recall result.
#
# USAGE:
#   conda activate btc
#   python reproduce_all.py
#
# EXPECTED DURATION: ~30-60 minutes (depends on hardware)
# ============================================================================

import sys
import os
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm

# ============================================================================
# STEP 0: SETUP PATHS AND IMPORTS
# ============================================================================
print("=" * 80)
print("🔬 RESEARCH REPRODUCTION SCRIPT")
print("=" * 80)
print(f"Started at: {datetime.now()}")
print()

# Set up paths - we're in RESEARCH-REPRODUCE folder
REPRODUCE_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = REPRODUCE_ROOT.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(REPRODUCE_ROOT))

# Import from step5 (most complete version of modules)
sys.path.insert(0, str(REPRODUCE_ROOT / "step5_deep_tuning"))

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"REPRODUCE_ROOT: {REPRODUCE_ROOT}")

# ============================================================================
# STEP 1: CHECK DEPENDENCIES
# ============================================================================
print("\n" + "=" * 80)
print("📦 STEP 1: Checking Dependencies")
print("=" * 80)

import importlib.util as iu
required = ["xgboost", "sklearn", "matplotlib", "seaborn", "tqdm", "pyarrow", "psycopg2", "swisseph"]
missing = [pkg for pkg in required if iu.find_spec(pkg) is None]

if missing:
    print(f"❌ Missing packages: {', '.join(missing)}")
    print("Install with: conda install -c conda-forge " + " ".join(missing))
    sys.exit(1)
else:
    print("✅ All dependencies found")

# ============================================================================  
# STEP 2: IMPORT MODULES
# ============================================================================
print("\n" + "=" * 80)
print("📥 STEP 2: Importing Modules from step5_deep_tuning")
print("=" * 80)

try:
    # Import from step5 versions (these are the versions that achieved 60.3%)
    from step5_deep_tuning.config import cfg
    from step5_deep_tuning.data_loader import load_market_data
    from step5_deep_tuning.labeling import create_balanced_labels
    from step5_deep_tuning.astro_engine import (
        init_ephemeris,
        calculate_bodies_for_dates_multi,
        calculate_aspects_for_dates,
        calculate_transits_for_dates,
        calculate_phases_for_dates,
        get_natal_bodies,
    )
    from step5_deep_tuning.features import build_full_features, merge_features_with_labels
    from step5_deep_tuning.model_training import (
        split_dataset, prepare_xy, train_xgb_model, 
        tune_threshold, predict_with_threshold, check_cuda_available
    )
    from sklearn.metrics import classification_report, matthews_corrcoef, recall_score
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Trying alternative import path...")
    
    # Fallback to RESEARCH folder
    sys.path.insert(0, str(PROJECT_ROOT / "RESEARCH"))
    from config import cfg
    from data_loader import load_market_data
    from labeling import create_balanced_labels
    from astro_engine import (
        init_ephemeris,
        calculate_bodies_for_dates_multi,
        calculate_aspects_for_dates,
        calculate_transits_for_dates,
        calculate_phases_for_dates,
        get_natal_bodies,
    )
    from features import build_full_features, merge_features_with_labels
    from model_training import (
        split_dataset, prepare_xy, train_xgb_model, 
        tune_threshold, predict_with_threshold, check_cuda_available
    )
    from sklearn.metrics import classification_report, matthews_corrcoef, recall_score
    print("✅ Modules imported from RESEARCH folder")

# ============================================================================
# STEP 3: LOAD MARKET DATA
# ============================================================================
print("\n" + "=" * 80)
print("📊 STEP 3: Loading Market Data")
print("=" * 80)

df_market = load_market_data()
df_market = df_market[df_market["date"] >= "2017-11-01"].reset_index(drop=True)
print(f"✅ Loaded {len(df_market)} days of market data")
print(f"   Date range: {df_market['date'].min().date()} → {df_market['date'].max().date()}")

# ============================================================================
# STEP 4: RUN THE WINNING CONFIGURATION (Step 5 from research)
# ============================================================================
print("\n" + "=" * 80)
print("🎯 STEP 4: Running Winning Configuration (2009-10-10 Birth Date)")
print("=" * 80)

# THE EXACT CONFIGURATION THAT ACHIEVED 60.3%
TARGET_DATE = date(2009, 10, 10)  # Economic Birth

ASTRO_CONFIG = {
    "coord_mode": "both",
    "orb_mult": 0.15,
    "gauss_window": 300,
    "gauss_std": 70.0,
    "exclude_bodies": None,
}

# Best XGB params found
BEST_XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "colsample_bytree": 0.6,
    "subsample": 0.8,
}

print(f"Birth Date: {TARGET_DATE}")
print(f"Astro Config: {ASTRO_CONFIG}")
print(f"XGB Params: {BEST_XGB_PARAMS}")

# ============================================================================
# STEP 5: PREPARE LABELS
# ============================================================================
print("\n" + "=" * 80)
print("🏷️ STEP 5: Creating Balanced Labels")
print("=" * 80)

df_labels = create_balanced_labels(
    df_market,
    gauss_window=ASTRO_CONFIG["gauss_window"],
    gauss_std=ASTRO_CONFIG["gauss_std"]
)
print(f"✅ Created {len(df_labels)} labeled samples")

# ============================================================================
# STEP 6: CALCULATE ASTRO DATA
# ============================================================================
print("\n" + "=" * 80)
print("🌙 STEP 6: Calculating Astro Data")
print("=" * 80)

settings = init_ephemeris()
cuda_available, device = check_cuda_available()
print(f"   Device: {device}")

# Calculate body positions
print("   Calculating body positions...")
df_bodies, geo_by_date, helio_by_date = calculate_bodies_for_dates_multi(
    df_market["date"], settings, coord_mode="both"
)
bodies_by_date = geo_by_date

# Calculate phases
print("   Calculating phases...")
df_phases = calculate_phases_for_dates(bodies_by_date)

# Calculate aspects
print("   Calculating aspects...")
df_aspects = calculate_aspects_for_dates(
    bodies_by_date, settings, 
    orb_mult=ASTRO_CONFIG["orb_mult"]
)

# Calculate natal bodies for birth date
print(f"   Calculating natal bodies for {TARGET_DATE}...")
natal_dt_str = f"{TARGET_DATE.isoformat()}T12:00:00"
natal_bodies = get_natal_bodies(natal_dt_str, settings)

# Calculate transits to natal
print("   Calculating transits to natal...")
df_transits = calculate_transits_for_dates(
    bodies_by_date, natal_bodies, settings,
    orb_mult=ASTRO_CONFIG["orb_mult"]
)

print(f"✅ Astro data calculated:")
print(f"   Bodies: {len(df_bodies)} records")
print(f"   Aspects: {len(df_aspects) if df_aspects is not None else 0} records")
print(f"   Transits: {len(df_transits) if df_transits is not None else 0} records")
print(f"   Phases: {len(df_phases)} records")

# ============================================================================
# STEP 7: BUILD FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("🔧 STEP 7: Building Features")
print("=" * 80)

df_features = build_full_features(
    df_bodies, df_aspects, df_transits, df_phases,
    include_pair_aspects=True,
    include_transit_aspects=True,
    exclude_bodies=ASTRO_CONFIG["exclude_bodies"]
)

df_merged = merge_features_with_labels(df_features, df_labels)
print(f"✅ Features built: {len(df_merged)} samples, {len(df_merged.columns)-2} features")

# ============================================================================
# STEP 8: TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("✂️ STEP 8: Train/Test Split (80/20)")
print("=" * 80)

df_train, df_test = split_dataset(df_merged, test_size=0.2)
X_train, y_train = prepare_xy(df_train)
X_test, y_test = prepare_xy(df_test)

print(f"   Train: {len(df_train)} samples ({len(df_train)/(len(df_train)+len(df_test))*100:.0f}%)")
print(f"   Test: {len(df_test)} samples ({len(df_test)/(len(df_train)+len(df_test))*100:.0f}%)")
print(f"   Features: {X_train.shape[1]}")

# ============================================================================
# STEP 9: TRAIN MODEL
# ============================================================================
print("\n" + "=" * 80)
print("🤖 STEP 9: Training XGBoost Model")
print("=" * 80)

model = train_xgb_model(
    X_train, y_train,
    device=device,
    **BEST_XGB_PARAMS
)
print("✅ Model trained")

# ============================================================================
# STEP 10: TUNE THRESHOLD AND EVALUATE
# ============================================================================
print("\n" + "=" * 80)
print("📈 STEP 10: Tuning Threshold and Evaluating")
print("=" * 80)

best_threshold = tune_threshold(model, X_train, y_train, metric="recall_min")
print(f"   Best threshold: {best_threshold:.3f}")

y_pred = predict_with_threshold(model, X_test, best_threshold)

# Calculate metrics
recall_0 = recall_score(y_test, y_pred, pos_label=0)
recall_1 = recall_score(y_test, y_pred, pos_label=1)
recall_min = min(recall_0, recall_1)
mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "=" * 80)
print("🏆 FINAL RESULTS")
print("=" * 80)
print(f"   Recall DOWN (class 0): {recall_0:.4f}")
print(f"   Recall UP (class 1):   {recall_1:.4f}")
print(f"   Recall MIN:            {recall_min:.4f} ({recall_min*100:.1f}%)")
print(f"   MCC:                   {mcc:.4f}")
print(f"   Features:              {X_train.shape[1]}")

# ============================================================================
# STEP 11: VERIFY REPRODUCTION
# ============================================================================
print("\n" + "=" * 80)
print("✅ REPRODUCTION VERIFICATION")
print("=" * 80)

TARGET_RMIN = 0.603
TARGET_MCC = 0.315
TOLERANCE = 0.02  # 2% tolerance

rmin_match = abs(recall_min - TARGET_RMIN) <= TOLERANCE
mcc_match = abs(mcc - TARGET_MCC) <= TOLERANCE

if rmin_match and mcc_match:
    print("🎉 SUCCESS! Results reproduced within tolerance!")
    print(f"   Target R_MIN: {TARGET_RMIN} | Got: {recall_min:.3f} | Match: ✅")
    print(f"   Target MCC:   {TARGET_MCC} | Got: {mcc:.3f} | Match: ✅")
else:
    print("⚠️ WARNING: Results differ from original")
    print(f"   Target R_MIN: {TARGET_RMIN} | Got: {recall_min:.3f} | Match: {'✅' if rmin_match else '❌'}")
    print(f"   Target MCC:   {TARGET_MCC} | Got: {mcc:.3f} | Match: {'✅' if mcc_match else '❌'}")
    print("\nPossible reasons:")
    print("   - Different data range")
    print("   - Random seed differences")
    print("   - Library version differences")

print("\n" + "=" * 80)
print(f"Finished at: {datetime.now()}")
print("=" * 80)
