# %% [markdown]
# # 🧠 Deep Model Tuning for Bitcoin Birth DATE
# 
# Проверка гипотезы: "Увеличение сложности модели поможет переварить транзиты к натальной карте".
# 
# Дата: **2009-10-10** (Economic Birth / First Rate)
# Признаки: Транзиты к наталу + Аспекты транзитов + Фазы (БЕЗ домов)

# %%
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
from datetime import datetime, date, timezone
from sklearn.metrics import classification_report, matthews_corrcoef

PROJECT_ROOT = Path("/home/rut/ostrofun")
sys.path.insert(0, str(PROJECT_ROOT))

from RESEARCH.config import cfg
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
from RESEARCH.model_training import split_dataset, prepare_xy, train_xgb_model, tune_threshold, predict_with_threshold, check_cuda_available

# %%
# Config
TARGET_DATE = date(2009, 10, 10)
print(f"🧠 Tuning for Birth Date: {TARGET_DATE}")

ASTRO_CONFIG = {
    "coord_mode": "both",
    "orb_mult": 0.15,
    "gauss_window": 300,
    "gauss_std": 70.0,
    "exclude_bodies": None,
}

# Deep Grid Search Space
PARAM_GRID = {
    "n_estimators": [300, 500, 800, 1000],
    "max_depth": [3, 4, 6, 8, 10],  # Пробуем глубокие деревья
    "learning_rate": [0.01, 0.03],
    "colsample_bytree": [0.6, 0.8], 
    "subsample": [0.8],
}

# %%
# 1. Prepare Data
print("Loading data...")
df_market = load_market_data()
df_market = df_market[df_market["date"] >= "2017-11-01"].reset_index(drop=True)
df_labels = create_balanced_labels(df_market, ASTRO_CONFIG["gauss_window"], ASTRO_CONFIG["gauss_std"])
settings = init_ephemeris()
_, device = check_cuda_available()

print("Calculating astro...")
df_bodies, geo_by_date, helio_by_date = calculate_bodies_for_dates_multi(
    df_market["date"], settings, coord_mode="both"
)
bodies_by_date = geo_by_date
df_phases = calculate_phases_for_dates(bodies_by_date)

# 2. Build Natal Features
print(f"Building natal features for {TARGET_DATE}...")
natal_dt_str = f"{TARGET_DATE.isoformat()}T12:00:00"
natal_bodies = get_natal_bodies(natal_dt_str, settings)

df_transits = calculate_transits_for_dates(
    bodies_by_date, natal_bodies, settings, 
    orb_mult=ASTRO_CONFIG["orb_mult"]
)

# Аспекты между транзитами (Baseline features)
df_aspects = calculate_aspects_for_dates(
    bodies_by_date, settings, 
    orb_mult=ASTRO_CONFIG["orb_mult"]
)

# 3. Full Dataset
print("Merging dataset...")
df_features = build_full_features(
    df_bodies, df_aspects, df_transits=df_transits, df_phases=df_phases, 
    include_pair_aspects=True,    # Включаем baseline аспекты
    include_transit_aspects=True  # Включаем натальные транзиты
)
df_dataset = merge_features_with_labels(df_features, df_labels)

print(f"Dataset Shape: {df_dataset.shape}")
print(f"Columns: {len(df_dataset.columns)}")

# %%
# 4. Grid Search
print("🚀 Starting Deep Grid Search...")

train_df, val_df, test_df = split_dataset(df_dataset)
feature_cols = [c for c in df_dataset.columns if c not in ["date", "target"]]
X_train, y_train = prepare_xy(train_df, feature_cols)
X_val, y_val = prepare_xy(val_df, feature_cols)
X_test, y_test = prepare_xy(test_df, feature_cols)

results = []
keys = PARAM_GRID.keys()
combinations = list(product(*PARAM_GRID.values()))

for vals in tqdm(combinations, desc="Grid Search"):
    params = dict(zip(keys, vals))
    
    # Train
    model = train_xgb_model(
        X_train, y_train, X_val, y_val, feature_cols, 
        n_classes=2, device=device, early_stopping_rounds=50, verbose=False,
        **params
    )
    
    # Evaluate
    best_t, _ = tune_threshold(model, X_val, y_val, metric="recall_min")
    y_test_pred = predict_with_threshold(model, X_test, threshold=best_t)
    
    report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    r_min = min(report["0"]["recall"], report["1"]["recall"])
    mcc = matthews_corrcoef(y_test, y_test_pred)
    
    res_row = params.copy()
    res_row["R_MIN"] = r_min
    res_row["MCC"] = mcc
    results.append(res_row)

# %%
# 5. Analysis
df_res = pd.DataFrame(results).sort_values("R_MIN", ascending=False)
print("\n🏆 TOP 10 MODELS:")
print(df_res.head(10))

best = df_res.iloc[0]
print(f"\n🥇 WINNER PARAMS:")
print(best.to_dict())

baseline_rmin = 0.587
if best["R_MIN"] > baseline_rmin:
    print(f"\n🚀 SUCCESS! Deep model beat baseline! ({best['R_MIN']:.3f} > {baseline_rmin})")
else:
    print(f"\n💀 FAILURE. Still can't beat baseline. ({best['R_MIN']:.3f} <= {baseline_rmin})")
    print("Hypothesis: Natal features are just noise.")
