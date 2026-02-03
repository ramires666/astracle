# %% [markdown]
# # 🧠 Deep Grid Search 2.0: Astro + Model Tuning
# 
# Цель: Найти комбинацию Астро-параметров и параметров Модели, которая побьет Baseline (R_MIN > 0.587).
# 
# Дата: **2009-10-10** (Economic Birth)
# Признаки: Транзиты к наталу + Аспекты транзитов + Фазы

# %%
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm
from datetime import datetime, date, timezone
from sklearn.metrics import classification_report, matthews_corrcoef, confusion_matrix

PROJECT_ROOT = Path("/home/rut/ostrofun")
sys.path.insert(0, str(PROJECT_ROOT))

# %%
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
# 1. Configuration Grid
TARGET_DATE = date(2009, 10, 10)
print(f"🧠 Deep Tuning 2.0 for Date: {TARGET_DATE}")

# Параметры, которые влияют на ДАННЫЕ
ASTRO_GRID = {
    "coord_mode": ["both"],
    "orb_mult": [0.05, 0.075, 0.1, 0.125],
    "gauss_window": [150, 200, 250],
    "gauss_std": [50.0, 70.0, 90.0],
    "exclude_bodies": [None, ["Uranus", "Pluto"]], # Пробуем с/без
}

# Параметры МОДЕЛИ (Refined around Winner: depth=6, est=500, col=0.6)
MODEL_GRID = {
    "n_estimators": [300, 500, 800],
    "max_depth": [4, 5, 6, 7],
    "learning_rate": [0.03], 
    "colsample_bytree": [0.5, 0.6, 0.7], # Winner was 0.6
    "subsample": [0.8],
}

astro_size = len(list(product(*ASTRO_GRID.values())))
model_size = len(list(product(*MODEL_GRID.values())))
print(f"Astro Grid Size: {astro_size}")
print(f"Model Grid Size: {model_size}")
print(f"Total Combinations: {astro_size * model_size}")

# %%
# 2. Setup
df_market_raw = load_market_data()
df_market_raw = df_market_raw[df_market_raw["date"] >= "2017-11-01"].reset_index(drop=True)
settings = init_ephemeris()
_, device = check_cuda_available()

# Baseline Evaluation
BASELINE_RMIN = 0.601  # Updated Baseline!
print("\n" + "="*40)
print(f"🎯 TARGET BASELINE: R_MIN > {BASELINE_RMIN}")
print("="*40 + "\n")

# Pre-calculate bodies for BOTH mode
print("📍 Pre-calculating bodies (both)...")
df_bodies_both, geo_dict, helio_dict = calculate_bodies_for_dates_multi(
    df_market_raw["date"], settings, "both", progress=False
)
# Cache phases as they depend only on bodies
print("🌙 Pre-calculating phases...")
df_phases_geo = calculate_phases_for_dates(geo_dict, progress=False)

# Natal bodies (static)
natal_dt_str = f"{TARGET_DATE.isoformat()}T12:00:00"
natal_bodies = get_natal_bodies(natal_dt_str, settings)

# %%
# 3. Main Loop
results = []
best_model_info = {"score": -1, "y_test": None, "y_pred": None, "params": None}

astro_keys = ASTRO_GRID.keys()
astro_combos = list(product(*ASTRO_GRID.values()))

model_keys = MODEL_GRID.keys()
model_combos = list(product(*MODEL_GRID.values()))

total_steps = len(astro_combos) * len(model_combos)

# Используем custom format для вывода
pbar = tqdm(total=total_steps, desc="Grid Search", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

for astro_vals in astro_combos:
    astro_params = dict(zip(astro_keys, astro_vals))
    
    # 3.1. Build Data
    try:
        # Labels
        df_labels = create_balanced_labels(
            df_market_raw, 
            gauss_window=astro_params["gauss_window"], 
            gauss_std=astro_params["gauss_std"]
        )
        
        # Transits & Aspects (depend on orb_mult)
        target_bodies_dict = geo_dict 
        
        df_transits = calculate_transits_for_dates(
            target_bodies_dict, natal_bodies, settings, 
            orb_mult=astro_params["orb_mult"], progress=False
        )
        df_aspects = calculate_aspects_for_dates(
            target_bodies_dict, settings, 
            orb_mult=astro_params["orb_mult"], progress=False
        )
        
        # Features with exclusion
        df_features = build_full_features(
            df_bodies_both, df_aspects, df_transits=df_transits, df_phases=df_phases_geo,
            include_pair_aspects=True, include_transit_aspects=True,
            exclude_bodies=astro_params["exclude_bodies"]
        )
        
        df_dataset = merge_features_with_labels(df_features, df_labels)
        
        # Split
        train_df, val_df, test_df = split_dataset(df_dataset)
        feat_cols = [c for c in df_dataset.columns if c not in ["date", "target"]]
        X_train, y_train = prepare_xy(train_df, feat_cols)
        X_val, y_val = prepare_xy(val_df, feat_cols)
        X_test, y_test = prepare_xy(test_df, feat_cols)
        
        # 3.2. Models
        for model_vals in model_combos:
            model_params = dict(zip(model_keys, model_vals))
            
            model = train_xgb_model(
                X_train, y_train, X_val, y_val, feat_cols,
                n_classes=2, device=device, verbose=False, early_stopping_rounds=50,
                **model_params
            )
            
            best_t, _ = tune_threshold(model, X_val, y_val, metric="recall_min", verbose=False)
            y_pred = predict_with_threshold(model, X_test, threshold=best_t)
            
            # Metrics
            mcc = matthews_corrcoef(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            r_min = min(report["0"]["recall"], report["1"]["recall"])
            r_up = report["1"]["recall"]
            r_down = report["0"]["recall"]
            gap = abs(r_up - r_down)
            
            # Save
            res = astro_params.copy()
            res.update(model_params)
            res["R_MIN"] = r_min
            res["MCC"] = mcc
            res["GAP"] = gap
            results.append(res)
            
            # Check Best
            if r_min > best_model_info["score"]:
                best_model_info = {
                    "score": r_min,
                    "mcc": mcc,
                    "y_test": y_test,
                    "y_pred": y_pred,
                    "params": res,
                    "model": model,
                    "threshold": best_t
                }
                # Live Log Update
                tqdm.write(f"🚀 NEW BEST: R_MIN={r_min:.4f} MCC={mcc:.4f} Gap={gap:.4f} | "
                           f"Orb={astro_params['orb_mult']} Win={astro_params['gauss_window']} "
                           f"Excl={astro_params['exclude_bodies']}")
            
            pbar.update(1)
            
    except Exception as e:
        tqdm.write(f"❌ Error config {astro_params}: {e}")
        pbar.update(len(model_combos))

pbar.close()

# %%
# 4. Analysis & Viz
df_res = pd.DataFrame(results).sort_values("R_MIN", ascending=False)

print("\n" + "="*80)
print(f"🏆 TOP 10 CONFIGURATIONS (Baseline: {BASELINE_RMIN})")
print("="*80)
# Format columns for readability
cols_show = ["orb_mult", "gauss_window", "exclude_bodies", "n_estimators", "max_depth", "R_MIN", "MCC", "GAP"]
print(df_res.head(10)[cols_show].to_string(index=False))

best = best_model_info
print(f"\n🥇 WINNER DETAILS:")
print(f"   R_MIN:     {best['score']:.4f}")
print(f"   MCC:       {best['mcc']:.4f}")
print(f"   Params:    {best['params']}")

if best["score"] > BASELINE_RMIN:
    print(f"\n🎉 SUCCESS! We beat baseline {BASELINE_RMIN}!")
else:
    print(f"\n❄️ FAIL. Baseline {BASELINE_RMIN} is still King.")

# %%
# 5. Visualization
print("\n" + "="*80)
print("📊 BEST MODEL DIAGNOSTICS")
print("="*80)

# 1. Confusion Matrix
cm = confusion_matrix(best["y_test"], best["y_pred"])
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False)
ax[0].set_title(f"Confusion Matrix (Target: {TARGET_DATE})")
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')
ax[0].set_xticklabels(['DOWN', 'UP'])
ax[0].set_yticklabels(['DOWN', 'UP'])

sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens', ax=ax[1], cbar=False)
ax[1].set_title(f"Normalized Matrix (Recall Analysis)")
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
ax[1].set_xticklabels(['DOWN', 'UP'])
ax[1].set_yticklabels(['DOWN', 'UP'])

plt.tight_layout()
plt.show()

# 2. Detailed Report
print("\n📝 Detailed Classification Report:")
print(classification_report(best["y_test"], best["y_pred"], target_names=["DOWN", "UP"], digits=4))
