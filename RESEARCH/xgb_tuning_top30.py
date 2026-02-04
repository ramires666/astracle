# %% [markdown]
# # XGBoost Hyperparameter Tuning (Phase 2)
# 
# Takes **Top 30** configs from Phase 1 and tunes XGBoost hyperparams.
# 
# **OPTIMIZED**: Pre-computes bodies, aspects, phases, labels ONCE.

# %% [markdown]
# ## Imports

# %%
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer, recall_score
from pathlib import Path
import ast
import warnings
from tqdm import tqdm

# %%
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', message='Falling back to prediction')

# %% [markdown]
# ## Project Root & Imports

# %%
current_dir = Path(os.getcwd())
if (current_dir / "RESEARCH").exists():
    PROJECT_ROOT = current_dir
elif (current_dir.parent / "RESEARCH").exists():
    PROJECT_ROOT = current_dir.parent
else:
    PROJECT_ROOT = current_dir

sys.path.append(str(PROJECT_ROOT))

# %%
from RESEARCH.data_loader import load_market_data
from RESEARCH.labeling import create_balanced_labels
from RESEARCH.astro_engine import (
    init_ephemeris,
    calculate_bodies_for_dates_multi,
    calculate_phases_for_dates,
)
from RESEARCH.astro.aspects import (
    precompute_angles_for_dates,
    calculate_aspects_from_cache,
)
from RESEARCH.numba_utils import warmup_jit, check_numba_available
from RESEARCH.features import build_full_features, merge_features_with_labels, get_feature_columns
from RESEARCH.model_training import split_dataset, prepare_xy, check_cuda_available, calc_metrics
from sklearn.utils.class_weight import compute_sample_weight

# %% [markdown]
# ## Configuration

# %%
REPORTS_DIR = PROJECT_ROOT / "RESEARCH" / "reports"
INPUT_CSV = REPORTS_DIR / "grid_search_partial.csv"
OUTPUT_CSV = REPORTS_DIR / "xgb_tuning_results.csv"

if not INPUT_CSV.exists():
    csv_files = sorted(REPORTS_DIR.glob("grid_search_*.csv"))
    if csv_files:
        INPUT_CSV = csv_files[-1]
        print(f"✅ Using: {INPUT_CSV.name}")

# %%
TEST_MODE = False  # False = full run | True = quick test

# %%
N_CANDIDATES = 5  # How many top candidates to tune

# %%
# Parameter grid (~150 combos, balanced between speed and coverage)
PARAM_GRID = {
    'n_estimators': [500, 700],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.03, 0.05],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 3],
    'scale_pos_weight': [1], 
}
# Total combos: 2 * 3 * 3 * 2 * 2 * 2 * 2 * 1 = 144 per candidate

# %%
use_cuda_check, _ = check_cuda_available()
N_JOBS = 1 if use_cuda_check else 4

# %%
def parse_list_string(s):
    if pd.isna(s) or s == 'None':
        return []
    try:
        return ast.literal_eval(s)
    except:
        return []

# %%
# Custom scorer: min(recall_up, recall_down)
# We want BOTH classes to be predicted well, not just one.
def recall_min_score(y_true, y_pred):
    recalls = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return min(recalls)  # Return the WORST recall

RECALL_MIN_SCORER = make_scorer(recall_min_score, greater_is_better=True)

# %%
# Threshold tuning for raw XGBClassifier (no wrapper)
# Tries thresholds from 0.1 to 0.9 and finds the one that maximizes recall_min
def tune_threshold_raw(model, X_val, y_val):
    """
    Find optimal probability threshold for best recall_min.
    Returns (best_threshold, best_recall_min)
    """
    proba = model.predict_proba(X_val)[:, 1]  # Probability of class 1 (UP)
    
    best_t = 0.5
    best_score = 0.0
    
    for t in np.linspace(0.1, 0.9, 41):  # 0.1, 0.12, 0.14, ..., 0.9
        pred = (proba >= t).astype(int)
        metrics = calc_metrics(y_val, pred, labels=[0, 1])
        if metrics['recall_min'] > best_score:
            best_score = metrics['recall_min']
            best_t = t
    
    return best_t, best_score

def predict_with_threshold(model, X, threshold=0.5):
    """Predict using custom probability threshold."""
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int)

# %% [markdown]
# ## Load Candidates

# %%
top_candidates = pd.DataFrame()

if not os.path.exists(INPUT_CSV):
    print(f"❌ File not found: {INPUT_CSV}")
else:
    df_results = pd.read_csv(INPUT_CSV)
    df_results = df_results.sort_values('recall_min', ascending=False)
    limit = 30 if not TEST_MODE else 2
    top_candidates = df_results.head(limit).copy()
    print(f"🏆 Loaded {len(top_candidates)} candidates")

# %% [markdown]
# ## Initialize

# %%
settings = init_ephemeris()
use_cuda, device = check_cuda_available()
print(f"🖥️ Device: {device}")

if check_numba_available():
    warmup_jit()

# %%
df_market = load_market_data()
df_market = df_market[df_market['date'] >= '2017-11-01'].reset_index(drop=True)
print(f"📈 Market: {len(df_market)} days")

# %% [markdown]
# ## 🚀 PRE-COMPUTE EVERYTHING
# 
# Calculate ONCE, reuse many times.

# %%
# Extract unique values from candidates
unique_coords = top_candidates['coord_mode'].unique().tolist()
unique_orbs = top_candidates['orb_mult'].unique().tolist()
unique_gauss = top_candidates[['gauss_window', 'gauss_std']].drop_duplicates().values.tolist()

print(f"📦 Unique coord modes: {unique_coords}")
print(f"📦 Unique orb mults: {unique_orbs}")
print(f"📦 Unique gauss params: {len(unique_gauss)} combinations")

# %% [markdown]
# ### 1. Pre-compute BODIES + ANGLES + PHASES (per coord_mode)

# %%
body_cache = {}  # coord_mode -> (df_bodies, geo_by_date, helio_by_date, angles_cache)
phase_cache = {}  # coord_mode -> df_phases

print("⏳ Pre-computing bodies, angles, phases...")

for coord in tqdm(unique_coords, desc="Coords"):
    # Bodies & Angles
    df_bodies, geo_by_date, helio_by_date = calculate_bodies_for_dates_multi(
        df_market['date'], settings, coord_mode=coord, progress=False
    )
    # Filter Chiron
    if 'Chiron' in df_bodies['body'].values:
        df_bodies = df_bodies[df_bodies['body'] != 'Chiron']
    for d in geo_by_date:
        geo_by_date[d] = [b for b in geo_by_date[d] if b.body != 'Chiron']
    if helio_by_date:
        for d in helio_by_date:
            helio_by_date[d] = [b for b in helio_by_date[d] if b.body != 'Chiron']
    
    angles_cache = precompute_angles_for_dates(geo_by_date, progress=False)
    body_cache[coord] = (df_bodies, geo_by_date, helio_by_date, angles_cache)
    
    # Phases (only depends on geo_by_date)
    df_phases = calculate_phases_for_dates(geo_by_date, progress=False)
    phase_cache[coord] = df_phases

print(f"✅ Bodies/Angles/Phases cached for {len(unique_coords)} coord modes")

# %% [markdown]
# ### 2. Pre-compute ASPECTS (per coord_mode + orb_mult)

# %%
aspect_cache = {}  # (coord_mode, orb_mult) -> df_aspects

print("⏳ Pre-computing aspects...")

for coord in unique_coords:
    _, geo_by_date, _, angles_cache = body_cache[coord]
    for orb in tqdm(unique_orbs, desc=f"Orbs ({coord})", leave=False):
        key = (coord, orb)
        df_aspects = calculate_aspects_from_cache(angles_cache, settings, orb_mult=orb, progress=False)
        aspect_cache[key] = df_aspects

print(f"✅ Aspects cached: {len(aspect_cache)} combinations")

# %% [markdown]
# ### 3. Pre-compute LABELS (per gauss_window + gauss_std)

# %%
label_cache = {}  # (gauss_window, gauss_std) -> df_labels

print("⏳ Pre-computing labels...")

for gw, gs in tqdm(unique_gauss, desc="Gauss params"):
    key = (int(gw), float(gs))
    df_labels = create_balanced_labels(
        df_market, horizon=1, move_share=0.5, 
        gauss_window=key[0], gauss_std=key[1], 
        price_mode='raw', label_mode='balanced_detrended',
        verbose=False
    )
    label_cache[key] = df_labels

print(f"✅ Labels cached: {len(label_cache)} combinations")

# %% [markdown]
# ## Tuning Loop (FAST - only assembly + training)

# %%
final_results = []

print("\n🚀 STARTING TUNING LOOP (pre-computed data)...")

for i, (_, row) in enumerate(top_candidates.head(N_CANDIDATES).iterrows()):
    coord = row['coord_mode']
    gw = int(row['gauss_window'])
    gs = float(row['gauss_std'])
    orb = row['orb_mult']
    excl_list = parse_list_string(row['exclude_bodies'])
    
    print(f"\n⚡ [{i+1}/{len(top_candidates)}] Coord={coord} GW={gw} Orb={orb}")
    
    # --- FAST: Get from cache ---
    df_bodies, geo_by_date, helio_by_date, _ = body_cache[coord]
    df_phases = phase_cache[coord]
    df_aspects = aspect_cache[(coord, orb)]
    df_labels = label_cache[(gw, gs)]
    
    # --- Build Features (fast merge) ---
    df_features = build_full_features(df_bodies, df_aspects, df_phases, exclude_bodies=excl_list)
    df_dataset = merge_features_with_labels(df_features, df_labels, verbose=False)
    
    # --- Split ---
    train_df, val_df, test_df = split_dataset(df_dataset, train_ratio=0.7, val_ratio=0.15)
    feature_cols = get_feature_columns(df_dataset)
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_val, y_val = prepare_xy(val_df, feature_cols)
    X_test, y_test = prepare_xy(test_df, feature_cols)  # TEST set for final evaluation!
    
    print(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # --- PredefinedSplit ---
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    test_fold = np.concatenate([np.full(len(X_train), -1), np.full(len(X_val), 0)])
    ps = PredefinedSplit(test_fold)
    
    # --- Manual Grid Search with tqdm (visible progress in Jupyter) ---
    from itertools import product
    
    # Generate all parameter combinations
    param_names = list(PARAM_GRID.keys())
    param_values = [PARAM_GRID[k] for k in param_names]
    all_combos = list(product(*param_values))
    
    best_score = -1
    best_params = None
    all_test_results = []  # Store ALL test results for intermediate saves
    
    for test_idx, combo in enumerate(tqdm(all_combos, desc=f"Candidate {i+1}", leave=False)):
        params = dict(zip(param_names, combo))
        
        model = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='logloss',
            device=device if device == 'cuda' else 'cpu', 
            tree_method='hist' if device == 'cuda' else 'auto',
            **params
        )
        model.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))
        y_pred_val = model.predict(X_val)
        score = recall_min_score(y_val, y_pred_val)
        
        # Store test result
        all_test_results.append({'params': params, 'recall_min': score})
        
        # Print EVERY test result
        is_best = "✨" if score > best_score else "  "
        print(f"   {is_best} d={params['max_depth']} lr={params['learning_rate']:.2f} sub={params['subsample']:.1f} γ={params.get('gamma', 0):.1f} → R_MIN={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_params = params
        
        # --- INTERMEDIATE SAVE every 50 tests ---
        if (test_idx + 1) % 50 == 0:
            interim_df = pd.DataFrame([{
                'candidate': i + 1, 'coord_mode': coord, 'gauss_window': gw, 'orb_mult': orb,
                'tests_done': test_idx + 1, 'best_recall_min': best_score,
                'best_params': str(best_params)
            }])
            interim_df.to_csv(OUTPUT_CSV.parent / f"xgb_interim_candidate_{i+1}.csv", index=False)
            print(f"   💾 Saved interim at {test_idx + 1} tests")
    
    # Skip if no valid params found
    if best_params is None:
        print(f"   ❌ No valid params found")
        continue
    
    # Retrain on X_train only
    final_model = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        device=device if device == 'cuda' else 'cpu', 
        tree_method='hist' if device == 'cuda' else 'auto',
        **best_params
    )
    final_model.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))
    
    # --- Tune threshold on VAL, evaluate on TEST (like Phase 1!) ---
    best_t_tuned, _ = tune_threshold_raw(final_model, X_val, y_val)
    y_pred_test = predict_with_threshold(final_model, X_test, threshold=best_t_tuned)
    metrics_tuned = calc_metrics(y_test, y_pred_test, labels=[0, 1])
    
    # --- LOCAL BASELINE: Default params, same process ---
    local_baseline = xgb.XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        n_estimators=500, max_depth=6, learning_rate=0.03,  # Phase 1 defaults
        device=device if device == 'cuda' else 'cpu', 
        tree_method='hist' if device == 'cuda' else 'auto',
    )
    local_baseline.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))
    best_t_base, _ = tune_threshold_raw(local_baseline, X_val, y_val)
    y_pred_base = predict_with_threshold(local_baseline, X_test, threshold=best_t_base)
    metrics_base = calc_metrics(y_test, y_pred_base, labels=[0, 1])
    
    print(f"   📊 LOCAL BASELINE: R_MIN={metrics_base['recall_min']:.3f} MCC={metrics_base['mcc']:.3f} (t={best_t_base:.2f})")
    print(f"   📈 TUNED:          R_MIN={metrics_tuned['recall_min']:.3f} MCC={metrics_tuned['mcc']:.3f} (t={best_t_tuned:.2f})")
    print(f"   📉 Phase1 (ref):   R_MIN={row['recall_min']:.3f} MCC={row['mcc']:.3f}")
    print(f"   🔧 Best: depth={best_params['max_depth']}, lr={best_params['learning_rate']:.3f}, sub={best_params['subsample']}, gamma={best_params.get('gamma', 0)}")
    
    record = {
        'rank': i + 1, 'coord_mode': coord, 'gauss_window': gw, 'gauss_std': gs,
        'orb_mult': orb, 'exclude_bodies': row['exclude_bodies'],
        'baseline_recall_min': metrics_base['recall_min'], 'baseline_mcc': metrics_base['mcc'],
        'tuned_recall_min': metrics_tuned['recall_min'], 'tuned_mcc': metrics_tuned['mcc'],
        'best_params': str(best_params),
    }
    final_results.append(record)
    pd.DataFrame(final_results).to_csv(OUTPUT_CSV, index=False)

# %%
print("=" * 60)
print(f"💾 DONE. Results: {OUTPUT_CSV}")
print("=" * 60)
