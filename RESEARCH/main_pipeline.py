# %% [markdown]
# # Astro Trading Pipeline - Modular Research Notebook
# 
# This is the **central orchestrating notebook** that imports and runs all modules.
# 
# ## Pipeline Steps:
# 1. Setup & Config
# 2. Load Market Data (from PostgreSQL)
# 3. Visualize Price
# 4. Create Labels (balanced UP/DOWN)
# 5. Compute Astro Data
# 6. Build Features
# 7. Train Model
# 8. Evaluate & Visualize
# 9. (Optional) Grid Search
# 10. Save Model

# %% [markdown]
# ## 0. Environment Check

# %%
# Check dependencies
import importlib.util as iu

required = ["xgboost", "sklearn", "matplotlib", "seaborn", "tqdm", "pyarrow", "psycopg2", "swisseph"]
missing = [pkg for pkg in required if iu.find_spec(pkg) is None]

if missing:
    print("Missing packages:", ", ".join(missing))
    print("Install with: conda install -c conda-forge " + " ".join(missing))
else:
    print("✓ All dependencies found")

# %% [markdown]
# ## 1. Setup & Configuration

# %%
# Import RESEARCH modules
import sys
from pathlib import Path

# Find project root and add to path
PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / "RESEARCH").exists():
    for parent in PROJECT_ROOT.parents:
        if (parent / "RESEARCH").exists():
            PROJECT_ROOT = parent
            break

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# %%
# Import all modules
from RESEARCH.config import cfg, PROJECT_ROOT
from RESEARCH.data_loader import load_market_data, get_latest_date, get_data_paths
from RESEARCH.labeling import create_balanced_labels, gaussian_smooth_centered
from RESEARCH.astro_engine import (
    init_ephemeris,
    calculate_bodies_for_dates,
    calculate_aspects_for_dates,
    calculate_transits_for_dates,
    get_natal_bodies,
)
from RESEARCH.features import (
    build_full_features,
    merge_features_with_labels,
    get_feature_columns,
    get_feature_inventory,
)
from RESEARCH.model_training import (
    split_dataset,
    prepare_xy,
    train_xgb_model,
    tune_threshold,
    predict_with_threshold,
    evaluate_model,
    get_feature_importance,
    check_cuda_available,
)
from RESEARCH.visualization import (
    plot_price_distribution,
    plot_class_distribution,
    plot_price_with_labels,
    plot_last_n_days,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_predictions,
)

print("✓ All RESEARCH modules imported")

# %%
# Show configuration
print(f"Active subject: {cfg.active_subject_id}")
print(f"Data root: {cfg.data_root}")
print(f"DB URL configured: {bool(cfg.db_url)}")
print(f"Label config: {cfg.get_label_config()}")

# %% [markdown]
# ## 2. Load Market Data

# %%
# Load market data from database
df_market = load_market_data()

# Optional: filter by start date
DATA_START = "2017-11-01"
df_market = df_market[df_market["date"] >= DATA_START].reset_index(drop=True)

print(f"\nMarket data: {len(df_market)} rows")
print(f"Date range: {df_market['date'].min().date()} → {df_market['date'].max().date()}")
df_market.head()

# %% [markdown]
# ## 3. Visualize Price

# %%
# Price and return distribution
plot_price_distribution(df_market, price_mode="log")

# %% [markdown]
# ## 4. Create Labels

# %%
# Configuration for labeling
LABEL_CONFIG = {
    "horizon": 1,           # Prediction horizon (days)
    "move_share": 0.5,      # Total share of samples to keep
    "gauss_window": 201,    # Gaussian window for detrending (odd)
    "gauss_std": 50.0,      # Gaussian std
    "price_mode": "raw",    # 'raw' or 'log'
    "label_mode": "balanced_detrended",
}

# Create balanced labels
df_labels = create_balanced_labels(
    df_market,
    horizon=LABEL_CONFIG["horizon"],
    move_share=LABEL_CONFIG["move_share"],
    gauss_window=LABEL_CONFIG["gauss_window"],
    gauss_std=LABEL_CONFIG["gauss_std"],
    price_mode=LABEL_CONFIG["price_mode"],
    label_mode=LABEL_CONFIG["label_mode"],
)

df_labels.head()

# %%
# Class distribution
plot_class_distribution(df_labels)

# %%
# Price with labels
plot_price_with_labels(df_market, df_labels, price_mode="raw")

# %%
# Last 30 days
plot_last_n_days(df_market, df_labels, n_days=30)

# %% [markdown]
# ## 5. Compute Astro Data

# %%
# Initialize ephemeris
settings = init_ephemeris()
print(f"Bodies: {[b.name for b in settings.bodies]}")
print(f"Aspects: {[a.name for a in settings.aspects]}")

# %%
# Calculate body positions for all dates
# (This is fast - no need to cache)
df_bodies, bodies_by_date = calculate_bodies_for_dates(
    df_market["date"],
    settings,
    progress=True,
)

print(f"\nBodies calculated: {len(df_bodies)} rows")
df_bodies.head()

# %%
# Calculate aspects
ORB_MULTIPLIER = 1.0  # Adjust orbs (1.0 = default)

df_aspects = calculate_aspects_for_dates(
    bodies_by_date,
    settings,
    orb_mult=ORB_MULTIPLIER,
    progress=True,
)

print(f"\nAspects: {len(df_aspects)} rows")
df_aspects.head()

# %% [markdown]
# ## 6. Build Features

# %%
# Build feature matrix
df_features = build_full_features(
    df_bodies,
    df_aspects,
    df_transits=None,  # Add transit aspects if needed
    include_pair_aspects=True,
    include_transit_aspects=False,
    exclude_bodies=None,  # List of body names to exclude
)

print(f"Features shape: {df_features.shape}")
df_features.head()

# %%
# Merge with labels
df_dataset = merge_features_with_labels(df_features, df_labels)
print(f"\nDataset shape: {df_dataset.shape}")

# %%
# Feature inventory
feature_inventory = get_feature_inventory(df_dataset)
print("\nFeature groups:")
print(feature_inventory.groupby("group").size())

# %% [markdown]
# ## 7. Train Model

# %%
# Check CUDA availability
use_cuda, device = check_cuda_available()
print(f"Using device: {device}")

# %%
# Split dataset (time-based)
train_df, val_df, test_df = split_dataset(df_dataset, train_ratio=0.7, val_ratio=0.15)

print(f"Train: {train_df['date'].min().date()} → {train_df['date'].max().date()}")
print(f"Val:   {val_df['date'].min().date()} → {val_df['date'].max().date()}")
print(f"Test:  {test_df['date'].min().date()} → {test_df['date'].max().date()}")

# %%
# Prepare X, y
feature_cols = get_feature_columns(df_dataset)
X_train, y_train = prepare_xy(train_df, feature_cols)
X_val, y_val = prepare_xy(val_df, feature_cols)
X_test, y_test = prepare_xy(test_df, feature_cols)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

# %%
# Train XGBoost model
MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

model = train_xgb_model(
    X_train, y_train,
    X_val, y_val,
    feature_cols,
    n_classes=2,
    device=device,
    **MODEL_PARAMS,
)

# %%
# Tune threshold on validation set
best_threshold, best_score = tune_threshold(model, X_val, y_val, metric="bal_acc")

# %% [markdown]
# ## 8. Evaluate & Visualize

# %%
# Predict on test set
y_pred = predict_with_threshold(model, X_test, threshold=best_threshold)

# Evaluate
results = evaluate_model(y_test, y_pred, label_names=["DOWN", "UP"])

# %%
# Confusion matrix
plot_confusion_matrix(y_test, y_pred)

# %%
# Feature importance
imp_df = get_feature_importance(model, feature_cols, top_n=20)
plot_feature_importance(imp_df)

# %%
# Predictions on test set
plot_predictions(test_df, y_pred, y_true=y_test, price_mode="log")

# %% [markdown]
# ## 9. (Optional) Grid Search
# 
# Uncomment the cell below to run grid search over orb and gaussian parameters.

# %%
# # GRID SEARCH (uncomment to run)
# from RESEARCH.grid_search import run_grid_search, GridSearchConfig, get_best_params
# 
# grid_config = GridSearchConfig(
#     orb_multipliers=[0.8, 1.0, 1.2],
#     gauss_windows=[101, 151, 201],
#     gauss_stds=[30.0, 50.0, 70.0],
# )
# 
# results_df = run_grid_search(df_market, config=grid_config)
# best_params = get_best_params(results_df)
# print("Best params:", best_params)

# %% [markdown]
# ## 10. Save Model

# %%
# Save model
from joblib import dump

artifact_dir = PROJECT_ROOT / "models_artifacts"
artifact_dir.mkdir(parents=True, exist_ok=True)

artifact = {
    "model": model.model,
    "scaler": model.scaler,
    "feature_names": feature_cols,
    "threshold": best_threshold,
    "config": {
        "label_config": LABEL_CONFIG,
        "orb_multiplier": ORB_MULTIPLIER,
        "model_params": MODEL_PARAMS,
    },
}

out_path = artifact_dir / f"xgb_astro_research.joblib"
dump(artifact, out_path)
print(f"✓ Model saved: {out_path}")

# %% [markdown]
# ---
# ## Summary
# 
# This modular pipeline makes it easy to:
# - **Debug**: Each module can be tested independently
# - **Extend**: Add new features, models, or visualizations
# - **Experiment**: Quickly try different configurations
# 
# ### TODO:
# - [ ] Add moon phases to features
# - [ ] Grid search for astro body exclusion
# - [ ] Add houses for birth date grid search
# - [ ] Save best grid search results
