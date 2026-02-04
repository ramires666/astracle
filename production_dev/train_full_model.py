"""
Train FULL Model (Forecast Model)

This script trains the *forecast* model that is used to predict future dates.

Why we need a separate FULL model:
1) SPLIT model (backtest / history)
   - Exported by the research notebook: `RESEARCH/birthdate_deep_search.ipynb`
   - Trained on a time split (train/val/test) so its metrics are honest
   - File: `models_artifacts/btc_astro_predictor.joblib`

2) FULL model (forecast)
   - Trained on ALL available historical data
   - This is allowed because future dates are always unseen anyway
   - File: `models_artifacts/btc_astro_predictor.full.joblib`

IMPORTANT RULES (matching the project intent):
- We DO NOT modify the SPLIT model artifact here.
- We DO reuse the SAME pipeline and parameters as the SPLIT model so that:
  - Features match (same column names / order)
  - Forecast model is "the same model, just retrained on all data"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

# Add project root so we can import `src/` and `RESEARCH/`
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.xgb import XGBBaseline


SPLIT_MODEL_PATH = PROJECT_ROOT / "models_artifacts" / "btc_astro_predictor.joblib"
FULL_MODEL_PATH = PROJECT_ROOT / "models_artifacts" / "btc_astro_predictor.full.joblib"

# Local, file-based fallback (useful when DB is not available in dev)
LOCAL_MARKET_FALLBACK = PROJECT_ROOT / "data" / "market" / "processed" / "BTC_full_market_daily.parquet"


@dataclass(frozen=True)
class SplitReference:
    """
    A small "package" with everything we need from the split model.

    We use it as the source of truth for:
    - hyperparameters (n_estimators, max_depth, ...)
    - astro configuration (orb_mult, coord_mode, birth_date, ...)
    - feature columns order (feature_names)

    This prevents the FULL model from silently drifting away from research.
    """

    config: Dict
    feature_names: List[str]


def _load_split_reference(path: Path = SPLIT_MODEL_PATH) -> SplitReference:
    """
    Load split model artifact and extract config + feature_names.

    We intentionally do NOT use the split model itself for training here.
    We only read its settings so the FULL model stays consistent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Split model artifact not found at {path}. "
            "Run the research notebook export first."
        )

    artifact = joblib.load(path)
    cfg = dict(artifact.get("config", {}))
    feature_names = list(artifact.get("feature_names", []))

    if not feature_names:
        raise ValueError(
            "Split artifact has empty feature_names. "
            "Cannot train a compatible full model."
        )

    return SplitReference(config=cfg, feature_names=feature_names)


def _load_market_data_with_fallback(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load market data (date, close).

    Preferred source:
    - PostgreSQL via `RESEARCH.data_loader.load_market_data`

    Fallback source (for local development):
    - `data/market/processed/BTC_full_market_daily.parquet`

    We keep this logic here (inside production_dev) so RESEARCH stays "pure"
    and focused on database-first research workflows.
    """
    # 1) Try DB first (this is what production expects)
    try:
        from RESEARCH.data_loader import load_market_data

        df = load_market_data(start_date=start_date, end_date=end_date)
        df = df[["date", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print("⚠️ Could not load market data from DB. Falling back to local file.")
        print(f"   DB error: {e}")

    # 2) Fallback to local parquet
    if not LOCAL_MARKET_FALLBACK.exists():
        raise FileNotFoundError(
            f"Local market fallback file not found: {LOCAL_MARKET_FALLBACK}. "
            "Either configure the DB connection or provide a local parquet."
        )

    df = pd.read_parquet(LOCAL_MARKET_FALLBACK)
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    return df.reset_index(drop=True)


def _build_full_training_dataset(
    df_market: pd.DataFrame,
    split_ref: SplitReference,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the exact same dataset format as in the research notebook.

    High-level steps (simple explanation):
    1) We take every calendar day and calculate astro features for that day.
    2) We create a target label (UP/DOWN) from market prices.
    3) We merge features + labels into one big table.

    Returns:
        (df_dataset, feature_cols)
        - df_dataset has columns: date, target, ...features...
        - feature_cols is the list of feature names in the order we will use
    """
    # We import research modules locally, so this file can still be imported
    # by tooling even if some heavy deps are missing.
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

    cfg = split_ref.config

    # ---------------------------------------------------------------------
    # 1) Labels (target)
    # ---------------------------------------------------------------------
    # The research notebook (birthdate_deep_search.ipynb) calls:
    #   create_balanced_labels(df_market, ASTRO_CONFIG["gauss_window"], ASTRO_CONFIG["gauss_std"])
    #
    # With the CURRENT function signature, those positional arguments map to:
    #   horizon = gauss_window
    #   move_share = gauss_std
    #
    # This is confusing naming, but it is what the exported split model
    # was trained on, and we must keep FULL model training consistent with it.
    #
    # We intentionally do NOT pass `gauss_window` / `gauss_std` keyword args here,
    # so the labeling detrending uses the default values from `configs/labels.yaml`.
    df_labels = create_balanced_labels(
        df_market,
        horizon=cfg.get("gauss_window"),
        move_share=cfg.get("gauss_std"),
        verbose=True,
    )

    # ---------------------------------------------------------------------
    # 2) Astro feature blocks (bodies, aspects, transits, phases)
    # ---------------------------------------------------------------------
    settings = init_ephemeris()

    coord_mode = cfg.get("coord_mode", "both")
    orb_mult = cfg.get("orb_mult", 0.1)
    exclude_bodies = cfg.get("exclude_bodies")

    print("🪐 Calculating bodies for all dates (this can take a while)...")
    df_bodies, geo_by_date, _helio_by_date = calculate_bodies_for_dates_multi(
        df_market["date"],
        settings,
        coord_mode=coord_mode,
        progress=True,
    )

    # We use geocentric bodies for aspects/transits (same as notebook)
    bodies_by_date = geo_by_date

    print("🌙 Calculating phases...")
    df_phases = calculate_phases_for_dates(bodies_by_date, progress=False)

    print("🧬 Building natal chart (static, one-time)...")
    birth_date = cfg.get("birth_date", "2009-10-10")
    natal_dt_str = f"{birth_date}T12:00:00"  # noon avoids timezone edge cases
    natal_bodies = get_natal_bodies(natal_dt_str, settings)

    print("🧲 Calculating transits (current sky -> natal sky)...")
    df_transits = calculate_transits_for_dates(
        bodies_by_date,
        natal_bodies,
        settings,
        orb_mult=orb_mult,
        progress=False,
    )

    print("📐 Calculating aspects (current sky -> current sky)...")
    df_aspects = calculate_aspects_for_dates(
        bodies_by_date,
        settings,
        orb_mult=orb_mult,
        progress=False,
    )

    print("🧱 Building full feature matrix...")
    df_features = build_full_features(
        df_bodies,
        df_aspects,
        df_transits=df_transits,
        df_phases=df_phases,
        include_pair_aspects=True,
        include_transit_aspects=True,
        exclude_bodies=exclude_bodies,
    )

    print("🔗 Merging features + labels (forward-fill labels to all days)...")
    df_dataset = merge_features_with_labels(df_features, df_labels, verbose=True)

    # ---------------------------------------------------------------------
    # 3) Ensure feature columns match the SPLIT model (order matters!)
    # ---------------------------------------------------------------------
    # The split model was trained on a very specific list of feature columns.
    # If we change the order or drop columns, the model becomes incompatible.
    #
    # So we reindex the FULL dataset to EXACTLY the split feature list.
    feature_cols = split_ref.feature_names
    missing = [c for c in feature_cols if c not in df_dataset.columns]
    extra = [c for c in df_dataset.columns if c not in {"date", "target"} and c not in set(feature_cols)]

    if missing:
        # Missing features are filled with 0.0 (means "feature not active")
        # This is safe because the research pipeline also uses 0 for inactive aspects.
        print(f"⚠️ FULL dataset is missing {len(missing)} split features. Filling with zeros.")

        # IMPORTANT PERFORMANCE NOTE:
        # Adding thousands of columns one-by-one makes pandas create a highly
        # fragmented dataframe (slow and memory-heavy).
        #
        # Instead, we create one "zero block" dataframe and concat once.
        zero_block = pd.DataFrame(0.0, index=df_dataset.index, columns=missing)
        df_dataset = pd.concat([df_dataset, zero_block], axis=1)

    if extra:
        # Extra features can appear if we build features on a different date range.
        # We drop them to keep the full model 100% compatible with the split model.
        print(f"ℹ️ FULL dataset has {len(extra)} extra features not in split model. Dropping them.")

    # Final, ordered feature set
    df_dataset = df_dataset[["date", "target", *feature_cols]].copy()

    return df_dataset, feature_cols


def _train_xgb_baseline_full(
    df_dataset: pd.DataFrame,
    feature_cols: List[str],
    split_ref: SplitReference,
) -> XGBBaseline:
    """
    Train the FULL model on ALL rows in df_dataset.

    Key points:
    - We use the SAME XGB hyperparameters as the split model.
    - We DO NOT do a train/val split here because we want to use *all* data.
    - We DO keep class balancing via sample weights (important for recall_min).
    """
    from RESEARCH.model_training import check_cuda_available, prepare_xy

    # Convert DataFrame -> numpy arrays (same helper as research)
    X, y = prepare_xy(df_dataset, feature_cols)

    # Sample weights: make DOWN and UP equally important
    weights = compute_sample_weight(class_weight="balanced", y=y)

    # Device selection (cpu/cuda). Most servers will be CPU.
    _use_cuda, device = check_cuda_available()

    cfg = split_ref.config

    # Hyperparameters copied from split model config
    # (So "full model" really is the same model, just trained on more data.)
    model = XGBBaseline(
        n_classes=2,
        device=device,
        random_state=42,
        early_stopping_rounds=None,  # no validation split, so early stopping is disabled
        n_estimators=int(cfg.get("n_estimators", 500)),
        max_depth=int(cfg.get("max_depth", 6)),
        learning_rate=float(cfg.get("learning_rate", 0.03)),
        colsample_bytree=float(cfg.get("colsample_bytree", 0.6)),
        subsample=float(cfg.get("subsample", 0.8)),
        tree_method="hist",
    )

    print("🏋️ Training FULL model on ALL available data...")
    model.fit(
        X,
        y,
        X_val=None,
        y_val=None,
        feature_names=feature_cols,
        sample_weight=weights,
        sample_weight_val=None,
    )

    return model


def train_final_model(
    output_path: Path = FULL_MODEL_PATH,
    split_model_path: Path = SPLIT_MODEL_PATH,
    start_date: str = "2017-11-01",
) -> Path:
    """
    Public entry point (used by daily_retrain.py).

    Args:
        output_path: Where to write FULL artifact (.joblib)
        split_model_path: Split model to read config/features from
        start_date: We typically start from 2017+ (more stable market structure)

    Returns:
        Path to saved artifact.
    """
    print("=" * 70)
    print("🚀 Training FULL Forecast Model (btc_astro_predictor.full.joblib)")
    print("=" * 70)

    split_ref = _load_split_reference(split_model_path)
    df_market = _load_market_data_with_fallback(start_date=start_date)

    print(f"📈 Market rows: {len(df_market)}")
    if len(df_market) > 0:
        print(f"   Date range: {df_market['date'].min().date()} -> {df_market['date'].max().date()}")

    df_dataset, feature_cols = _build_full_training_dataset(df_market, split_ref)
    model = _train_xgb_baseline_full(df_dataset, feature_cols, split_ref)

    # Decision threshold:
    # If a tuned threshold exists in split config, reuse it.
    # Otherwise default to 0.5 (standard).
    decision_threshold = float(
        split_ref.config.get("decision_threshold", split_ref.config.get("threshold", 0.5))
    )

    artifact = {
        "model": model,
        "feature_names": feature_cols,
        "config": {
            # Keep the same astro + ML settings as split model
            **split_ref.config,
            # Explicitly store threshold used at inference
            "decision_threshold": decision_threshold,
        },
        "training_date_range": (
            df_dataset["date"].min().strftime("%Y-%m-%d"),
            df_dataset["date"].max().strftime("%Y-%m-%d"),
        ),
        "train_samples": int(len(df_dataset)),
        "trained_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    print("✅ FULL model saved")
    print(f"   Path: {output_path}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Threshold: {decision_threshold}")
    print(f"   Samples: {artifact['train_samples']}")
    print(f"   Range: {artifact['training_date_range'][0]} -> {artifact['training_date_range'][1]}")

    return output_path


if __name__ == "__main__":
    train_final_model()
