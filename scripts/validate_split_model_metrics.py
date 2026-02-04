#!/usr/bin/env python3
"""
Validate Split-Model Metrics Against The Research Notebook

This script is a "sanity checker" for the exact problem you described:

> "The frontend backtest looks worse than the model from birthdate_deep_search."

What it does (high-level, plain words):
1) Loads the SPLIT model artifact:
   `models_artifacts/btc_astro_predictor.joblib`
2) Rebuilds the dataset using the SAME pipeline as the notebook:
   - Swiss Ephemeris bodies
   - aspects + transits + phases
   - labels created by `RESEARCH.labeling.create_balanced_labels`
   - label forward-fill like `RESEARCH.features.merge_features_with_labels`
   - time-based Train/Val/Test split (70/15/15)
3) Tunes the probability threshold on the validation slice (recall_min)
4) Evaluates on the test slice and prints:
   - R_MIN
   - MCC

Why this is useful:
- If this script matches the notebook numbers, but the dashboard doesn't,
  then the bug is in the production inference / cache generation logic.
- If this script does NOT match, the most common reason is:
  the market dataset changed after the model was exported (new rows added).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, matthews_corrcoef


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# We reuse the exact same market-data loading fallback logic that the production
# full-model training uses (DB first, then local parquet).
from production_dev.train_full_model import _load_market_data_with_fallback


def _load_split_artifact(path: Path) -> Tuple[object, Dict, List[str]]:
    """
    Load the split artifact and return (model, config, feature_names).

    The artifact format is produced by the notebook export cell and looks like:
      {
        "model": XGBBaseline(...),
        "feature_names": [...],
        "config": {...}
      }
    """
    if not path.exists():
        raise FileNotFoundError(f"Split model artifact not found: {path}")

    artifact = joblib.load(path)
    model = artifact["model"]
    config = dict(artifact.get("config", {}))
    feature_names = list(artifact.get("feature_names", []))

    if not feature_names:
        raise ValueError("Artifact feature_names is empty. Cannot evaluate reliably.")

    return model, config, feature_names


def _build_dataset(df_market: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Rebuild the dataset in the same way the research notebook does.

    This is intentionally "boring and explicit" to avoid hidden mismatches.
    """
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

    # 1) Labels (target)
    #
    # IMPORTANT: Notebook-exact behavior
    # The research notebook calls:
    #   create_balanced_labels(df_market, ASTRO_CONFIG["gauss_window"], ASTRO_CONFIG["gauss_std"])
    #
    # With the current function signature, those positional arguments map to:
    # - horizon    = gauss_window
    # - move_share = gauss_std
    #
    # The naming is confusing, but this is what the exported split model
    # was trained on, so this validation script must do the same to match
    # the artifact metrics.
    df_labels = create_balanced_labels(
        df_market,
        horizon=cfg.get("gauss_window"),
        move_share=cfg.get("gauss_std"),
        verbose=True,
    )

    # 2) Astro features
    settings = init_ephemeris()

    coord_mode = cfg.get("coord_mode", "both")
    orb_mult = cfg.get("orb_mult", 0.1)
    exclude_bodies = cfg.get("exclude_bodies")

    print("🪐 Calculating bodies (this is the slowest step)...")
    df_bodies, geo_by_date, _helio_by_date = calculate_bodies_for_dates_multi(
        df_market["date"],
        settings,
        coord_mode=coord_mode,
        progress=True,
    )

    bodies_by_date = geo_by_date

    print("🌙 Calculating phases...")
    df_phases = calculate_phases_for_dates(bodies_by_date, progress=False)

    print("🧬 Building natal chart (static)...")
    birth_date = cfg.get("birth_date", "2009-10-10")
    natal_dt_str = f"{birth_date}T12:00:00"
    natal_bodies = get_natal_bodies(natal_dt_str, settings)

    print("🧲 Calculating transits...")
    df_transits = calculate_transits_for_dates(
        bodies_by_date,
        natal_bodies,
        settings,
        orb_mult=orb_mult,
        progress=False,
    )

    print("📐 Calculating aspects...")
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

    print("🔗 Merging features + labels (forward-fill)...")
    df_dataset = merge_features_with_labels(df_features, df_labels, verbose=True)

    return df_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "models_artifacts" / "btc_astro_predictor.joblib"),
        help="Path to split model artifact (.joblib).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2017-11-01",
        help="Start date for market data (YYYY-MM-DD). Must match notebook.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date for market data (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Train ratio for time split (must match notebook).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation ratio for time split (must match notebook).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Allowed absolute difference between computed and stored metrics.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    model, cfg, feature_names = _load_split_artifact(model_path)

    print("=" * 80)
    print("🔍 SPLIT MODEL METRICS VALIDATION")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Stored (artifact) r_min={cfg.get('r_min')} mcc={cfg.get('mcc')}")

    # Market data (DB or local fallback)
    df_market = _load_market_data_with_fallback(start_date=args.start_date, end_date=args.end_date)
    df_market = df_market[["date", "close"]].copy()
    df_market["date"] = pd.to_datetime(df_market["date"])
    df_market = df_market.sort_values("date").reset_index(drop=True)

    print(f"Market rows: {len(df_market)}")
    if len(df_market) > 0:
        print(f"Date range: {df_market['date'].min().date()} -> {df_market['date'].max().date()}")

    # Build dataset (features + target)
    df_dataset = _build_dataset(df_market, cfg)

    # Reindex to the exact feature order stored in the artifact.
    # This is critical: even a correct model will look "wrong" if columns are shuffled.
    missing = [c for c in feature_names if c not in df_dataset.columns]
    if missing:
        print(f"⚠️ Dataset is missing {len(missing)} artifact features. Filling with zeros.")
        for c in missing:
            df_dataset[c] = 0.0

    # Drop extra features to match the model's expected input shape exactly.
    df_dataset = df_dataset[["date", "target", *feature_names]].copy()

    # Split
    from RESEARCH.model_training import split_dataset, tune_threshold, predict_with_threshold

    train_df, val_df, test_df = split_dataset(
        df_dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    # Prepare X/y (use numpy arrays, the same way the notebook does)
    X_val = val_df[feature_names].to_numpy(dtype=np.float64)
    y_val = val_df["target"].to_numpy(dtype=np.int32)
    X_test = test_df[feature_names].to_numpy(dtype=np.float64)
    y_test = test_df["target"].to_numpy(dtype=np.int32)

    # Threshold tuning on validation, then evaluation on test
    best_t, _ = tune_threshold(model, X_val, y_val, metric="recall_min", verbose=True)
    y_pred = predict_with_threshold(model, X_test, threshold=best_t)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    r_min = float(min(report["0"]["recall"], report["1"]["recall"]))
    mcc = float(matthews_corrcoef(y_test, y_pred))

    print("-" * 80)
    print(f"Computed (test) r_min={r_min:.6f} mcc={mcc:.6f} threshold={best_t:.3f}")

    stored_r_min = cfg.get("r_min")
    stored_mcc = cfg.get("mcc")

    # If stored metrics exist, compare them.
    if stored_r_min is not None and stored_mcc is not None:
        dr = abs(float(stored_r_min) - r_min)
        dm = abs(float(stored_mcc) - mcc)
        print(f"Delta vs artifact: |dr_min|={dr:.6g} |dmcc|={dm:.6g}")

        if dr <= args.tolerance and dm <= args.tolerance:
            print("✅ MATCH: computed metrics match the artifact (within tolerance).")
            return 0

        print("❌ MISMATCH: computed metrics do NOT match the artifact.")
        print("Most common reasons:")
        print("- New market data rows were added after the model was exported (split boundaries changed).")
        print("- Different start/end date than the notebook used.")
        print("- Different labeling config (gauss_window/std/horizon/move_share).")
        return 2

    print("ℹ️ Artifact does not contain stored metrics. Nothing to compare against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
