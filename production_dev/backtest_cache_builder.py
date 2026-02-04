"""
Backtest Cache Builder (Research-Exact)

This module exists for one reason:
make the *frontend history/backtest* match the research notebook metrics 1:1.

Why we cannot just "predict yesterday -> compare with tomorrow price":
- The research notebook does NOT train on simple next-day returns.
- It creates a special binary target from market data using:
  1) Gaussian detrending
  2) "balanced" selection of strong UP/DOWN moves
  3) FORWARD-FILL of those labels to ALL calendar days
     (see `RESEARCH.features.merge_features_with_labels`)

So, if the dashboard computes accuracy against next-day price change,
it will often look *much worse* than the notebook (apples vs oranges).

What we do here:
1) Load the SPLIT artifact exported by `RESEARCH/birthdate_deep_search.ipynb`
2) Rebuild the exact same daily dataset (features + forward-filled labels)
3) Split into Train/Val/Test by time (70/15/15 by default)
4) Tune the probability threshold on the VALIDATION slice (recall_min)
5) Run inference for the FULL timeline and cache:
   - prediction direction + confidence
   - actual label used by the notebook
   - correctness
   - split tag ("train" / "val" / "test")
   - actual close price (for the chart line)

Important rule:
- We do NOT retrain the split model here. Only inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


# We reuse the same "DB first, then local parquet" market loading logic that
# the FULL-model training already uses. This keeps behavior consistent.
from production_dev.train_full_model import (
    SplitReference,
    _build_full_training_dataset,
    _load_market_data_with_fallback,
)


@dataclass(frozen=True)
class BacktestBuildResult:
    """
    Container returned by `build_split_model_backtest_cache`.

    df_backtest:
      A daily table (one row per calendar day) with:
      - date
      - split (train/val/test)
      - direction (model prediction)
      - confidence (probability of the predicted class)
      - actual_direction (research target label)
      - correct (direction == actual_direction)
      - actual_price (close price for the date)

    meta:
      Small metadata that is useful for the UI (split boundaries, threshold, ...).
      This metadata is NOT required to reproduce metrics (the dataframe is enough),
      but it helps the frontend draw honest labels.
    """

    df_backtest: pd.DataFrame
    meta: Dict


def _extract_split_artifact(path: Path) -> Tuple[object, Dict, list[str]]:
    """
    Read the split artifact (.joblib) exported by the notebook.

    The file format is expected to be:
      {
        "model": XGBBaseline(...),
        "feature_names": [...],
        "config": {...}
      }
    """
    if not path.exists():
        raise FileNotFoundError(f"Split model artifact not found: {path}")

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise TypeError("Split artifact has unexpected format (expected dict).")

    model = artifact.get("model")
    feature_names = list(artifact.get("feature_names", []))
    cfg = dict(artifact.get("config", {}))

    if model is None:
        raise ValueError("Split artifact missing 'model'.")
    if not feature_names:
        raise ValueError("Split artifact missing 'feature_names' (empty list).")

    return model, cfg, feature_names


def _predict_proba_up(model: object, X: np.ndarray) -> np.ndarray:
    """
    Predict P(UP) for a binary model.

    We support two model shapes found in this repo:
    1) XGBBaseline wrapper used in research (has `.scaler` + `.model`)
    2) Plain XGBoost-like estimator with `.predict_proba`
    """
    # Research wrapper: scale then predict
    if hasattr(model, "scaler") and hasattr(model, "model"):
        X_scaled = model.scaler.transform(X)
        proba = model.model.predict_proba(X_scaled)
        return np.asarray(proba)[:, 1]

    # Plain estimator: assume it was trained on raw features
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return np.asarray(proba)[:, 1]

    raise TypeError("Unsupported model type: cannot compute predict_proba.")


def build_split_model_backtest_cache(
    split_model_path: Path,
    start_date: str = "2017-11-01",
    end_date: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> BacktestBuildResult:
    """
    Build the full backtest cache for the SPLIT model.

    Args:
        split_model_path: Path to `btc_astro_predictor.joblib`
        start_date: Market start date (must match notebook to match metrics)
        end_date: Optional market end date (mostly for debugging)
        train_ratio: Time split train ratio (notebook default: 0.7)
        val_ratio: Time split val ratio (notebook default: 0.15)

    Returns:
        BacktestBuildResult with dataframe + metadata.
    """
    # ------------------------------------------------------------------
    # 1) Load artifact (model + config + feature list)
    # ------------------------------------------------------------------
    model, cfg, feature_names = _extract_split_artifact(split_model_path)

    # ------------------------------------------------------------------
    # 2) Load market data (DB first, then local parquet fallback)
    # ------------------------------------------------------------------
    df_market = _load_market_data_with_fallback(start_date=start_date, end_date=end_date)
    df_market = df_market[["date", "close"]].copy()
    df_market["date"] = pd.to_datetime(df_market["date"])
    df_market = df_market.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3) Rebuild the exact same daily dataset as the notebook
    # ------------------------------------------------------------------
    # We reuse the dataset builder from `train_full_model.py` because it already:
    # - reproduces the NOTEBOOK label creation call exactly (see comments there)
    # - computes astro features for ALL dates
    # - forward-fills labels to ALL days (so target exists daily)
    # - reindexes features to the artifact feature order
    split_ref = SplitReference(config=cfg, feature_names=feature_names)
    df_dataset, feature_cols = _build_full_training_dataset(df_market, split_ref)

    # ------------------------------------------------------------------
    # 4) Split into train/val/test and tune threshold on validation
    # ------------------------------------------------------------------
    from RESEARCH.model_training import split_dataset, tune_threshold

    train_df, val_df, test_df = split_dataset(
        df_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    y_val = val_df["target"].to_numpy(dtype=np.int32)

    # Threshold is tuned to maximize recall_min (best "worst recall" between classes).
    best_t, best_score = tune_threshold(model, X_val, y_val, metric="recall_min", verbose=True)

    # ------------------------------------------------------------------
    # 5) Inference for ALL days (train+val+test)
    # ------------------------------------------------------------------
    X_all = df_dataset[feature_cols].to_numpy(dtype=np.float64)
    y_true = df_dataset["target"].to_numpy(dtype=np.int32)

    proba_up = _predict_proba_up(model, X_all)
    y_pred = (proba_up >= float(best_t)).astype(np.int32)

    # Confidence = probability of the predicted class
    # - if we predicted UP (1): confidence = P(UP)
    # - if we predicted DOWN (0): confidence = P(DOWN) = 1 - P(UP)
    confidence = np.where(y_pred == 1, proba_up, 1.0 - proba_up)

    # Split tags for each row, in the same order as df_dataset
    n_train = len(train_df)
    n_val = len(val_df)
    n_total = len(df_dataset)
    split_tags = np.array(["test"] * n_total, dtype=object)
    split_tags[:n_train] = "train"
    split_tags[n_train : n_train + n_val] = "val"

    # ------------------------------------------------------------------
    # 6) Build final dataframe for caching and frontend usage
    # ------------------------------------------------------------------
    df_backtest = pd.DataFrame(
        {
            "date": pd.to_datetime(df_dataset["date"]),
            "split": split_tags,
            "direction_code": y_pred.astype(int),
            "direction": np.where(y_pred == 1, "UP", "DOWN"),
            "confidence": confidence.astype(float),
            "proba_up": proba_up.astype(float),
            "actual_direction_code": y_true.astype(int),
            "actual_direction": np.where(y_true == 1, "UP", "DOWN"),
            "correct": (y_pred == y_true),
            # Store the tuned threshold as a constant column for transparency.
            # This is useful when someone wants to verify the cache later.
            "decision_threshold": float(best_t),
        }
    )

    # Attach the actual close price for each date (needed for the chart line).
    # This is a simple left join; dates should match 1:1 for daily market data.
    df_backtest = pd.merge(
        df_backtest,
        df_market.rename(columns={"close": "actual_price"})[["date", "actual_price"]],
        on="date",
        how="left",
    )

    df_backtest = df_backtest.sort_values("date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 7) Metadata: split boundaries and tuning information
    # ------------------------------------------------------------------
    meta = {
        "start_date": df_backtest["date"].min().strftime("%Y-%m-%d") if len(df_backtest) else None,
        "end_date": df_backtest["date"].max().strftime("%Y-%m-%d") if len(df_backtest) else None,
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        # Notebook-exact label settings:
        # In the notebook export config these are stored under confusing names
        # (`gauss_window`, `gauss_std`), but they are passed positionally into
        # create_balanced_labels() and therefore become (horizon, move_share).
        "label_horizon_days": int(cfg["gauss_window"]) if cfg.get("gauss_window") is not None else None,
        "label_move_share": float(cfg["gauss_std"]) if cfg.get("gauss_std") is not None else None,
        "decision_threshold": float(best_t),
        "val_recall_min_at_threshold": float(best_score),
        "splits": {
            "train": {
                "start": train_df["date"].min().strftime("%Y-%m-%d") if len(train_df) else None,
                "end": train_df["date"].max().strftime("%Y-%m-%d") if len(train_df) else None,
                "rows": int(len(train_df)),
            },
            "val": {
                "start": val_df["date"].min().strftime("%Y-%m-%d") if len(val_df) else None,
                "end": val_df["date"].max().strftime("%Y-%m-%d") if len(val_df) else None,
                "rows": int(len(val_df)),
            },
            "test": {
                "start": test_df["date"].min().strftime("%Y-%m-%d") if len(test_df) else None,
                "end": test_df["date"].max().strftime("%Y-%m-%d") if len(test_df) else None,
                "rows": int(len(test_df)),
            },
        },
    }

    return BacktestBuildResult(df_backtest=df_backtest, meta=meta)
