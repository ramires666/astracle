"""
Backtest Stats (Small, Honest, UI-Friendly)

We intentionally keep this logic in a separate file because:
- `production_dev/cache_service.py` was getting too large (>500 lines),
  and the project rules require each module to be <= 500 lines.

What this file does:
- Compute simple, explainable classification metrics from the cached backtest table.

Why this is tricky in this project:
- We have two cache "eras":

  1) NEW (research-exact) cache
     - Has a `split` column ("train" / "val" / "test")
     - Has `actual_direction` that matches the RESEARCH notebook target
     - This is the format we WANT, because it matches notebook metrics.

  2) OLD (legacy) cache
     - Only has `correct` derived from next-day price movement
     - This does NOT match notebook metrics, but we keep it as a fallback
       so older caches don't crash the UI.

UI convention we follow:
- The "headline" stats returned by the API should be the TEST split
  whenever it exists (honest, out-of-sample).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


def _direction_series_to_code(s: pd.Series) -> pd.Series:
    """Map text directions to 0/1 codes, keep NaN for unknowns."""
    return s.map({"DOWN": 0, "UP": 1})


def _compute_binary_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute a small set of easy-to-explain metrics.

    We intentionally include:
    - accuracy: overall correctness
    - recall_down / recall_up: "how many of each class we catch"
    - r_min: min(recall_down, recall_up) (this is the notebook's R_MIN)
    - mcc: Matthew's correlation coefficient (robust for imbalance)
    """
    total = int(len(y_true))
    if total == 0:
        return {
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "recall_down": 0.0,
            "recall_up": 0.0,
            "r_min": 0.0,
            "mcc": 0.0,
        }

    correct = int((y_true == y_pred).sum())
    accuracy = float(correct / total)

    # Recalls (avoid division by zero)
    down_mask = y_true == 0
    up_mask = y_true == 1

    n_down = int(down_mask.sum())
    n_up = int(up_mask.sum())

    recall_down = float(((y_pred == 0) & down_mask).sum() / n_down) if n_down > 0 else 0.0
    recall_up = float(((y_pred == 1) & up_mask).sum() / n_up) if n_up > 0 else 0.0

    r_min = float(min(recall_down, recall_up))
    mcc = float(matthews_corrcoef(y_true, y_pred)) if total > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "recall_down": round(recall_down, 4),
        "recall_up": round(recall_up, 4),
        "r_min": round(r_min, 6),
        "mcc": round(mcc, 6),
    }


def compute_backtest_stats(df: pd.DataFrame) -> Dict:
    """
    Compute stats dict from a backtest dataframe.

    The returned dict is designed to be JSON-serializable and UI-friendly.
    """
    if df is None or df.empty:
        return {"accuracy": 0.0, "total": 0, "correct": 0}

    # ------------------------------------------------------------------
    # Preferred path: research-exact cache with split + actual_direction
    # ------------------------------------------------------------------
    if "split" in df.columns and "actual_direction" in df.columns:
        y_true_all = _direction_series_to_code(df["actual_direction"])
        y_pred_all = _direction_series_to_code(df["direction"])
        valid = y_true_all.notna() & y_pred_all.notna()

        df_valid = df.loc[valid].copy()
        if df_valid.empty:
            return {"accuracy": 0.0, "total": 0, "correct": 0}

        # Per-split stats + date ranges (for chart labels)
        split_stats: Dict[str, Dict] = {}
        split_ranges: Dict[str, Dict] = {}
        for split_name in ["train", "val", "test"]:
            part = df_valid[df_valid["split"] == split_name].copy()
            if part.empty:
                continue

            y_true = _direction_series_to_code(part["actual_direction"]).astype(int).to_numpy()
            y_pred = _direction_series_to_code(part["direction"]).astype(int).to_numpy()

            split_stats[split_name] = _compute_binary_stats(y_true, y_pred)
            split_ranges[split_name] = {
                "start": part["date"].min().strftime("%Y-%m-%d"),
                "end": part["date"].max().strftime("%Y-%m-%d"),
                "rows": int(len(part)),
            }

        # Headline stats = TEST split when available (honest, out-of-sample).
        if "test" in split_stats:
            headline = dict(split_stats["test"])
            headline["scope"] = "test"
        else:
            y_true_all = y_true_all.loc[valid].astype(int).to_numpy()
            y_pred_all = y_pred_all.loc[valid].astype(int).to_numpy()
            headline = _compute_binary_stats(y_true_all, y_pred_all)
            headline["scope"] = "all"

        headline["splits"] = split_stats
        headline["split_ranges"] = split_ranges

        # Expose tuned threshold if the cache stored it.
        if "decision_threshold" in df.columns and df["decision_threshold"].notna().any():
            headline["decision_threshold"] = float(df["decision_threshold"].dropna().iloc[0])

        return headline

    # ------------------------------------------------------------------
    # Legacy fallback: just compute accuracy from `correct` column.
    # ------------------------------------------------------------------
    if "correct" in df.columns:
        valid_rows = df[df["correct"].notna()].copy()
        total = int(len(valid_rows))
        correct = int(valid_rows["correct"].sum()) if total > 0 else 0
        accuracy = float(correct / total) if total > 0 else 0.0
        return {"total": total, "correct": correct, "accuracy": round(accuracy, 4)}

    return {"accuracy": 0.0, "total": 0, "correct": 0}

