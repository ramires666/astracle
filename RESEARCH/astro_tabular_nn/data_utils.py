"""Dataset utilities for astro tabular neural experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .config import DatasetConfig, SplitConfig


@dataclass(frozen=True)
class LoadedDataset:
    """In-memory dataset used by training and notebook analysis."""

    dataframe: pd.DataFrame
    feature_cols: Tuple[str, ...]
    X_raw: np.ndarray
    y: np.ndarray
    dates: np.ndarray


@dataclass(frozen=True)
class SplitIndices:
    """Chronological train/val/test indices."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class PreparedSplit:
    """Scaled arrays and class weights for a single split."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    class_weights: np.ndarray
    scaler: RobustScaler


def _validate_dataset_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Provide an existing parquet in DatasetConfig.dataset_path."
        )


def _feature_columns(df: pd.DataFrame, cfg: DatasetConfig) -> List[str]:
    drop_set = {cfg.date_col, cfg.target_col, *cfg.drop_cols}
    return [c for c in df.columns if c not in drop_set]


def load_tabular_dataset(cfg: DatasetConfig) -> LoadedDataset:
    """Load parquet dataset and build a numeric feature matrix."""
    path = Path(cfg.dataset_path)
    _validate_dataset_path(path)

    df = pd.read_parquet(path)
    if cfg.date_col not in df.columns:
        raise KeyError(f"Missing date column '{cfg.date_col}' in {path}")
    if cfg.target_col not in df.columns:
        raise KeyError(f"Missing target column '{cfg.target_col}' in {path}")

    df = df.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    if df[cfg.date_col].isna().any():
        bad = int(df[cfg.date_col].isna().sum())
        raise ValueError(f"Date parsing failed for {bad} rows in {path}")

    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    feature_cols = _feature_columns(df, cfg)
    if not feature_cols:
        raise ValueError("No feature columns found after applying drop rules")

    numeric_view = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    numeric_view = numeric_view.fillna(float(cfg.fillna_value))

    X_raw = numeric_view.to_numpy(dtype=np.float32, copy=True)
    y = df[cfg.target_col].to_numpy(dtype=np.int64, copy=True)
    dates = df[cfg.date_col].to_numpy(copy=True)

    return LoadedDataset(
        dataframe=df,
        feature_cols=tuple(feature_cols),
        X_raw=X_raw,
        y=y,
        dates=dates,
    )


def build_time_split(n_rows: int, cfg: SplitConfig) -> SplitIndices:
    """Build a chronological split with no shuffling."""
    if n_rows < 20:
        raise ValueError(f"Dataset too small for reliable split: n_rows={n_rows}")

    train_end = int(n_rows * cfg.train_ratio)
    val_end = int(n_rows * (cfg.train_ratio + cfg.val_ratio))

    train_end = max(10, min(train_end, n_rows - 4))
    val_end = max(train_end + 2, min(val_end, n_rows - 2))

    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, val_end, dtype=np.int64)
    test_idx = np.arange(val_end, n_rows, dtype=np.int64)

    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Split produced empty validation or test segment. "
            f"n_rows={n_rows}, train_end={train_end}, val_end={val_end}"
        )

    return SplitIndices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def compute_class_weights(y_train: np.ndarray, n_classes: int, power: float) -> np.ndarray:
    """Compute balanced class weights and optionally amplify minority classes."""
    if y_train.ndim != 1:
        raise ValueError("y_train must be 1D")

    class_ids = np.arange(n_classes, dtype=np.int64)
    counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    base = float(len(y_train)) / (float(n_classes) * counts)
    weights = np.power(base, float(power))
    weights = weights / weights.mean()

    return weights.astype(np.float32)


def prepare_split(
    dataset: LoadedDataset,
    split: SplitIndices,
    class_weight_power: float,
    n_classes: int,
) -> PreparedSplit:
    """Fit scaler on train and prepare scaled arrays for all split parts."""
    X_train_raw = dataset.X_raw[split.train_idx]
    X_val_raw = dataset.X_raw[split.val_idx]
    X_test_raw = dataset.X_raw[split.test_idx]

    y_train = dataset.y[split.train_idx]
    y_val = dataset.y[split.val_idx]
    y_test = dataset.y[split.test_idx]

    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0))
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32, copy=False)
    X_val = scaler.transform(X_val_raw).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test_raw).astype(np.float32, copy=False)

    class_weights = compute_class_weights(
        y_train=y_train,
        n_classes=n_classes,
        power=class_weight_power,
    )

    return PreparedSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        class_weights=class_weights,
        scaler=scaler,
    )


def split_summary(dataset: LoadedDataset, split: SplitIndices) -> Dict[str, object]:
    """Create a compact split summary dict for logging and notebook display."""
    y = dataset.y
    train_y = y[split.train_idx]
    val_y = y[split.val_idx]
    test_y = y[split.test_idx]

    def _shares(arr: np.ndarray) -> Dict[int, float]:
        n = max(len(arr), 1)
        uniq, cnt = np.unique(arr, return_counts=True)
        return {int(k): float(v) / float(n) for k, v in zip(uniq.tolist(), cnt.tolist())}

    return {
        "rows_total": int(len(y)),
        "features": int(dataset.X_raw.shape[1]),
        "train_rows": int(len(split.train_idx)),
        "val_rows": int(len(split.val_idx)),
        "test_rows": int(len(split.test_idx)),
        "target_share_train": _shares(train_y),
        "target_share_val": _shares(val_y),
        "target_share_test": _shares(test_y),
        "date_start": str(pd.to_datetime(dataset.dates.min()).date()),
        "date_end": str(pd.to_datetime(dataset.dates.max()).date()),
        "mean_zero_fraction": float((dataset.X_raw == 0.0).mean()),
    }
