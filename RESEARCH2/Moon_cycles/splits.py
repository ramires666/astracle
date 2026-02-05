"""
Split helpers for Moon-cycle research.

We keep split logic in one small module so the notebook can stay readable.
The notebook only asks for two protocols:

1) Classic split: 70% train, 15% validation, 15% test.
2) Walk-forward split: warm-up train chunk + multiple future evaluation blocks.

All splits are strictly time-ordered (no shuffling) to avoid future leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitDefinition:
    """
    One time split definition.

    We keep index arrays instead of sliced DataFrames because this is easier to reuse
    in different places (training, plotting, diagnostics, and caching).
    """

    protocol: str
    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _ensure_dataset_is_valid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize input dataset before splitting.

    Why this check exists:
    - Time split must use chronological order.
    - Duplicate dates can break boundary logic.
    - Missing `date` column means we cannot build time blocks safely.
    """
    if "date" not in df.columns:
        raise ValueError("Dataset must contain a 'date' column for time-based splitting.")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    if out["date"].duplicated().any():
        raise ValueError("Dataset has duplicate dates. Remove duplicates before splitting.")

    if len(out) < 20:
        raise ValueError(
            "Dataset is too short for train/validation/test research. Need at least 20 rows."
        )

    return out


def _to_idx(start: int, end: int) -> np.ndarray:
    """Create a safe integer index range [start, end)."""
    if end <= start:
        return np.array([], dtype=np.int64)
    return np.arange(start, end, dtype=np.int64)


def make_classic_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SplitDefinition:
    """
    Build one classic split (70/15/15 by default).

    Notes:
    - Ratios are applied to row count.
    - Test receives the remaining tail, so total is always exactly 100%.
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1.")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 so test split is non-empty.")

    df_checked = _ensure_dataset_is_valid(df)
    n = len(df_checked)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    split = SplitDefinition(
        protocol="classic_70_15_15",
        fold_id=1,
        train_idx=_to_idx(0, train_end),
        val_idx=_to_idx(train_end, val_end),
        test_idx=_to_idx(val_end, n),
    )

    _validate_non_empty_chunks(split)
    return split


def make_walk_forward_splits(
    df: pd.DataFrame,
    warmup_ratio: float = 0.50,
    block_ratios: Sequence[float] = (0.10, 0.10, 0.10, 0.10, 0.10),
    val_fraction_inside_block: float = 0.50,
) -> List[SplitDefinition]:
    """
    Build expanding-window walk-forward splits.

    Default protocol:
    - First 50% of history is warm-up training chunk.
    - Remaining history is divided into five 10% blocks.
    - Inside every block: first half = validation, second half = test.

    This directly matches the requested "50 + 10/10/10/10/10" style,
    with each 10% split into 5% validation + 5% test.

    Important detail:
    - Training window is expanding. For fold k, train uses everything before
      that fold's validation period.
    """
    if not (0.0 < warmup_ratio < 1.0):
        raise ValueError("warmup_ratio must be between 0 and 1.")
    if len(block_ratios) == 0:
        raise ValueError("block_ratios cannot be empty.")
    if any(r <= 0.0 for r in block_ratios):
        raise ValueError("Each block ratio must be positive.")
    if not (0.0 < val_fraction_inside_block < 1.0):
        raise ValueError("val_fraction_inside_block must be between 0 and 1.")

    df_checked = _ensure_dataset_is_valid(df)
    n = len(df_checked)

    warmup_end = int(n * warmup_ratio)
    if warmup_end < 10:
        raise ValueError("Warm-up train period is too short. Increase dataset size or warmup_ratio.")

    # Convert requested ratios into integer block boundaries.
    # We normalize block ratios to fit the entire tail after warm-up.
    tail_len = n - warmup_end
    if tail_len < 10:
        raise ValueError("Tail after warm-up is too short for walk-forward blocks.")

    block_ratios = np.asarray(block_ratios, dtype=float)
    block_ratios = block_ratios / block_ratios.sum()

    raw_sizes = (block_ratios * tail_len).astype(int)

    # Ensure every block has at least 2 rows so val/test can both exist.
    raw_sizes = np.maximum(raw_sizes, 2)

    # Adjust sizes so total equals exact tail length.
    size_diff = int(raw_sizes.sum() - tail_len)
    if size_diff > 0:
        for i in range(len(raw_sizes) - 1, -1, -1):
            take = min(size_diff, max(0, raw_sizes[i] - 2))
            raw_sizes[i] -= take
            size_diff -= take
            if size_diff == 0:
                break
    elif size_diff < 0:
        raw_sizes[-1] += abs(size_diff)

    splits: List[SplitDefinition] = []
    cursor = warmup_end

    for fold_idx, block_size in enumerate(raw_sizes, start=1):
        block_start = cursor
        block_end = min(n, block_start + int(block_size))

        # In each block we split into validation then test.
        val_size = max(1, int((block_end - block_start) * val_fraction_inside_block))
        val_end = min(block_end - 1, block_start + val_size)

        split = SplitDefinition(
            protocol="walk_forward_50_plus_blocks",
            fold_id=fold_idx,
            train_idx=_to_idx(0, block_start),
            val_idx=_to_idx(block_start, val_end),
            test_idx=_to_idx(val_end, block_end),
        )

        _validate_non_empty_chunks(split)
        splits.append(split)
        cursor = block_end

    return splits


def _validate_non_empty_chunks(split: SplitDefinition) -> None:
    """
    Validate split internals.

    We fail early with a clear message, because empty train/val/test segments
    would later crash model training in less obvious places.
    """
    if len(split.train_idx) == 0:
        raise ValueError(f"{split.protocol} fold {split.fold_id}: train split is empty.")
    if len(split.val_idx) == 0:
        raise ValueError(f"{split.protocol} fold {split.fold_id}: validation split is empty.")
    if len(split.test_idx) == 0:
        raise ValueError(f"{split.protocol} fold {split.fold_id}: test split is empty.")


def describe_split(df: pd.DataFrame, split: SplitDefinition) -> Dict[str, object]:
    """
    Build a compact split summary dictionary for logs/tables.

    This helper keeps the notebook tidy and gives an easy way to verify
    the exact date boundaries used by each protocol.
    """
    df_checked = _ensure_dataset_is_valid(df)

    def _bounds(idx: np.ndarray) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = df_checked.iloc[int(idx[0])]["date"]
        end = df_checked.iloc[int(idx[-1])]["date"]
        return start, end

    train_start, train_end = _bounds(split.train_idx)
    val_start, val_end = _bounds(split.val_idx)
    test_start, test_end = _bounds(split.test_idx)

    return {
        "protocol": split.protocol,
        "fold": int(split.fold_id),
        "train_rows": int(len(split.train_idx)),
        "val_rows": int(len(split.val_idx)),
        "test_rows": int(len(split.test_idx)),
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
    }


def describe_splits_table(df: pd.DataFrame, splits: Iterable[SplitDefinition]) -> pd.DataFrame:
    """Return one DataFrame summary for a list of splits."""
    rows = [describe_split(df, s) for s in splits]
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    date_cols = [
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    ]
    for col in date_cols:
        out[col] = pd.to_datetime(out[col]).dt.strftime("%Y-%m-%d")

    return out.sort_values(["protocol", "fold"]).reset_index(drop=True)
