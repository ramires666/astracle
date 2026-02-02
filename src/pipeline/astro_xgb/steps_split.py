"""
Time-based train/val/test split for dataset parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.pipeline.artifacts import build_meta, is_cache_valid, meta_path_for, save_meta
from src.pipeline.astro_xgb.context import PipelineContext


def _split_paths(ctx: PipelineContext, dataset_path: Path) -> Tuple[Path, Path, Path]:
    stem = Path(dataset_path).stem
    base = stem.replace("dataset", "split")
    train_path = ctx.processed_dir / f"{base}_train.parquet"
    val_path = ctx.processed_dir / f"{base}_val.parquet"
    test_path = ctx.processed_dir / f"{base}_test.parquet"
    return train_path, val_path, test_path


def run_split_step(
    ctx: PipelineContext,
    dataset_path: Path,
    force: bool = False,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[Path, Path, Path]:
    """
    Split dataset into train/val/test by time order.
    """
    train_path, val_path, test_path = _split_paths(ctx, dataset_path)
    params = {"train_ratio": float(train_ratio), "val_ratio": float(val_ratio)}
    inputs = [Path(dataset_path)]

    # Cache check uses train_path as primary artifact.
    if not force and is_cache_valid(train_path, params=params, inputs=inputs, step="split"):
        print("[SPLIT] Using cached splits:")
        print("  train:", train_path)
        print("  val  :", val_path)
        print("  test :", test_path)
        return train_path, val_path, test_path

    df_dataset = pd.read_parquet(dataset_path).copy()
    df_dataset["date"] = pd.to_datetime(df_dataset["date"])
    df_dataset = df_dataset.sort_values("date").reset_index(drop=True)

    n = len(df_dataset)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_dataset.iloc[:train_end].copy()
    val_df = df_dataset.iloc[train_end:val_end].copy()
    test_df = df_dataset.iloc[val_end:].copy()

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    meta = build_meta(step="split", params=params, inputs=inputs)
    save_meta(meta_path_for(train_path), meta)
    print("[SPLIT] Saved splits:")
    print("  train:", train_path, f"rows={len(train_df)}",
          f"range={train_df['date'].min().date()}->{train_df['date'].max().date()}")
    print("  val  :", val_path, f"rows={len(val_df)}",
          f"range={val_df['date'].min().date()}->{val_df['date'].max().date()}")
    print("  test :", test_path, f"rows={len(test_df)}",
          f"range={test_df['date'].min().date()}->{test_df['date'].max().date()}")
    return train_path, val_path, test_path
