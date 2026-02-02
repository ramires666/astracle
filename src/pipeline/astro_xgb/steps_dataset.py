"""
Dataset merge step: features + labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.pipeline.artifacts import build_meta, is_cache_valid, meta_path_for, save_meta
from src.pipeline.astro_xgb.context import PipelineContext


def _extract_tag(path: Path, prefix: str) -> str:
    stem = Path(path).stem
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _feature_inventory(df_dataset: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    stats = df_dataset[feature_cols].describe().T[["mean", "std", "min", "max"]]
    missing_pct = df_dataset[feature_cols].isna().mean() * 100
    stats = stats.join(missing_pct.rename("missing_pct"), how="left")
    return stats.reset_index().rename(columns={"index": "feature"})


def run_dataset_step(
    ctx: PipelineContext,
    features_path: Path,
    labels_path: Path,
    force: bool = False,
    write_inventory: bool = False,
) -> Path:
    """
    Merge features and labels on date.
    """
    features_tag = _extract_tag(features_path, f"{ctx.subject.subject_id}_features_")
    labels_tag = _extract_tag(labels_path, f"{ctx.subject.subject_id}_labels_")
    out_path = ctx.processed_dir / f"{ctx.subject.subject_id}_dataset_{labels_tag}__{features_tag}.parquet"

    params = {"features": str(features_path.name), "labels": str(labels_path.name)}
    inputs = [Path(features_path), Path(labels_path)]

    if not force and is_cache_valid(out_path, params=params, inputs=inputs, step="dataset"):
        print("[DATASET] Using cached dataset:", out_path)
        return out_path

    df_features = pd.read_parquet(features_path)
    df_labels = pd.read_parquet(labels_path)

    df_features["date"] = pd.to_datetime(df_features["date"])
    df_labels["date"] = pd.to_datetime(df_labels["date"])

    df_dataset = pd.merge(df_labels[["date", "target"]], df_features, on="date", how="inner")
    if df_dataset["date"].duplicated().any():
        df_dataset = df_dataset.drop_duplicates(subset=["date"]).reset_index(drop=True)
    df_dataset = df_dataset.sort_values("date").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_dataset.to_parquet(out_path, index=False)

    meta = build_meta(step="dataset", params=params, inputs=inputs)
    save_meta(meta_path_for(out_path), meta)

    print("[DATASET] Built dataset:", out_path)
    print(f"[DATASET] Shape: {df_dataset.shape}")
    if not df_dataset.empty:
        print(f"[DATASET] Date range: {df_dataset['date'].min().date()} -> {df_dataset['date'].max().date()}")
        dist = df_dataset["target"].value_counts(normalize=True).sort_index() * 100
        print("[DATASET] Class share (%):", {int(k): round(v, 2) for k, v in dist.items()})

    if write_inventory:
        feature_cols = [c for c in df_dataset.columns if c not in ["date", "target"]]
        inv = _feature_inventory(df_dataset, feature_cols)
        inv_path = ctx.reports_dir / f"{ctx.subject.subject_id}_feature_inventory_{labels_tag}__{features_tag}.parquet"
        inv_path.parent.mkdir(parents=True, exist_ok=True)
        inv.to_parquet(inv_path, index=False)
        print("[DATASET] Feature inventory:", inv_path)

    return out_path
