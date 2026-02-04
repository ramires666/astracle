"""
Feature build step from astro bodies/aspects/transits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.features.builder import build_features_daily
from src.pipeline.artifacts import build_meta, is_cache_valid, meta_path_for, save_meta
from src.pipeline.astro_xgb.context import PipelineContext
from src.pipeline.astro_xgb.naming import orb_tag


def run_features_step(
    ctx: PipelineContext,
    astro_paths: Dict[str, Tuple[Path, Path, Optional[Path]]],
    force: bool = False,
    orb_multiplier: float = 1.0,
    include_pair_aspects: bool = True,
    include_transit_aspects: bool = False,
    include_both_centers: bool = False,
) -> Path:
    """
    Build astro feature matrix and return parquet path.
    """
    tag = orb_tag(float(orb_multiplier))
    centers = sorted(astro_paths.keys())
    centers_tag = "both" if include_both_centers and len(centers) > 1 else centers[0]
    out_path = ctx.processed_dir / f"{ctx.subject.subject_id}_features_{tag}_{centers_tag}.parquet"

    params = {
        "orb_multiplier": float(orb_multiplier),
        "include_pair_aspects": bool(include_pair_aspects),
        "include_transit_aspects": bool(include_transit_aspects),
        "include_both_centers": bool(include_both_centers),
        "centers": centers,
    }

    inputs: list[Path] = []
    for center in centers:
        bp, ap, tp = astro_paths[center]
        inputs.append(Path(bp))
        inputs.append(Path(ap))
        if include_transit_aspects and tp is not None:
            inputs.append(Path(tp))

    if not force and is_cache_valid(out_path, params=params, inputs=inputs, step="features"):
        print("[FEATURES] Using cached features:", out_path)
        return out_path

    def _build_for_center(prefix: str, paths: Tuple[Path, Path, Optional[Path]]) -> pd.DataFrame:
        bodies_path, aspects_path, transits_path = paths
        df_bodies = pd.read_parquet(bodies_path)
        df_aspects = pd.read_parquet(aspects_path)
        df_transits = None
        if include_transit_aspects and transits_path is not None:
            df_transits = pd.read_parquet(transits_path)

        df_feat = build_features_daily(
            df_bodies,
            df_aspects,
            df_transits,
            include_pair_aspects=include_pair_aspects,
            include_transit_aspects=include_transit_aspects,
        )
        # Prefix all feature columns with center to avoid collisions.
        rename = {c: f"{prefix}__{c}" for c in df_feat.columns if c != "date"}
        df_feat = df_feat.rename(columns=rename)
        return df_feat

    if include_both_centers and len(centers) > 1:
        merged: Optional[pd.DataFrame] = None
        for center in centers:
            df_feat = _build_for_center(center, astro_paths[center])
            if merged is None:
                merged = df_feat
            else:
                merged = pd.merge(merged, df_feat, on="date", how="outer")
        df_features = merged.fillna(0) if merged is not None else pd.DataFrame(columns=["date"])
    else:
        center = centers[0]
        df_features = _build_for_center(center, astro_paths[center])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(out_path, index=False)
    meta = build_meta(step="features", params=params, inputs=inputs)
    save_meta(meta_path_for(out_path), meta)
    print("[FEATURES] Built features:", out_path)
    print(f"[FEATURES] Shape: {df_features.shape}")
    return out_path
