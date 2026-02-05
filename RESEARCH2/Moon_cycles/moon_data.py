"""
Data preparation helpers for Moon-phase-only experiments.

This module intentionally keeps one job per function so the notebook stays simple:
1. Load market prices from local parquet (via RESEARCH.data_loader).
2. Build Moon-phase features only (Sun + Moon positions).
3. Build balanced labels with configurable Gaussian parameters.
4. Merge features + labels into a daily dataset for model training.

All expensive steps are cached via RESEARCH.cache_utils.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from RESEARCH.astro_engine import (
    calculate_bodies_for_dates,
    calculate_phases_for_dates,
    init_ephemeris,
)
from RESEARCH.cache_utils import load_cache, save_cache
from RESEARCH.data_loader import load_market_data
from RESEARCH.features import merge_features_with_labels
from RESEARCH.labeling import create_balanced_labels


@dataclass(frozen=True)
class MoonLabelConfig:
    """
    Label parameters for balanced binary targets.

    We keep this explicit so the notebook can sweep `gauss_window` and `gauss_std`
    while keeping the rest stable.
    """

    horizon: int = 1
    move_share: float = 0.5
    label_mode: str = "balanced_detrended"
    price_mode: str = "raw"


def _date_range_key(df: pd.DataFrame) -> Dict[str, str]:
    """Build a small cache key fragment from market date range."""
    return {
        "start_date": pd.to_datetime(df["date"]).min().strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d"),
        "rows": int(len(df)),
    }


def load_market_slice(
    start_date: str = "2017-11-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load market data from parquet and optionally cache the slice.

    Notes:
    - RESEARCH.data_loader.load_market_data is already parquet-only in this repo.
    - We still cache the exact slice to speed up repeated notebook reruns.
    """
    cache_params = {
        "kind": "market_slice",
        "start_date": start_date,
        "end_date": end_date or "latest",
    }
    if use_cache:
        cached = load_cache("research2_moon", "market", cache_params, verbose=verbose)
        if cached is not None:
            return cached

    df_market = load_market_data(start_date=start_date, end_date=end_date)
    df_market = df_market[["date", "close"]].copy()
    df_market["date"] = pd.to_datetime(df_market["date"])
    df_market = df_market.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if use_cache:
        save_cache(df_market, "research2_moon", "market", cache_params, verbose=verbose)

    return df_market


def _keep_only_sun_and_moon(settings) -> None:
    """
    Mutate AstroSettings in-place to keep only Sun and Moon bodies.

    This is a major speed optimization:
    - full body list computes many planets/nodes,
    - moon-cycle research only needs Sun + Moon for phase angle.
    """
    wanted = {"Sun", "Moon"}
    filtered = [b for b in settings.bodies if getattr(b, "name", "") in wanted]
    if len(filtered) != 2:
        raise ValueError(
            "Failed to build Sun+Moon settings. Check bodies config for Sun/Moon presence."
        )
    settings.bodies = filtered


def build_moon_phase_features(
    df_market: pd.DataFrame,
    use_cache: bool = True,
    verbose: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Build Moon-phase-only features for each market day.

    Output columns:
    - moon_phase_angle
    - moon_phase_ratio
    - moon_illumination
    - lunar_day
    - moon_phase_sin / moon_phase_cos (cyclic encoding)
    - simple regime flags (waxing/waning, new/full zones)
    """
    range_key = _date_range_key(df_market)
    cache_params = {
        "kind": "moon_features",
        **range_key,
        "schema": "v1",
    }

    if use_cache:
        cached = load_cache("research2_moon", "moon_features", cache_params, verbose=verbose)
        if cached is not None:
            return cached

    settings = init_ephemeris()
    _keep_only_sun_and_moon(settings)

    # Calculate only Sun/Moon positions, then derive phase features.
    _, bodies_by_date = calculate_bodies_for_dates(
        dates=df_market["date"],
        settings=settings,
        center="geo",
        progress=progress,
    )

    df_phases = calculate_phases_for_dates(bodies_by_date, progress=progress)
    keep_cols = [
        "date",
        "moon_phase_angle",
        "moon_phase_ratio",
        "moon_illumination",
        "lunar_day",
    ]
    df_moon = df_phases[keep_cols].copy()
    df_moon["date"] = pd.to_datetime(df_moon["date"])

    # Cyclic encoding (important for angles near 0/360 wrap-around).
    angle_rad = np.deg2rad(df_moon["moon_phase_angle"].astype(float))
    df_moon["moon_phase_sin"] = np.sin(angle_rad)
    df_moon["moon_phase_cos"] = np.cos(angle_rad)

    # Extra simple phase regime flags (easy for tree model to use).
    angle = df_moon["moon_phase_angle"].astype(float)
    df_moon["is_waxing"] = ((angle > 0.0) & (angle < 180.0)).astype(int)
    df_moon["is_waning"] = ((angle > 180.0) & (angle < 360.0)).astype(int)
    df_moon["is_new_moon_zone"] = ((angle < 22.5) | (angle >= 337.5)).astype(int)
    df_moon["is_full_moon_zone"] = ((angle >= 157.5) & (angle < 202.5)).astype(int)

    df_moon = df_moon.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if use_cache:
        save_cache(df_moon, "research2_moon", "moon_features", cache_params, verbose=verbose)

    return df_moon


def build_balanced_labels_for_gauss(
    df_market: pd.DataFrame,
    gauss_window: int,
    gauss_std: float,
    label_cfg: MoonLabelConfig,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build balanced labels for one Gaussian configuration.

    We pass parameters by explicit keyword names to avoid accidental argument mixups.
    """
    range_key = _date_range_key(df_market)
    cache_params = {
        "kind": "labels",
        **range_key,
        "horizon": int(label_cfg.horizon),
        "move_share": float(label_cfg.move_share),
        "gauss_window": int(gauss_window),
        "gauss_std": float(gauss_std),
        "label_mode": str(label_cfg.label_mode),
        "price_mode": str(label_cfg.price_mode),
    }

    if use_cache:
        cached = load_cache("research2_moon", "labels", cache_params, verbose=verbose)
        if cached is not None:
            return cached

    df_labels = create_balanced_labels(
        df_market=df_market,
        horizon=label_cfg.horizon,
        move_share=label_cfg.move_share,
        gauss_window=gauss_window,
        gauss_std=gauss_std,
        price_mode=label_cfg.price_mode,
        label_mode=label_cfg.label_mode,
        verbose=verbose,
    )

    if use_cache:
        save_cache(df_labels, "research2_moon", "labels", cache_params, verbose=verbose)

    return df_labels


def build_moon_dataset_for_gauss(
    df_market: pd.DataFrame,
    df_moon_features: pd.DataFrame,
    gauss_window: int,
    gauss_std: float,
    label_cfg: MoonLabelConfig,
    use_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Merge moon features with one label configuration into a full daily dataset.
    """
    range_key = _date_range_key(df_market)
    cache_params = {
        "kind": "dataset",
        **range_key,
        "gauss_window": int(gauss_window),
        "gauss_std": float(gauss_std),
        "horizon": int(label_cfg.horizon),
        "move_share": float(label_cfg.move_share),
        "schema": "moon_only_v2",
    }

    if use_cache:
        cached = load_cache("research2_moon", "dataset", cache_params, verbose=verbose)
        if cached is not None:
            return cached

    df_labels = build_balanced_labels_for_gauss(
        df_market=df_market,
        gauss_window=gauss_window,
        gauss_std=gauss_std,
        label_cfg=label_cfg,
        use_cache=use_cache,
        verbose=verbose,
    )

    df_dataset = merge_features_with_labels(
        df_features=df_moon_features,
        df_labels=df_labels,
        verbose=verbose,
    )

    # Keep raw close price for visual diagnostics in notebook.
    # This column is NOT used as a model feature (see get_moon_feature_columns).
    df_close = df_market[["date", "close"]].copy()
    df_close["date"] = pd.to_datetime(df_close["date"])
    df_dataset = pd.merge(df_dataset, df_close, on="date", how="left")

    if use_cache:
        save_cache(df_dataset, "research2_moon", "dataset", cache_params, verbose=verbose)

    return df_dataset


def get_moon_feature_columns(df_dataset: pd.DataFrame) -> List[str]:
    """Return feature columns for moon-only dataset."""
    return [c for c in df_dataset.columns if c not in {"date", "target", "close"}]
