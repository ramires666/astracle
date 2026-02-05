"""
Ephemeris feature builder for the "ALL astro features" experiment.

Why this module exists (research goal, in simple words):
--------------------------------------------------------
Our Moon-only notebook answers:
    "Does Moon phase alone contain edge?"

If that answer looks close to random, the next honest question is:
    "What if we give the model EVERYTHING we can compute from ephemerides?"

Here "everything" means features that come ONLY from the ephemeris:
- positions of all bodies (longitude, speed, retrograde flag, declination, sign)
- aspects between bodies (pair interactions)
- Moon phase and planet elongations from the Sun

We do NOT use any price-based technical indicators here.
The price is used only to create labels and to plot the equity/price charts.

Practical engineering goal:
---------------------------
Ephemeris calculations are expensive. We cache the final daily feature matrix
so repeated notebook runs do not re-compute thousands of days.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from RESEARCH.astro_engine import (
    calculate_aspects_for_dates_multi,
    calculate_bodies_for_dates_multi,
    calculate_phases_for_dates,
    init_ephemeris,
)
from RESEARCH.cache_utils import load_cache, save_cache
from RESEARCH.features import build_full_features


def _date_range_key(df: pd.DataFrame) -> Dict[str, str]:
    """
    Build a small "date range signature" for stable caching.

    We include:
    - start/end date
    - number of rows

    This makes sure that if you download more market data later, the feature
    cache will not be accidentally reused for a different date range.
    """
    return {
        "start_date": pd.to_datetime(df["date"]).min().strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d"),
        "rows": int(len(df)),
    }


@dataclass(frozen=True)
class EphemerisFeatureConfig:
    """
    Configuration for building "all ephemeris features".

    coord_mode:
        - "geo"   : Earth-centered coordinates (traditional astrology view)
        - "helio" : Sun-centered coordinates (physical orbit view)
        - "both"  : combine both into one feature set (maximum info, also more noise)

    orb_mult:
        Aspect "orb" multiplier (how strict aspect detection is).
        - smaller orb -> fewer aspect hits, but more "precise"
        - bigger orb  -> more aspect hits, but may add noise

    add_trig_*:
        Tree models can sometimes handle 0..360 angles, but adding sin/cos is a
        simple way to remove the wrap-around discontinuity (359° ~ 1°).
    """

    coord_mode: str = "both"
    orb_mult: float = 0.25
    include_pair_aspects: bool = True
    include_phases: bool = True

    # Extra derived features for cyclic angles (recommended defaults).
    add_trig_for_longitudes: bool = True
    add_trig_for_moon_phase: bool = True
    add_trig_for_elongations: bool = True

    # If you want to exclude bodies (ablation), put names here, e.g. ("Pluto", "Neptune").
    # Note: build_full_features already understands geo_/helio_ prefixes.
    exclude_bodies: tuple[str, ...] = ()

    schema: str = "ephemeris_all_v1"


def _add_sin_cos(df: pd.DataFrame, angle_cols: Sequence[str], suffix: str) -> pd.DataFrame:
    """
    Add sin/cos columns for angle columns measured in degrees.

    We keep this helper separate because:
    - it is easy to read,
    - it is easy to extend later (e.g., add harmonics sin(2x), cos(2x)).
    """
    out = df
    for c in angle_cols:
        if c not in out.columns:
            continue
        rad = np.deg2rad(pd.to_numeric(out[c], errors="coerce").astype(float))
        out[f"{c}_{suffix}_sin"] = np.sin(rad)
        out[f"{c}_{suffix}_cos"] = np.cos(rad)
    return out


def build_ephemeris_feature_set(
    df_market: pd.DataFrame,
    cfg: EphemerisFeatureConfig = EphemerisFeatureConfig(),
    cache_namespace: str = "research2_ephem",
    use_cache: bool = True,
    verbose: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Build and cache a daily feature matrix from ephemerides (no price features).

    Inputs:
    - df_market must contain at least: ["date", "close"].
      We only use the date list here. Close is not used as a feature.

    Output:
    - DataFrame with column "date" + many ephemeris-derived feature columns.
    """
    if "date" not in df_market.columns:
        raise ValueError("df_market must contain a 'date' column.")

    df_market = df_market.copy()
    df_market["date"] = pd.to_datetime(df_market["date"])
    df_market = df_market.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    range_key = _date_range_key(df_market)
    cache_params = {
        "kind": "ephemeris_features",
        **range_key,
        **asdict(cfg),
    }

    if use_cache:
        cached = load_cache(str(cache_namespace), "ephemeris_features", cache_params, verbose=verbose)
        if cached is not None:
            return cached

    # 1) Load full body/aspect configuration from project settings (YAML).
    # We rely on existing RESEARCH.astro_engine wrappers so we reuse the same logic
    # as the main pipeline.
    settings = init_ephemeris()

    # 2) Compute body positions for all dates (geo / helio / both).
    # This is the most expensive part.
    df_bodies, geo_by_date, helio_by_date = calculate_bodies_for_dates_multi(
        dates=df_market["date"],
        settings=settings,
        coord_mode=str(cfg.coord_mode),
        progress=progress,
    )

    # 3) Compute aspects (optional). Aspects are "pair interactions" between bodies.
    if cfg.include_pair_aspects:
        df_aspects = calculate_aspects_for_dates_multi(
            geo_bodies_by_date=geo_by_date,
            helio_bodies_by_date=helio_by_date,
            settings=settings,
            coord_mode=str(cfg.coord_mode),
            orb_mult=float(cfg.orb_mult),
            progress=progress,
        )
    else:
        df_aspects = pd.DataFrame()

    # 4) Compute Moon phases + planet elongations (optional).
    # We compute it only from GEO bodies because:
    # - Moon phase is defined from Earth perspective (Sun/Moon longitudes).
    if cfg.include_phases:
        df_phases = calculate_phases_for_dates(bodies_by_date=geo_by_date, progress=progress)
    else:
        df_phases = pd.DataFrame()

    # 5) Build the final daily feature matrix (wide table).
    # build_full_features already produces:
    # - body features (lon/speed/retro/declination/sign)
    # - aspect features (hit/min_orb per pair/aspect)
    # - optional phase/elongation columns (merged by date)
    exclude = list(cfg.exclude_bodies) if cfg.exclude_bodies else None
    df_features = build_full_features(
        df_bodies=df_bodies,
        df_aspects=df_aspects,
        df_transits=None,
        df_phases=df_phases,
        include_pair_aspects=bool(cfg.include_pair_aspects),
        include_transit_aspects=False,
        exclude_bodies=exclude,
    )

    df_features["date"] = pd.to_datetime(df_features["date"])
    df_features = df_features.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # 6) Add sin/cos expansions for cyclic angles (optional but recommended).
    # Longitudes from build_full_features end with "_lon".
    if cfg.add_trig_for_longitudes:
        lon_cols = [c for c in df_features.columns if c.endswith("_lon")]
        df_features = _add_sin_cos(df_features, lon_cols, suffix="trig")

    if cfg.add_trig_for_moon_phase and "moon_phase_angle" in df_features.columns:
        df_features = _add_sin_cos(df_features, ["moon_phase_angle"], suffix="trig")

    if cfg.add_trig_for_elongations:
        elong_cols = [c for c in df_features.columns if c.endswith("_elongation")]
        df_features = _add_sin_cos(df_features, elong_cols, suffix="trig")

    # Safety: make sure missing numeric values do not propagate into the model.
    # (Most columns should already be filled, but trig expansions can introduce NaN
    # if an angle column is missing or non-numeric.)
    feature_cols = [c for c in df_features.columns if c != "date"]
    df_features[feature_cols] = df_features[feature_cols].fillna(0)

    if use_cache:
        save_cache(df_features, str(cache_namespace), "ephemeris_features", cache_params, verbose=verbose)

    if verbose:
        n_features = len([c for c in df_features.columns if c != "date"])
        print(f"Ephemeris feature set ready: {len(df_features)} days, {n_features} features")

    return df_features
