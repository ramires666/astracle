"""
Astro feature builders for turning-point experiments.

Goal:
- Reuse existing ephemeris feature pipeline.
- Add transit-to-natal features that depend on birth datetime.
- Provide compact feature-group diagnostics for notebook logging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import pandas as pd

from RESEARCH.astro_engine import (
    calculate_bodies_for_dates,
    calculate_transits_for_dates,
    get_natal_bodies,
    init_ephemeris,
)
from RESEARCH.cache_utils import load_cache, save_cache
from src.features.builder import build_transit_aspect_features

from .ephemeris_data import EphemerisFeatureConfig, build_ephemeris_feature_set


@dataclass(frozen=True)
class TurningAstroFeatureConfig:
    """
    Config for full astro feature matrix used in turning-point modeling.

    We intentionally keep this close to EphemerisFeatureConfig so it remains
    easy to compare with previous experiments.
    """

    coord_mode: str = "both"
    orb_mult: float = 0.10
    include_pair_aspects: bool = True
    include_phases: bool = True
    include_transit_aspects: bool = True

    add_trig_for_longitudes: bool = True
    add_trig_for_moon_phase: bool = True
    add_trig_for_elongations: bool = True

    schema: str = "turning_astro_full_v1"


def _date_range_key(df: pd.DataFrame) -> dict[str, object]:
    """Build stable cache key fragment from date range."""
    d = pd.to_datetime(df["date"])
    return {
        "start_date": d.min().strftime("%Y-%m-%d"),
        "end_date": d.max().strftime("%Y-%m-%d"),
        "rows": int(len(df)),
    }


def _normalize_market_dates(df_market: pd.DataFrame) -> pd.DataFrame:
    """Return sorted unique date-only market frame."""
    if "date" not in df_market.columns:
        raise ValueError("df_market must contain 'date'")

    out = df_market[["date"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return out


def build_transit_to_natal_feature_set(
    df_market: pd.DataFrame,
    birth_dt_utc: str,
    orb_mult: float,
    cache_namespace: str = "research2_turning_transits",
    use_cache: bool = True,
    verbose: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Build transit->natal aspect feature matrix for each market day.

    Output:
    - 'date' + transit aspect hit/orb columns.
    """
    base = _normalize_market_dates(df_market)

    cache_params = {
        "kind": "transit_to_natal_features",
        **_date_range_key(base),
        "birth_dt_utc": str(birth_dt_utc),
        "orb_mult": float(orb_mult),
        "schema": "turning_transit_v1",
    }

    if use_cache:
        cached = load_cache(cache_namespace, "transit_to_natal_features", cache_params, verbose=verbose)
        if cached is not None:
            return cached

    settings = init_ephemeris()

    # Transit positions are computed in geocentric mode for natal interactions.
    _, bodies_by_date = calculate_bodies_for_dates(
        dates=base["date"],
        settings=settings,
        center="geo",
        progress=progress,
    )

    natal_bodies = get_natal_bodies(
        birth_dt_str=str(birth_dt_utc),
        settings=settings,
        center="geo",
    )

    df_transits = calculate_transits_for_dates(
        bodies_by_date=bodies_by_date,
        natal_bodies=natal_bodies,
        settings=settings,
        orb_mult=float(orb_mult),
        progress=progress,
    )

    if df_transits.empty:
        out = base.copy()
    else:
        df_transits = df_transits.copy()
        df_transits["date"] = pd.to_datetime(df_transits["date"])
        feat = build_transit_aspect_features(df_transits)
        feat["date"] = pd.to_datetime(feat["date"])
        out = pd.merge(base, feat, on="date", how="left")

    feature_cols = [c for c in out.columns if c != "date"]
    if feature_cols:
        out[feature_cols] = out[feature_cols].fillna(0)

    if use_cache:
        save_cache(out, cache_namespace, "transit_to_natal_features", cache_params, verbose=verbose)

    if verbose:
        print(
            "Transit->natal features ready:",
            f"rows={len(out)}",
            f"cols={len(feature_cols)}",
            f"birth_dt_utc={birth_dt_utc}",
        )

    return out


def build_turning_astro_feature_set(
    df_market: pd.DataFrame,
    birth_dt_utc: str,
    cfg: TurningAstroFeatureConfig = TurningAstroFeatureConfig(),
    cache_namespace: str = "research2_turning_astro",
    use_cache: bool = True,
    verbose: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Build full astro feature matrix for turning-point experiments.

    Includes:
    - all ephemeris features (bodies, pair aspects, phases)
    - optional transit->natal features from birth datetime
    """
    base = _normalize_market_dates(df_market)

    ephem_cfg = EphemerisFeatureConfig(
        coord_mode=str(cfg.coord_mode),
        orb_mult=float(cfg.orb_mult),
        include_pair_aspects=bool(cfg.include_pair_aspects),
        include_phases=bool(cfg.include_phases),
        add_trig_for_longitudes=bool(cfg.add_trig_for_longitudes),
        add_trig_for_moon_phase=bool(cfg.add_trig_for_moon_phase),
        add_trig_for_elongations=bool(cfg.add_trig_for_elongations),
        schema=f"{cfg.schema}_ephem",
    )

    df_ephem = build_ephemeris_feature_set(
        df_market=base,
        cfg=ephem_cfg,
        cache_namespace=f"{cache_namespace}_ephem",
        use_cache=use_cache,
        verbose=verbose,
        progress=progress,
    )

    df_all = df_ephem.copy()

    if cfg.include_transit_aspects:
        df_transit_feat = build_transit_to_natal_feature_set(
            df_market=base,
            birth_dt_utc=birth_dt_utc,
            orb_mult=float(cfg.orb_mult),
            cache_namespace=f"{cache_namespace}_transits",
            use_cache=use_cache,
            verbose=verbose,
            progress=progress,
        )
        df_all = pd.merge(df_all, df_transit_feat, on="date", how="left")

    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all = df_all.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    feature_cols = [c for c in df_all.columns if c != "date"]
    if feature_cols:
        df_all[feature_cols] = df_all[feature_cols].fillna(0)

    if verbose:
        print(
            "Full turning astro feature set:",
            f"rows={len(df_all)}",
            f"cols={len(feature_cols)}",
            f"birth_dt_utc={birth_dt_utc}",
            f"cfg={asdict(cfg)}",
        )

    return df_all


def classify_feature_group(col: str) -> str:
    """Assign one readable group name for feature diagnostics."""
    if col.startswith("transit_aspect_hit_"):
        return "transit_aspect_hit"
    if col.startswith("transit_aspect_min_orb_"):
        return "transit_aspect_orb"
    if col.startswith("aspect_hit_"):
        return "pair_aspect_hit"
    if col.startswith("aspect_min_orb_"):
        return "pair_aspect_orb"
    if col.startswith("moon_phase_") or col.startswith("lunar_day"):
        return "moon_phase"
    if col.endswith("_elongation") or "_elongation_" in col:
        return "elongation"
    if col.startswith("geo_"):
        return "geo_body"
    if col.startswith("helio_"):
        return "helio_body"
    if col.endswith("_trig_sin") or col.endswith("_trig_cos"):
        return "trig"
    if col.endswith("_lon") or col.endswith("_speed") or col.endswith("_declination"):
        return "body_raw"
    return "other"


def summarize_feature_groups(feature_cols: Sequence[str], examples_per_group: int = 4) -> pd.DataFrame:
    """
    Return compact feature group summary for notebook console logging.

    Columns:
    - group
    - n_features
    - example_cols
    """
    rows: list[dict[str, object]] = []
    if not feature_cols:
        return pd.DataFrame(columns=["group", "n_features", "example_cols"])

    groups: dict[str, list[str]] = {}
    for c in feature_cols:
        g = classify_feature_group(str(c))
        groups.setdefault(g, []).append(str(c))

    for g, cols in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        ex = ", ".join(cols[: max(1, int(examples_per_group))])
        rows.append(
            {
                "group": g,
                "n_features": int(len(cols)),
                "example_cols": ex,
            }
        )

    return pd.DataFrame(rows)
