"""
Build astro features into a wide table.

MVP version:
1) Pivot body positions (lon/speed/retro/declination/sign)
2) Build aspect features per specific body pair
"""

from __future__ import annotations

from typing import List

import pandas as pd


def _coerce_date_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure merge key ``date`` is a normalized datetime64 column.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        bad_count = int(out["date"].isna().sum())
        raise ValueError(f"Column 'date' contains {bad_count} invalid values after parsing")
    # Keep calendar date stable and remove time/tz noise before joins.
    if out["date"].dt.tz is not None:
        out["date"] = out["date"].dt.tz_localize(None)
    out["date"] = out["date"].dt.normalize()
    return out


def _sign_to_index(sign: str) -> int:
    """
    Convert zodiac sign to index 0..11.
    """
    mapping = {
        "Aries": 0, "Taurus": 1, "Gemini": 2, "Cancer": 3,
        "Leo": 4, "Virgo": 5, "Libra": 6, "Scorpio": 7,
        "Sagittarius": 8, "Capricorn": 9, "Aquarius": 10, "Pisces": 11,
    }
    return mapping.get(sign, -1)


def build_body_features(df_bodies: pd.DataFrame) -> pd.DataFrame:
    """
    Turn body table into wide features.
    Expected columns: date, body, lon, speed, is_retro, declination, sign
    """
    df = _coerce_date_key(df_bodies)
    df["sign_idx"] = df["sign"].apply(_sign_to_index)

    # Pivot by bodies and parameters
    feature_frames: List[pd.DataFrame] = []
    for col in ["lon", "speed", "is_retro", "declination", "sign_idx"]:
        pivot = df.pivot(index="date", columns="body", values=col)
        pivot.columns = [f"{c}_{col}" for c in pivot.columns]
        feature_frames.append(pivot)

    features = pd.concat(feature_frames, axis=1).reset_index()
    return features


def _canonicalize_pairs(df: pd.DataFrame, left_col: str, right_col: str) -> pd.DataFrame:
    """
    Ensure pair order is stable by sorting body names in each row.
    """
    out = df.copy()
    p1 = out[left_col].astype(str)
    p2 = out[right_col].astype(str)
    swap = p1 > p2
    if swap.any():
        out.loc[swap, [left_col, right_col]] = out.loc[swap, [right_col, left_col]].values
    return out


def build_aspect_pair_features(df_aspects: pd.DataFrame) -> pd.DataFrame:
    """
    Aspect features per specific body pair:
    - hit (0/1) per (p1, p2, aspect)
    - min orb per (p1, p2, aspect)
    """
    if df_aspects.empty:
        return pd.DataFrame(columns=["date"])

    df = _coerce_date_key(df_aspects)
    df = _canonicalize_pairs(df, "p1", "p2")
    df["key"] = (
        df["p1"].astype(str)
        + "__"
        + df["p2"].astype(str)
        + "__"
        + df["aspect"].astype(str)
    )

    grouped = df.groupby(["date", "key"]).agg(
        hit=("aspect", "count"),
        min_orb=("orb", "min"),
    ).reset_index()
    grouped["hit"] = (grouped["hit"] > 0).astype(int)

    hit_pivot = grouped.pivot(index="date", columns="key", values="hit")
    hit_pivot.columns = [f"aspect_hit_{c}" for c in hit_pivot.columns]

    orb_pivot = grouped.pivot(index="date", columns="key", values="min_orb")
    orb_pivot.columns = [f"aspect_min_orb_{c}" for c in orb_pivot.columns]

    features = pd.concat([hit_pivot, orb_pivot], axis=1).reset_index()
    return features


def build_transit_aspect_features(df_transit_aspects: pd.DataFrame) -> pd.DataFrame:
    """
    Transit->natal aspect aggregates per day:
    - hit (0/1) per (transit_body, natal_body, aspect)
    - min orb per (transit_body, natal_body, aspect)
    """
    if df_transit_aspects.empty:
        return pd.DataFrame(columns=["date"])

    df = _coerce_date_key(df_transit_aspects)
    grouped = df.groupby(
        ["date", "transit_body", "natal_body", "aspect"]
    ).agg(
        hit=("aspect", "count"),
        min_orb=("orb", "min"),
    ).reset_index()
    grouped["hit"] = (grouped["hit"] > 0).astype(int)

    grouped["key"] = (
        grouped["transit_body"].astype(str)
        + "__"
        + grouped["natal_body"].astype(str)
        + "__"
        + grouped["aspect"].astype(str)
    )

    hit_pivot = grouped.pivot(index="date", columns="key", values="hit")
    hit_pivot.columns = [f"transit_aspect_hit_{c}" for c in hit_pivot.columns]

    orb_pivot = grouped.pivot(index="date", columns="key", values="min_orb")
    orb_pivot.columns = [f"transit_aspect_min_orb_{c}" for c in orb_pivot.columns]

    features = pd.concat([hit_pivot, orb_pivot], axis=1).reset_index()
    return features


def build_features_daily(
    df_bodies: pd.DataFrame,
    df_aspects: pd.DataFrame,
    df_transit_aspects: pd.DataFrame | None = None,
    include_pair_aspects: bool = True,
    include_transit_aspects: bool = False,
) -> pd.DataFrame:
    """
    Full feature build (bodies + aspects + transit->natal aspects).
    """
    body_features = build_body_features(df_bodies)

    merged = body_features
    if include_pair_aspects:
        pair_features = build_aspect_pair_features(df_aspects)
        if not pair_features.empty:
            merged = pd.merge(merged, pair_features, on="date", how="left")

    if include_transit_aspects and df_transit_aspects is not None and not df_transit_aspects.empty:
        transit_features = build_transit_aspect_features(df_transit_aspects)
        merged = pd.merge(merged, transit_features, on="date", how="left")

    merged = merged.fillna(0)
    return merged
