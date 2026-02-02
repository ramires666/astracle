"""
Features module for RESEARCH pipeline.
Builds astro feature matrix from body positions and aspects.

TODO: Grid search for excluding individual astro objects
TODO: Add moon phases and other planet phases as features
TODO: Add houses and aspects to houses for birth date grid search
"""
import pandas as pd
import numpy as np
from typing import Optional, List

from src.features.builder import (
    build_body_features,
    build_aspect_pair_features,
    build_transit_aspect_features,
    build_features_daily,
)


def build_full_features(
    df_bodies: pd.DataFrame,
    df_aspects: pd.DataFrame,
    df_transits: Optional[pd.DataFrame] = None,
    include_pair_aspects: bool = True,
    include_transit_aspects: bool = False,
    exclude_bodies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build full feature matrix from astro data.
    
    Args:
        df_bodies: Body positions DataFrame
        df_aspects: Pair aspects DataFrame
        df_transits: Transit aspects DataFrame (optional)
        include_pair_aspects: Include body pair aspects
        include_transit_aspects: Include transit-to-natal aspects
        exclude_bodies: List of body names to exclude (for grid search)
    
    Returns:
        Feature DataFrame with date column + feature columns
    """
    # Filter out excluded bodies if specified
    if exclude_bodies and len(exclude_bodies) > 0:
        exclude_set = set(exclude_bodies)
        df_bodies = df_bodies[~df_bodies["body"].isin(exclude_set)].copy()
        
        if df_aspects is not None and not df_aspects.empty:
            df_aspects = df_aspects[
                ~(df_aspects["p1"].isin(exclude_set) | df_aspects["p2"].isin(exclude_set))
            ].copy()
        
        if df_transits is not None and not df_transits.empty:
            df_transits = df_transits[
                ~(df_transits["transit_body"].isin(exclude_set) | 
                  df_transits["natal_body"].isin(exclude_set))
            ].copy()
    
    # Use existing feature builder
    df_features = build_features_daily(
        df_bodies,
        df_aspects,
        df_transits,
        include_pair_aspects=include_pair_aspects,
        include_transit_aspects=include_transit_aspects,
    )
    
    return df_features


def merge_features_with_labels(
    df_features: pd.DataFrame,
    df_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge features with labels on date.
    
    Args:
        df_features: Feature DataFrame with 'date' column
        df_labels: Labels DataFrame with 'date' and 'target' columns
    
    Returns:
        Merged DataFrame with features and target
    """
    df_features = df_features.copy()
    df_labels = df_labels.copy()
    
    df_features["date"] = pd.to_datetime(df_features["date"])
    df_labels["date"] = pd.to_datetime(df_labels["date"])
    
    # Merge on date (inner join - only dates with both)
    df_merged = pd.merge(
        df_features,
        df_labels[["date", "target"]],
        on="date",
        how="inner"
    )
    
    # Remove duplicates
    if df_merged["date"].duplicated().any():
        df_merged = df_merged.drop_duplicates(subset=["date"]).reset_index(drop=True)
    
    df_merged = df_merged.sort_values("date").reset_index(drop=True)
    
    print(f"Merged dataset: {len(df_merged)} samples")
    print(f"Features: {len([c for c in df_merged.columns if c not in ['date', 'target']])}")
    
    return df_merged


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature column names (excluding date and target)."""
    return [c for c in df.columns if c not in ["date", "target"]]


def get_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Get statistics for all features."""
    feature_cols = get_feature_columns(df)
    
    stats = df[feature_cols].describe().T[["mean", "std", "min", "max"]]
    missing_pct = df[feature_cols].isna().mean() * 100
    stats = stats.join(missing_pct.rename("missing_%"), how="left")
    
    return stats


def feature_group(name: str) -> str:
    """Classify feature into group by name prefix."""
    if name.startswith("transit_aspect_"):
        return "transit_aspect"
    if name.startswith("aspect_"):
        return "aspect"
    if "_" in name:
        return name.split("_", 1)[0]
    return "other"


def get_feature_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Get feature inventory with groups and stats."""
    feature_cols = get_feature_columns(df)
    
    info = pd.DataFrame({"feature": feature_cols})
    info["group"] = info["feature"].apply(feature_group)
    
    stats = get_feature_stats(df)
    info = info.merge(stats, left_on="feature", right_index=True, how="left")
    info = info.sort_values(["group", "feature"]).reset_index(drop=True)
    
    return info
