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
    df_phases: Optional[pd.DataFrame] = None,  # NEW: фазы Луны и элонгации
    include_pair_aspects: bool = True,
    include_transit_aspects: bool = False,
    exclude_bodies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    ═══════════════════════════════════════════════════════════════════════════════
    ПОСТРОЕНИЕ ПОЛНОЙ МАТРИЦЫ ПРИЗНАКОВ ИЗ АСТРОЛОГИЧЕСКИХ ДАННЫХ
    ═══════════════════════════════════════════════════════════════════════════════
    
    Эта функция объединяет ВСЕ виды астрологических признаков:
    
    1. ПОЗИЦИИ ТЕЛ (df_bodies):
       - Долгота планеты (Sun_lon, Moon_lon, Mars_lon...)
       - Скорость движения (Sun_speed, Moon_speed...)
       - Ретроградность (Mars_is_retro, Mercury_is_retro...)
       - Склонение (declination)
       - Знак зодиака (sign_idx 0-11)
    
    2. АСПЕКТЫ (df_aspects):
       - Есть ли аспект между парой планет сегодня
       - Орбис (точность) аспекта
       
    3. ТРАНЗИТЫ (df_transits, опционально):
       - Аспекты транзитных планет к натальным
       
    4. ФАЗЫ И ЭЛОНГАЦИИ (df_phases, НОВОЕ!):
       - moon_phase_angle - угол фазы Луны (0-360°)
       - moon_phase_ratio - нормализованная фаза (0=новолуние, 0.5=полнолуние)
       - moon_illumination - освещённость Луны (0-1)
       - lunar_day - лунный день (1-29.5)
       - Mercury_elongation - элонгация Меркурия от Солнца
       - Venus_elongation - элонгация Венеры
       - ... и т.д.
    
    EXCLUDE_BODIES:
    ─────────────────────────────────────────────────────────────────────────────
    Список тел для исключения (используется при grid search).
    Например: ["Pluto", "Neptune"] — исключить Плутон и Нептун из признаков.
    ═══════════════════════════════════════════════════════════════════════════════
    
    Args:
        df_bodies: Body positions DataFrame
        df_aspects: Pair aspects DataFrame
        df_transits: Transit aspects DataFrame (optional)
        df_phases: Moon phases and planet elongations DataFrame (optional)
        include_pair_aspects: Include body pair aspects
        include_transit_aspects: Include transit-to-natal aspects
        exclude_bodies: List of body names to exclude (for grid search)
    
    Returns:
        Feature DataFrame with date column + feature columns
    """
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 1: Фильтруем исключённые тела (если указаны)
    # ─────────────────────────────────────────────────────────────────────────────
    if exclude_bodies and len(exclude_bodies) > 0:
        # Расширяем список исключений для учета префиксов (geo_, helio_)
        # Это важно для режимов coord_mode="both" и "helio"
        exclude_set = set()
        for body in exclude_bodies:
            exclude_set.add(body)
            exclude_set.add(f"geo_{body}")
            exclude_set.add(f"helio_{body}")
            
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
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 2: Строим базовые признаки (позиции + аспекты)
    # ─────────────────────────────────────────────────────────────────────────────
    df_features = build_features_daily(
        df_bodies,
        df_aspects,
        df_transits,
        include_pair_aspects=include_pair_aspects,
        include_transit_aspects=include_transit_aspects,
    )
    
    # ─────────────────────────────────────────────────────────────────────────────
    # ШАГ 3: НОВОЕ - Добавляем фазы Луны и элонгации планет
    # ─────────────────────────────────────────────────────────────────────────────
    if df_phases is not None and not df_phases.empty:
        # Убеждаемся что date в нужном формате
        df_phases = df_phases.copy()
        df_phases["date"] = pd.to_datetime(df_phases["date"])
        df_features["date"] = pd.to_datetime(df_features["date"])
        
        # Объединяем по дате
        df_features = pd.merge(
            df_features,
            df_phases,
            on="date",
            how="left"
        )
        
        # Заполняем пропуски нулями
        phase_cols = [c for c in df_phases.columns if c != "date"]
        df_features[phase_cols] = df_features[phase_cols].fillna(0)
    
    return df_features


def merge_features_with_labels(
    df_features: pd.DataFrame,
    df_labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge features with labels on date (LEFT JOIN + FORWARD-FILL).
    
    All feature days are kept. Labels are forward-filled so each day
    inherits the previous label until a new label appears.
    
    Args:
        df_features: Feature DataFrame with 'date' column
        df_labels: Labels DataFrame with 'date' and 'target' columns
    
    Returns:
        Merged DataFrame with features and target (NO GAPS)
    """
    df_features = df_features.copy()
    df_labels = df_labels.copy()
    
    df_features["date"] = pd.to_datetime(df_features["date"])
    df_labels["date"] = pd.to_datetime(df_labels["date"])
    
    # LEFT JOIN: keep ALL feature dates
    df_merged = pd.merge(
        df_features,
        df_labels[["date", "target"]],
        on="date",
        how="left"  # CHANGED from 'inner' to 'left'
    )
    
    # Sort by date and forward-fill labels
    df_merged = df_merged.sort_values("date").reset_index(drop=True)
    df_merged["target"] = df_merged["target"].ffill()  # State continues until change
    
    # Drop initial rows without any label (before first label appears)
    first_label_idx = df_merged["target"].first_valid_index()
    if first_label_idx is not None and first_label_idx > 0:
        df_merged = df_merged.iloc[first_label_idx:].reset_index(drop=True)
    
    # Remove duplicates
    if df_merged["date"].duplicated().any():
        df_merged = df_merged.drop_duplicates(subset=["date"]).reset_index(drop=True)
    
    # Convert target to int
    df_merged["target"] = df_merged["target"].astype(int)
    
    print(f"Merged dataset: {len(df_merged)} samples (ALL days, forward-filled)")
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
