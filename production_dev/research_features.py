"""
Research Features Bridge

This module provides production access to the same feature engineering
code used in RESEARCH for model training, ensuring consistent predictions.

Usage:
    from production_dev.research_features import FeatureBuilder
    
    builder = FeatureBuilder(config)
    features = builder.build_features_for_date(date(2025, 1, 15))
"""

import sys
from pathlib import Path
from datetime import date, datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Add project root to path to import RESEARCH modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import RESEARCH modules
from RESEARCH.astro_engine import (
    calculate_bodies_for_dates,
    calculate_bodies_for_dates_multi,
    calculate_aspects_for_dates,
    calculate_phases_for_dates,
)
from RESEARCH.features import build_full_features

# Import settings from src/RESEARCH
from src.astro.engine.settings import AstroSettings
from src.astro.engine.calculator import set_ephe_path
from RESEARCH.config import cfg


class FeatureBuilder:
    """
    Production-ready feature builder using RESEARCH code.
    
    Ensures features are calculated exactly as during training.
    """
    
    def __init__(
        self,
        birth_date: str = "2009-10-10",
        coord_mode: str = "both",
        orb_mult: float = 0.1,
        exclude_bodies: Optional[List[str]] = None,
    ):
        """
        Initialize feature builder.
        
        Args:
            birth_date: Bitcoin's natal date
            coord_mode: Coordinate mode ('geo', 'helio', or 'both')
            orb_mult: Orb multiplier for aspects
            exclude_bodies: Bodies to exclude from features
        """
        self.birth_date = birth_date
        self.coord_mode = coord_mode
        self.orb_mult = orb_mult
        self.exclude_bodies = exclude_bodies
        
        # Load astro config and set ephemeris path
        astro_cfg = cfg.get_astro_config()
        set_ephe_path(str(astro_cfg["ephe_path"]))
        
        # Create AstroSettings (same as RESEARCH/astro_engine.py)
        self.settings = AstroSettings(
            bodies_path=astro_cfg["bodies_path"],
            aspects_path=astro_cfg["aspects_path"],
        )
        
        self._feature_names: Optional[List[str]] = None
    
    def build_features_for_dates(
        self,
        dates: List[date],
        progress: bool = False,
    ) -> pd.DataFrame:
        """
        Build complete feature matrix for multiple dates.
        
        This uses the exact same code as RESEARCH/features.py
        to ensure consistent feature calculation.
        
        Args:
            dates: List of dates to calculate features for
            progress: Show progress bar
            
        Returns:
            DataFrame with columns: date + all feature columns
        """
        # Ensure dates are python date objects (not datetime)
        # calculate_bodies_for_dates handles pd.to_datetime internally
        date_objs = [
            d.date() if isinstance(d, datetime) else d
            for d in dates
        ]
        
        # Step 1: Calculate body positions for all dates
        df_bodies, bodies_by_date = calculate_bodies_for_dates(
            dates=date_objs,
            settings=self.settings,
            progress=progress,
        )
        
        # Step 2: Calculate aspects between bodies
        df_aspects = calculate_aspects_for_dates(
            bodies_by_date=bodies_by_date,
            settings=self.settings,
            orb_mult=self.orb_mult,
            progress=progress,
        )
        
        # Step 3: Calculate Moon phases and elongations
        df_phases = calculate_phases_for_dates(
            bodies_by_date=bodies_by_date,
            progress=progress,
        )
        
        # Step 4: Build full feature matrix
        df_features = build_full_features(
            df_bodies=df_bodies,
            df_aspects=df_aspects,
            df_phases=df_phases,
            exclude_bodies=self.exclude_bodies,
        )
        
        # Store feature names (excluding date)
        self._feature_names = [c for c in df_features.columns if c != 'date']
        
        return df_features
    
    def build_features_for_date(self, target_date: date) -> np.ndarray:
        """
        Build feature vector for a single date.
        
        Args:
            target_date: Date to calculate features for
            
        Returns:
            1D numpy array of feature values
        """
        df = self.build_features_for_dates([target_date])
        
        if len(df) == 0:
            raise ValueError(f"Could not calculate features for {target_date}")
        
        # Return feature values (excluding date column)
        feature_cols = [c for c in df.columns if c != 'date']
        return df[feature_cols].values[0]
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names from last calculation."""
        if self._feature_names is None:
            # Calculate for a sample date to get names
            self.build_features_for_date(date(2025, 1, 1))
        return self._feature_names


def build_features_matching_model(
    dates: List[date],
    model_config: Dict[str, Any],
    model_feature_names: List[str],
) -> pd.DataFrame:
    """
    Build features that exactly match the model's expected input.
    
    Args:
        dates: Dates to calculate features for
        model_config: Configuration from loaded model
        model_feature_names: Feature names from loaded model
        
    Returns:
        DataFrame with features in correct order
    """
    builder = FeatureBuilder(
        birth_date=model_config.get('birth_date', '2009-10-10'),
        coord_mode=model_config.get('coord_mode', 'both'),
        orb_mult=model_config.get('orb_mult', 0.1),
        exclude_bodies=model_config.get('exclude_bodies'),
    )
    
    df = builder.build_features_for_dates(dates, progress=True)
    
    # Ensure columns match model expectations
    missing = set(model_feature_names) - set(df.columns)
    extra = set(df.columns) - set(model_feature_names) - {'date'}
    
    if missing:
        print(f"⚠️ Missing {len(missing)} features: {list(missing)[:5]}...")
        # Add missing as zeros
        for col in missing:
            df[col] = 0.0
    
    if extra:
        print(f"⚠️ Extra {len(extra)} features (will ignore): {list(extra)[:5]}...")
    
    # Reorder to match model
    return df[['date'] + model_feature_names]
