"""
Bitcoin Astrology Predictor Module

This module provides the core prediction functionality for the Bitcoin Astro Prediction Service.
It wraps the trained XGBoost model and generates future price direction predictions based on
astrological features calculated from ephemeris data.

Model Configuration (from birthdate_deep_search.ipynb):
- Birth Date: 2009-10-10 (Bitcoin Economic Birth)
- R_MIN: 0.603, MCC: 0.315
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import joblib

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.xgb import XGBBaseline
from production_dev.coingecko_client import fetch_current_btc_price_usd


class BtcAstroPredictor:
    """
    Bitcoin Astrology Predictor
    
    Uses a trained XGBoost model with astrological features to predict
    Bitcoin price direction (UP/DOWN) for future dates.
    
    Attributes:
        model: Trained XGBBaseline model
        scaler: Feature scaler (integrated in model)
        natal_date: Bitcoin's natal chart date
        config: Model and astro configuration
    """
    
    # =========================================================================
    # WINNING MODEL CONFIGURATION
    # =========================================================================
    DEFAULT_CONFIG = {
        # Astro Parameters
        "birth_date": "2009-10-10",
        "coord_mode": "both",
        "orb_mult": 0.1,
        "gauss_window": 200,
        "gauss_std": 70.0,
        "exclude_bodies": None,
        
        # XGBoost Parameters
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "colsample_bytree": 0.6,
        "subsample": 0.8,
        
        # Performance Metrics (for reference)
        "r_min": 0.603,
        "mcc": 0.315,

        # -----------------------------------------------------------------
        # UI-ONLY: Simulated Price Path Settings
        # -----------------------------------------------------------------
        # The dashboard "Forecast" line is NOT a price target. It is a visual
        # random-walk that converts (UP/DOWN + confidence) into a smooth path.
        # We also apply a small asymmetry (UP days a bit stronger than DOWN),
        # because on BTC daily closes (2017-2025) the average absolute UP move
        # was ~7% larger than the average absolute DOWN move.
        "sim_base_move": 0.006,   # ~0.6% minimum daily move when confidence=50%
        "sim_conf_move": 0.020,   # +0%..+2.0% extra move when confidence goes 50%->100%
        "sim_jitter": 0.008,      # ±0.8% random noise for natural look
        "sim_up_mult": 1.035,     # UP days slightly stronger in magnitude
        "sim_down_mult": 0.965,   # DOWN days slightly weaker in magnitude
    }
    
    # Historical BTC daily volatility (approximate)
    BTC_DAILY_VOLATILITY = 0.035  # ~3.5% daily standard deviation
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to saved model file (.joblib)
            config: Optional configuration override
        """
        self.config = config or self.DEFAULT_CONFIG.copy()
        self.model: Optional[XGBBaseline] = None
        self.feature_names: Optional[List[str]] = None
        self.is_loaded = False
        
        # Default model path
        if model_path is None:
            model_path = PROJECT_ROOT / "models_artifacts" / "btc_astro_predictor.joblib"
        self.model_path = Path(model_path)
        
        # Initialize ephemeris
        self._init_ephemeris()
    
    def _init_ephemeris(self) -> None:
        """Initialize Swiss Ephemeris for astronomical calculations."""
        try:
            import swisseph as swe
            # Set ephemeris path if available
            ephe_path = PROJECT_ROOT / "data" / "ephe"
            if ephe_path.exists():
                swe.set_ephe_path(str(ephe_path))
            self.swe = swe
        except ImportError:
            raise RuntimeError("pyswisseph is required for ephemeris calculations")
    
    def load_model(self) -> bool:
        """
        Load the trained model from disk.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not self.model_path.exists():
            print(f"Model file not found: {self.model_path}")
            return False
        
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data["model"]
            self.feature_names = model_data.get("feature_names", [])
            self.config.update(model_data.get("config", {}))
            self.is_loaded = True
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_direction(self, target_date: date) -> Tuple[int, float]:
        """
        Predict price direction for a single date.
        
        Args:
            target_date: Date to predict
            
        Returns:
            Tuple of (direction, confidence) where:
            - direction: 1 for UP, 0 for DOWN
            - confidence: Probability of the predicted direction
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Generate features for target date
        features = self._calculate_features(target_date)
        
        # Prepare input array
        X = np.array([features])

        # ---------------------------------------------------------------------
        # IMPORTANT: How we do inference (this is where the previous bug was)
        # ---------------------------------------------------------------------
        # Our research training code uses `src.models.xgb.XGBBaseline`.
        #
        # XGBBaseline is NOT just an XGBoost model:
        # - It contains a fitted `RobustScaler` (model.scaler)
        # - It trains the internal XGBoost classifier on SCALED features
        #
        # That means at inference time we MUST apply the same scaler,
        # otherwise the XGBoost trees will receive numbers in the wrong scale
        # and predictions will look almost random (very bad backtest).
        proba = self._predict_proba(X)[0]

        # ---------------------------------------------------------------------
        # Threshold (decision boundary)
        # ---------------------------------------------------------------------
        # In research we tune a probability threshold on the validation split
        # (see RESEARCH.model_training.tune_threshold, metric="recall_min").
        #
        # To reproduce the exact notebook metrics, the tuned threshold should be
        # stored in the model artifact config.
        #
        # If it is missing, we fall back to 0.5 (standard).
        threshold = float(
            self.config.get("decision_threshold", self.config.get("threshold", 0.5))
        )

        # Binary class: index 1 is "UP", index 0 is "DOWN"
        direction = 1 if float(proba[1]) >= threshold else 0
        confidence = float(proba[direction])
        
        return direction, confidence

    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for a feature matrix.

        Why this helper exists:
        - We have TWO artifact formats in this repo:
          1) Split model from research: `XGBBaseline` (has a scaler!)
          2) Some older full models: plain `xgboost.sklearn.XGBClassifier`

        If we don't handle both cases, the frontend cache can silently become
        wrong (history looks much worse than the notebook).

        Args:
            X: Feature matrix (n_samples, n_features) in the SAME column order
               as `self.feature_names` (we ensure that in `_calculate_features`).

        Returns:
            Array of probabilities with shape (n_samples, 2):
            - [:, 0] = P(DOWN)
            - [:, 1] = P(UP)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Case 1: Our wrapper (research pipeline)
        if isinstance(self.model, XGBBaseline):
            # Extremely rare edge case: training data had only one class,
            # so the model becomes a constant predictor.
            if getattr(self.model, "constant_class", None) is not None:
                const = int(self.model.constant_class)
                out = np.zeros((X.shape[0], 2), dtype=float)
                out[:, const] = 1.0
                return out

            # Normal case: scale -> predict_proba
            X_scaled = self.model.scaler.transform(X)
            return self.model.model.predict_proba(X_scaled)

        # Case 2: Plain XGBoost classifier (no scaler stored)
        # We assume it was trained on raw (unscaled) features.
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        raise TypeError(
            "Unsupported model type in artifact. Expected XGBBaseline or XGBClassifier-like "
            "object with predict_proba()."
        )
    
    def predict_next_n_days(self, n_days: int = 90) -> List[Dict]:
        """
        Generate predictions for the next N days.
        
        Args:
            n_days: Number of days to predict (default: 90)
            
        Returns:
            List of prediction dictionaries with keys:
            - date: ISO date string
            - direction: "UP" or "DOWN"
            - direction_code: 1 or 0
            - confidence: Prediction confidence (0-1)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        predictions = []
        start_date = date.today() + timedelta(days=1)  # Start from tomorrow
        
        for i in range(n_days):
            target_date = start_date + timedelta(days=i)
            direction, confidence = self.predict_direction(target_date)
            
            predictions.append({
                "date": target_date.isoformat(),
                "direction": "UP" if direction == 1 else "DOWN",
                "direction_code": direction,
                "confidence": round(confidence, 4),
            })
        
        return predictions
    
    def generate_price_path(
        self,
        predictions: List[Dict],
        start_price: Optional[float] = None,
        volatility: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate a pseudo-random price path based on predictions.
        
        Uses random walk with drift, where:
        - Drift strength is determined by confidence (50% = sideways, 100% = max move)
        - Drift direction is determined by model prediction (UP/DOWN)
        - Volatility is based on historical BTC volatility
        
        Args:
            predictions: List of prediction dicts from predict_next_n_days()
            start_price: Starting price (fetched from API if None)
            volatility: Daily volatility (uses historical if None)
            seed: Random seed for reproducibility
            
        Returns:
            Enhanced predictions list with simulated prices
        """
        if start_price is None:
            start_price = self._fetch_current_btc_price()
        
        if volatility is None:
            volatility = self.BTC_DAILY_VOLATILITY
        
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random walk with prediction-based drift
        prices = [start_price]
        
        for pred in predictions:
            # Get confidence (0.5 = neutral, 1.0 = max certainty)
            confidence = pred.get("confidence", 0.5)
            
            # Calculate confidence strength (0 when conf=0.5, 1 when conf=1.0)
            # This maps [0.5, 1.0] to [0, 1]
            confidence_strength = max(0, (confidence - 0.5) * 2)
            
            # Direction: convert 0/1 to -1/+1 for price movement
            # direction_code: 0=DOWN, 1=UP
            # We need: DOWN=-1, UP=+1
            direction = 1 if pred["direction_code"] == 1 else -1
            
            # DETERMINISTIC price movement with micro-jitter for natural look
            # Move amount:
            # - Base move is always applied (so the line doesn't look flat)
            # - Extra move is proportional to confidence_strength
            #
            # These constants are UI-only (see DEFAULT_CONFIG).
            base_move = float(self.config.get("sim_base_move", 0.006))
            conf_move = float(self.config.get("sim_conf_move", 0.020))
            move_percent = base_move + conf_move * confidence_strength

            # Small asymmetry: UP days are a bit stronger than DOWN days.
            # This matches a rough empirical property of BTC daily returns.
            if pred["direction_code"] == 1:
                move_percent *= float(self.config.get("sim_up_mult", 1.035))
            else:
                move_percent *= float(self.config.get("sim_down_mult", 0.965))
            
            # Random jitter for a more natural line.
            # We keep it symmetric so we don't accidentally add hidden drift.
            jitter_amp = float(self.config.get("sim_jitter", 0.008))
            jitter = np.random.uniform(-jitter_amp, jitter_amp)

            price_change = direction * move_percent + jitter
            
            # Calculate new price
            new_price = prices[-1] * (1 + price_change)
            new_price = max(new_price, 100)  # Floor at $100 for sanity
            prices.append(new_price)
            
            # Add price to prediction
            pred["simulated_price"] = round(new_price, 2)
        
        return predictions
    def _fetch_current_btc_price(self) -> float:
        """
        Fetch current BTC price from CoinGecko API.
        
        Returns:
            Current BTC price in USD
        """
        try:
            return fetch_current_btc_price_usd(timeout=10)
        except Exception as e:
            print(f"Error fetching BTC price: {e}. Falling back to local market close.")
            try:
                from production_dev.data_service import get_current_price
                fallback = float(get_current_price())
                if fallback > 0:
                    return fallback
            except Exception as fallback_error:
                print(f"Error loading local fallback price: {fallback_error}")
            return 100000.0  # Final emergency fallback
    
    def _calculate_features(self, target_date: date) -> np.ndarray:
        """
        Calculate astrological features for a target date.
        
        Uses the same feature pipeline as training (RESEARCH modules).
        Features are reindexed to match the model's expected feature order.
        
        Args:
            target_date: Date to calculate features for
            
        Returns:
            Feature vector as numpy array, aligned with self.feature_names
        """
        try:
            from RESEARCH.astro_engine import (
                init_ephemeris,
                calculate_bodies_for_dates_multi,
                calculate_aspects_for_dates,
                calculate_transits_for_dates,
                calculate_phases_for_dates,
                get_natal_bodies,
            )
            from RESEARCH.features import build_full_features
            
            settings = init_ephemeris()
            
            # Calculate for single date using the same pipeline as training
            dates = pd.Series([target_date])
            coord_mode = self.config.get("coord_mode", "both")
            orb_mult = self.config.get("orb_mult", 0.1)
            
            # Calculate body positions (geo + helio if coord_mode="both")
            df_bodies, geo_dict, helio_dict = calculate_bodies_for_dates_multi(
                dates, settings, coord_mode, progress=False
            )
            bodies_by_date = geo_dict
            
            # Calculate phases (moon phases, elongations)
            df_phases = calculate_phases_for_dates(bodies_by_date, progress=False)
            
            # Get natal bodies for transit calculations
            birth_date = self.config.get("birth_date", "2009-10-10")
            natal_dt_str = f"{birth_date}T12:00:00"
            natal_bodies = get_natal_bodies(natal_dt_str, settings)
            
            # Calculate transits (current planets to natal positions)
            df_transits = calculate_transits_for_dates(
                bodies_by_date, natal_bodies, settings,
                orb_mult=orb_mult, progress=False
            )
            
            # Calculate aspects (current planets to each other)
            df_aspects = calculate_aspects_for_dates(
                bodies_by_date, settings,
                orb_mult=orb_mult, progress=False
            )
            
            # Build full feature set using same function as training
            df_features = build_full_features(
                df_bodies, df_aspects, 
                df_transits=df_transits, 
                df_phases=df_phases,
                include_pair_aspects=True,
                include_transit_aspects=True
            )
            
            if len(df_features) == 0:
                raise ValueError("No features generated for date")
            
            # Get the feature row (exclude date column)
            feature_cols = [c for c in df_features.columns if c != "date"]
            feature_row = df_features[feature_cols].iloc[0]
            
            # IMPORTANT: Reindex to match model's expected feature order
            # This ensures features are in the exact order the model was trained on
            if self.feature_names:
                # Create a series with all expected features, fill missing with 0
                aligned_features = pd.Series(0.0, index=self.feature_names)
                
                # Fill in the features we calculated
                for col in feature_cols:
                    if col in aligned_features.index:
                        val = feature_row[col]
                        # Convert booleans to float
                        if isinstance(val, (bool, np.bool_)):
                            val = float(val)
                        aligned_features[col] = val
                
                n_matched = sum(1 for c in feature_cols if c in self.feature_names)
                n_expected = len(self.feature_names)
                
                # Note: Low match rate is EXPECTED for single-day predictions
                # because most aspect/transit features only fire on specific days
                # The remaining features are filled with 0 (aspect not active)
                
                # Ensure all values are float
                return aligned_features.astype(float).values
            else:
                return feature_row.astype(float).values
            
        except ImportError as e:
            raise RuntimeError(f"Feature calculation requires RESEARCH modules: {e}")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "is_loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "config": self.config,
            "feature_count": len(self.feature_names) if self.feature_names else 0,
            "natal_date": self.config.get("birth_date"),
            "expected_r_min": self.config.get("r_min"),
            "expected_mcc": self.config.get("mcc"),
        }
