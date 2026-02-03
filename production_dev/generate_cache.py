"""
Generate Prediction Cache

Pre-calculates predictions for:
- Backtest: Past 6 months (with actual prices for accuracy)
- Forecast: Future 1 year

Run this script once or periodically to update the cache:
    python -m production_dev.generate_cache

Progress will be shown with estimated time remaining.
"""

import os
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from production_dev.predictor import BtcAstroPredictor
from production_dev.cache_service import (
    save_predictions_to_cache,
    BACKTEST_DAYS,
    FORECAST_DAYS,
    ensure_cache_dir,
)


def generate_backtest_predictions(
    predictor: BtcAstroPredictor,
    days: int = BACKTEST_DAYS,
) -> List[Dict]:
    """
    Generate predictions for past dates (backtest).
    
    Args:
        predictor: Loaded BtcAstroPredictor instance
        days: Number of past days to generate
        
    Returns:
        List of prediction dictionaries
    """
    print(f"\n📊 Generating {days} days of BACKTEST predictions...")
    
    predictions = []
    end_date = date.today() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=days)
    
    current_date = start_date
    
    with tqdm(total=days, desc="Backtest", unit="day") as pbar:
        while current_date <= end_date:
            try:
                direction_code, confidence = predictor.predict_direction(current_date)
                
                predictions.append({
                    "date": current_date.isoformat(),
                    "direction": "UP" if direction_code == 1 else "DOWN",
                    "direction_code": direction_code,
                    "confidence": round(confidence, 4),
                })
                
            except Exception as e:
                print(f"\n⚠️ Error for {current_date}: {e}")
                # Add placeholder for failed predictions
                predictions.append({
                    "date": current_date.isoformat(),
                    "direction": "UNKNOWN",
                    "direction_code": -1,
                    "confidence": 0.0,
                    "error": str(e),
                })
            
            current_date += timedelta(days=1)
            pbar.update(1)
    
    print(f"✅ Generated {len(predictions)} backtest predictions")
    return predictions


def generate_forecast_predictions(
    predictor: BtcAstroPredictor,
    days: int = FORECAST_DAYS,
    start_price: float = None,
) -> List[Dict]:
    """
    Generate predictions for future dates (forecast).
    
    Args:
        predictor: Loaded BtcAstroPredictor instance
        days: Number of future days to generate
        start_price: Starting price for simulation
        
    Returns:
        List of prediction dictionaries with simulated prices
    """
    print(f"\n🔮 Generating {days} days of FORECAST predictions...")
    
    predictions = []
    start_date = date.today() + timedelta(days=1)  # Tomorrow
    
    with tqdm(total=days, desc="Forecast", unit="day") as pbar:
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            
            try:
                direction_code, confidence = predictor.predict_direction(current_date)
                
                predictions.append({
                    "date": current_date.isoformat(),
                    "direction": "UP" if direction_code == 1 else "DOWN",
                    "direction_code": direction_code,
                    "confidence": round(confidence, 4),
                })
                
            except Exception as e:
                print(f"\n⚠️ Error for {current_date}: {e}")
                predictions.append({
                    "date": current_date.isoformat(),
                    "direction": "UNKNOWN",
                    "direction_code": -1,
                    "confidence": 0.0,
                    "error": str(e),
                })
            
            pbar.update(1)
    
    # Add simulated price path
    if start_price is None:
        start_price = predictor._fetch_current_btc_price()
    
    predictions = predictor.generate_price_path(
        predictions, 
        start_price=start_price,
        seed=42,  # Reproducible
    )
    
    print(f"✅ Generated {len(predictions)} forecast predictions")
    return predictions


def load_historical_prices() -> pd.DataFrame:
    """Load historical prices from database for backtest accuracy."""
    try:
        from RESEARCH.data_loader import load_market_data
        
        df = load_market_data()
        # Ensure date column is datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # Keep only last 3 years for efficiency (matches backtest range)
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=1100)
        df = df[df["date"] >= cutoff].copy()
        
        print(f"📈 Loaded {len(df)} historical prices")
        return df
        
    except Exception as e:
        print(f"⚠️ Could not load historical prices: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def main():
    """Main cache generation function."""
    print("=" * 60)
    print("🚀 PREDICTION CACHE GENERATOR (Dual Model)")
    print("=" * 60)
    
    # Ensure cache directory exists
    cache_dir = ensure_cache_dir()
    print(f"📁 Cache directory: {cache_dir}")
    
    # Model paths
    SPLIT_MODEL = PROJECT_ROOT / "models_artifacts" / "btc_astro_predictor.joblib"  # Split-trained for honest backtest
    FULL_MODEL = PROJECT_ROOT / "models_artifacts" / "btc_astro_predictor.full.joblib"  # Full-trained for best forecast
    
    # =========================================
    # BACKTEST: Use split model (honest accuracy)
    # =========================================
    print("\n📦 Loading SPLIT model for backtest...")
    backtest_predictor = BtcAstroPredictor(model_path=SPLIT_MODEL)
    
    if not backtest_predictor.load_model():
        print("❌ ERROR: Could not load split model.")
        sys.exit(1)
    
    print(f"✅ Backtest model loaded: {backtest_predictor.config.get('birth_date')}")
    print(f"   Features: {len(backtest_predictor.feature_names)}")
    
    # Load historical prices for backtest
    historical_prices = load_historical_prices()
    
    # Generate backtest predictions
    backtest = generate_backtest_predictions(backtest_predictor, days=BACKTEST_DAYS)
    
    # Save backtest with actual prices
    save_predictions_to_cache(
        backtest, 
        "backtest",
        actual_prices=historical_prices if len(historical_prices) > 0 else None,
    )
    
    # =========================================
    # FORECAST: Use full model (best predictions)
    # =========================================
    print("\n📦 Loading FULL model for forecast...")
    
    # Check if full model exists, fallback to split if not
    if FULL_MODEL.exists():
        forecast_predictor = BtcAstroPredictor(model_path=FULL_MODEL)
        if not forecast_predictor.load_model():
            print("⚠️ Full model failed, falling back to split model")
            forecast_predictor = backtest_predictor
        else:
            print(f"✅ Forecast model loaded (FULL)")
    else:
        print("⚠️ Full model not found, using split model for forecast")
        forecast_predictor = backtest_predictor
    
    # Generate forecast predictions
    forecast = generate_forecast_predictions(forecast_predictor, days=FORECAST_DAYS)
    save_predictions_to_cache(forecast, "forecast")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ CACHE GENERATION COMPLETE")
    print("=" * 60)
    print(f"   Backtest: {len(backtest)} days (split model)")
    print(f"   Forecast: {len(forecast)} days (full model)")
    print(f"   Location: {cache_dir}")
    
    # Quick accuracy check
    if len(historical_prices) > 0:
        from production_dev.cache_service import get_backtest_with_accuracy
        _, stats = get_backtest_with_accuracy()
        print(f"\n📊 Backtest Accuracy: {stats['accuracy']:.1%}")
        print(f"   Total: {stats['total']}, Correct: {stats['correct']}")


if __name__ == "__main__":
    main()
