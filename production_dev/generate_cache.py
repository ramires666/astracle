"""
Generate Prediction Cache

Pre-calculates predictions for:
- Backtest: Full research period (train/val/test) with honest labels
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
    FORECAST_DAYS,
    ensure_cache_dir,
)
from production_dev.backtest_cache_builder import build_split_model_backtest_cache


def generate_forecast_predictions(
    predictor: BtcAstroPredictor,
    days: int = FORECAST_DAYS,
    forecast_start_date: date | None = None,
    start_price: float = None,
) -> List[Dict]:
    """
    Generate predictions for future dates (forecast).
    
    Args:
        predictor: Loaded BtcAstroPredictor instance
        days: Number of future days to generate
        forecast_start_date: First date to generate (inclusive).
            If None, defaults to "tomorrow" (date.today() + 1).
        start_price: Starting price for simulation
        
    Returns:
        List of prediction dictionaries with simulated prices
    """
    print(f"\n🔮 Generating {days} days of FORECAST predictions...")
    
    predictions = []
    # IMPORTANT:
    # We should NOT start the forecast from "tomorrow relative to when we ran the script",
    # because that creates gaps when the market data (backtest) ends earlier.
    #
    # Example (the exact bug you reported):
    # - last known actual price date in DB = Feb 4
    # - cache script was run later / with a wrong clock -> forecast starts Feb 8
    # - UI shows a gap Feb 5-7 (confusing and looks broken)
    #
    # So the caller can pass `forecast_start_date = last_backtest_date + 1`
    # to guarantee continuity between history and forecast.
    start_date = forecast_start_date or (date.today() + timedelta(days=1))  # default: tomorrow
    
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
    print("\n📦 Building research-exact BACKTEST cache (split model)...")

    # We build the backtest cache using the exact research pipeline:
    # - forward-filled labels (the same "target" used in the notebook)
    # - validation-tuned threshold (recall_min)
    # - explicit split tags (train/val/test) for honest chart labeling
    try:
        backtest_result = build_split_model_backtest_cache(
            split_model_path=SPLIT_MODEL,
            start_date="2017-11-01",
        )
    except Exception as e:
        print("❌ ERROR: Failed to build backtest cache.")
        print(f"   Reason: {e}")
        raise

    backtest_df = backtest_result.df_backtest
    backtest = backtest_df.to_dict(orient="records")

    # Save as-is. We do NOT merge with next-day price movement because
    # that would not match the notebook labels/metrics.
    save_predictions_to_cache(backtest, "backtest", actual_prices=None)

    print("✅ Backtest cache built")
    print(f"   Rows: {len(backtest_df)}")
    print(f"   Range: {backtest_result.meta.get('start_date')} -> {backtest_result.meta.get('end_date')}")
    print(f"   Threshold: {backtest_result.meta.get('decision_threshold')}")

    # Forecast should start right after the last backtest day (continuity, no gaps).
    forecast_start_date = None
    try:
        end_str = backtest_result.meta.get("end_date")
        if end_str:
            forecast_start_date = date.fromisoformat(str(end_str)) + timedelta(days=1)
    except Exception:
        forecast_start_date = None

    # Starting price for simulation:
    # - If we have the last actual close price in backtest, use it.
    #   This avoids a huge fake jump at the history->forecast boundary.
    start_price = None
    try:
        if "actual_price" in backtest_df.columns:
            last_price = backtest_df["actual_price"].dropna()
            if len(last_price) > 0:
                start_price = float(last_price.iloc[-1])
    except Exception:
        start_price = None
    
    # =========================================
    # FORECAST: Use full model (best predictions)
    # =========================================
    print("\n📦 Loading FULL model for forecast...")
    
    # Check if full model exists, fallback to split if not.
    forecast_predictor = None
    if FULL_MODEL.exists():
        candidate = BtcAstroPredictor(model_path=FULL_MODEL)
        if candidate.load_model():
            forecast_predictor = candidate
            print("✅ Forecast model loaded (FULL)")
        else:
            print("⚠️ Full model failed to load; will fall back to split model")

    if forecast_predictor is None:
        print("⚠️ Using SPLIT model for forecast fallback")
        candidate = BtcAstroPredictor(model_path=SPLIT_MODEL)
        if not candidate.load_model():
            print("❌ ERROR: Could not load split model for forecast fallback.")
            sys.exit(1)
        forecast_predictor = candidate
    
    # Generate forecast predictions
    forecast = generate_forecast_predictions(
        forecast_predictor,
        days=FORECAST_DAYS,
        forecast_start_date=forecast_start_date,
        start_price=start_price,
    )
    save_predictions_to_cache(forecast, "forecast")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ CACHE GENERATION COMPLETE")
    print("=" * 60)
    print(f"   Backtest: {len(backtest)} days (split model, full history)")
    print(f"   Forecast: {len(forecast)} days (full model)")
    print(f"   Location: {cache_dir}")
    
    # Quick accuracy check
    from production_dev.cache_service import get_backtest_with_accuracy
    _df, stats = get_backtest_with_accuracy(days=None)
    if stats.get("total", 0) > 0:
        print(f"\n📊 Backtest (TEST split) Accuracy: {stats['accuracy']:.1%}")
        if "r_min" in stats:
            print(f"   TEST R_MIN: {stats['r_min']:.3f}")
        if "mcc" in stats:
            print(f"   TEST MCC:   {stats['mcc']:.3f}")
        print(f"   Total: {stats['total']}, Correct: {stats['correct']}")

        # -----------------------------------------------------------------
        # Consistency check: cache metrics vs artifact metrics
        # -----------------------------------------------------------------
        # User expectation (explicit requirement):
        # - The dashboard history must match the notebook metrics.
        #
        # If this check fails, the #1 reason is: the market dataset changed
        # after the split artifact was exported (new rows added, prices differ, etc).
        try:
            import joblib

            artifact = joblib.load(SPLIT_MODEL)
            expected_cfg = artifact.get("config", {}) if isinstance(artifact, dict) else {}
            exp_r_min = expected_cfg.get("r_min")
            exp_mcc = expected_cfg.get("mcc")

            if exp_r_min is not None and exp_mcc is not None and "r_min" in stats and "mcc" in stats:
                dr = abs(float(exp_r_min) - float(stats["r_min"]))
                dm = abs(float(exp_mcc) - float(stats["mcc"]))
                print(f"\n🔎 Notebook (artifact) metrics: R_MIN={float(exp_r_min):.3f} MCC={float(exp_mcc):.3f}")
                print(f"   Delta vs cache (TEST):       |dR_MIN|={dr:.3f} |dMCC|={dm:.3f}")

                # Loose threshold: enough to catch major mismatches without being too noisy.
                if dr > 0.03 or dm > 0.03:
                    print("⚠️ WARNING: Cache metrics do not match artifact metrics.")
                    print("   Common reasons:")
                    print("   - DB market data has new rows after the model export (split boundaries shifted)")
                    print("   - Local parquet fallback differs from the DB data used in the notebook")
                    print("   - Different start/end date than the notebook used")
        except Exception as _e:
            # Cache generation must not fail just because of a debug check.
            pass


if __name__ == "__main__":
    main()
