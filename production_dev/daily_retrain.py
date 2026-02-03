#!/usr/bin/env python3
"""
Daily Model Retraining Script

This script is designed to run daily in Docker to:
1. Download new price data
2. Retrain the FULL model on all historical data
3. Regenerate the prediction cache

Usage:
    python -m production_dev.daily_retrain
    
Cron Example (run at 6 AM UTC):
    0 6 * * * cd /app && python -m production_dev.daily_retrain >> /var/log/retrain.log 2>&1
"""

import sys
from pathlib import Path
from datetime import date, datetime
import subprocess
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def log(message: str):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def update_price_data() -> bool:
    """
    Download latest price data.
    
    Returns True if successful.
    """
    log("📊 Updating price data...")
    try:
        # Import and run price updater
        from RESEARCH.data_loader import update_btc_prices
        new_rows = update_btc_prices()
        log(f"   Added {new_rows} new price records")
        return True
    except ImportError:
        log("⚠️ Price updater not found, skipping data update")
        return True  # Continue anyway
    except Exception as e:
        log(f"❌ Price update failed: {e}")
        return False


def retrain_full_model() -> bool:
    """
    Retrain the model on all available data.
    
    Returns True if successful.
    """
    log("🧠 Retraining FULL model on all data...")
    try:
        from production_dev.train_full_model import train_final_model
        train_final_model()
        log("✅ Model retrained successfully")
        return True
    except Exception as e:
        log(f"❌ Retraining failed: {e}")
        traceback.print_exc()
        return False


def regenerate_cache() -> bool:
    """
    Regenerate prediction cache with new model.
    
    Returns True if successful.
    """
    log("📦 Regenerating prediction cache...")
    try:
        from production_dev.generate_cache import main as generate_cache_main
        generate_cache_main()
        log("✅ Cache regenerated")
        return True
    except Exception as e:
        log(f"❌ Cache regeneration failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main daily retraining workflow."""
    log("=" * 60)
    log("🔄 DAILY MODEL RETRAINING")
    log("=" * 60)
    log(f"Date: {date.today()}")
    
    success = True
    
    # Step 1: Update price data
    if not update_price_data():
        log("⚠️ Continuing despite price update failure")
    
    # Step 2: Retrain full model
    if not retrain_full_model():
        success = False
        log("❌ RETRAINING FAILED - keeping old model")
    
    # Step 3: Regenerate cache (uses both split and full models)
    if not regenerate_cache():
        success = False
        log("❌ CACHE REGENERATION FAILED")
    
    # Summary
    log("")
    log("=" * 60)
    if success:
        log("✅ DAILY RETRAINING COMPLETE")
    else:
        log("⚠️ DAILY RETRAINING COMPLETED WITH ERRORS")
    log("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
