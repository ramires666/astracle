"""
Bitcoin Astro Prediction API

FastAPI application providing Bitcoin price direction predictions based on
astrological analysis. Uses XGBoost model trained on natal chart transits.

Endpoints:
- GET /           → Serves web UI
- GET /api/predict → Returns 90-day forecast JSON
- GET /api/health  → Health check

Port: 9742 (non-standard)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from production_dev.predictor import BtcAstroPredictor
from production_dev.schemas import ForecastResponse, HealthResponse, PredictionItem

# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================

APP_CONFIG = {
    "title": "Bitcoin Astro Predictor",
    "description": "AI-powered Bitcoin price direction predictions using astrological analysis",
    "version": "1.0.0",
    "port": 9742,  # Non-standard port
}

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title=APP_CONFIG["title"],
    description=APP_CONFIG["description"],
    version=APP_CONFIG["version"],
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for web UI
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# =============================================================================
# GLOBAL STATE
# =============================================================================

# ---------------------------------------------------------------------------
# Dual-model setup (IMPORTANT FOR FRONTEND CORRECTNESS)
# ---------------------------------------------------------------------------
#
# The UI has two "sides":
# 1) History / Backtest
#    - Must be HONEST (out-of-sample) and match notebook metrics
#    - Uses the SPLIT model artifact produced by `RESEARCH/birthdate_deep_search.ipynb`
#
# 2) Forecast / Future Predictions
#    - Should use ALL available data for best future performance
#    - Uses the FULL model artifact (retrained periodically)
#
# If we accidentally use the full model for history, the history will look
# "too good" (data leakage). If we accidentally use the split model for
# forecast, the forecast will be weaker than it could be.
#
# So we keep TWO predictor instances and route each endpoint to the right one.
SPLIT_MODEL_PATH = PROJECT_ROOT / "models_artifacts" / "btc_astro_predictor.joblib"
FULL_MODEL_PATH = PROJECT_ROOT / "models_artifacts" / "btc_astro_predictor.full.joblib"

_split_predictor: Optional[BtcAstroPredictor] = None
_full_predictor: Optional[BtcAstroPredictor] = None


def get_split_predictor() -> BtcAstroPredictor:
    """
    Get or initialize the SPLIT model predictor (honest backtest).

    This model must NEVER be retrained on all data, otherwise the backtest
    becomes dishonest. It should only be updated by explicitly running the
    research notebook that exports `btc_astro_predictor.joblib`.
    """
    global _split_predictor
    if _split_predictor is None:
        _split_predictor = BtcAstroPredictor(model_path=SPLIT_MODEL_PATH)
        if not _split_predictor.load_model():
            raise RuntimeError("Failed to load SPLIT prediction model")
    return _split_predictor


def get_full_predictor() -> BtcAstroPredictor:
    """
    Get or initialize the FULL model predictor (best forecast).

    This model is allowed to be retrained on ALL available historical data,
    because future dates are always unseen anyway.

    Fallback logic:
    - If the full model file is missing, we fall back to the split model,
      because the service should still work (but forecasting is weaker).
    """
    global _full_predictor
    if _full_predictor is None:
        if FULL_MODEL_PATH.exists():
            _full_predictor = BtcAstroPredictor(model_path=FULL_MODEL_PATH)
            if not _full_predictor.load_model():
                raise RuntimeError("Failed to load FULL prediction model")
        else:
            # Fallback: keep service alive even if full model isn't trained yet.
            _full_predictor = get_split_predictor()
    return _full_predictor


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    """
    Serve the main web interface.
    """
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(str(index_path))


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and model information.
    """
    try:
        # Health is reported for the split model (the one with honest metrics).
        predictor = get_split_predictor()
        model_info = predictor.get_model_info()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=model_info["is_loaded"],
            natal_date=model_info.get("natal_date", "unknown"),
            expected_accuracy=model_info.get("expected_r_min", 0.0),
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=False,
            error=str(e),
        )


@app.get("/api/predict", response_model=ForecastResponse)
async def get_predictions(
    days: int = 90,
    seed: Optional[int] = None,
):
    """
    Generate Bitcoin price direction predictions.
    
    Args:
        days: Number of days to predict (1-365, default: 90)
        seed: Random seed for reproducible price simulation
        
    Returns:
        ForecastResponse with predictions and simulated prices
    """
    # Validate input
    if days < 1 or days > 365:
        raise HTTPException(
            status_code=400,
            detail="Days must be between 1 and 365"
        )
    
    try:
        # Forecast endpoint must use FULL model (best possible future predictions).
        predictor = get_full_predictor()
        
        # Generate predictions
        predictions = predictor.predict_next_n_days(days)
        
        # Add simulated price path
        predictions = predictor.generate_price_path(predictions, seed=seed)
        
        # Calculate summary statistics
        up_count = sum(1 for p in predictions if p["direction"] == "UP")
        down_count = len(predictions) - up_count
        avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions)
        
        # Convert to response model (ensure native Python types for Pydantic)
        prediction_items = [
            PredictionItem(
                date=p["date"],
                direction=p["direction"],
                confidence=float(p["confidence"]),  # Convert from numpy
                simulated_price=float(p.get("simulated_price", 0.0)),  # Convert from numpy
            )
            for p in predictions
        ]
        
        # Ensure summary values are native Python types
        start_price = float(predictions[0].get("simulated_price", 0.0)) if predictions else 0.0
        end_price = float(predictions[-1].get("simulated_price", 0.0)) if predictions else 0.0
        
        return ForecastResponse(
            predictions=prediction_items,
            summary={
                "total_days": len(predictions),
                "up_predictions": up_count,
                "down_predictions": down_count,
                "up_ratio": round(up_count / len(predictions), 3),
                "average_confidence": round(float(avg_confidence), 3),
                "start_price": start_price,
                "end_price": end_price,
            },
            model_info=predictor.get_model_info(),
            generated_at=datetime.utcnow().isoformat(),
        )
        
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/api/config")
async def get_config():
    """
    Get current model configuration.
    """
    try:
        # UI displays split-model config, because its metrics are honest.
        predictor = get_split_predictor()
        return JSONResponse(content=predictor.get_model_info())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/historical")
async def get_historical_data(days: int = 30):
    """
    Get historical BTC price data from project database.
    
    Uses the same data loading functions that were used for
    initial data collection (RESEARCH.data_loader).
    
    Args:
        days: Number of historical days to load (1-1500, default: 30)
    """
    if days < 1 or days > 1500:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 1500")
    
    try:
        from production_dev.data_service import load_historical_prices, get_data_summary
        
        prices = load_historical_prices(days=days)
        summary = get_data_summary()
        
        return JSONResponse(content={
            "prices": prices,
            "summary": summary,
            "count": len(prices),
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data loading error: {str(e)}")


@app.get("/api/predictions/full")
async def get_full_cached_predictions():
    """
    Get all cached predictions (backtest + forecast).
    
    Returns backtest predictions with actual price accuracy,
    and future forecast predictions with simulated prices.
    
    This endpoint uses pre-calculated cached data for fast loading.
    Run `python -m production_dev.generate_cache` to update the cache.
    """
    try:
        from production_dev.cache_service import get_full_predictions
        
        data = get_full_predictions()
        
        # Convert numpy types to native Python for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        return JSONResponse(content=convert_types(data))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache loading error: {str(e)}")


# =============================================================================
# STARTUP / SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Application startup handler.
    Pre-loads the model and prediction cache for instant responses.
    """
    print(f"🚀 Starting {APP_CONFIG['title']} v{APP_CONFIG['version']}")
    print(f"📡 API available at http://localhost:{APP_CONFIG['port']}")
    print(f"📊 Web UI available at http://localhost:{APP_CONFIG['port']}/")
    
    # Pre-load model
    try:
        get_split_predictor()
        print("✅ Split model loaded successfully")
    except Exception as e:
        print(f"⚠️ Split model not loaded: {e}")
        print("   Split model will be loaded on first request that needs it")

    # Pre-load full model (optional)
    try:
        get_full_predictor()
        if FULL_MODEL_PATH.exists():
            print("✅ Full model loaded successfully")
        else:
            print("⚠️ Full model file not found; forecast will use split model fallback")
            print(f"   Expected path: {FULL_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Full model not loaded: {e}")
        print("   Forecast will use split model fallback until full model is fixed")
    
    # Pre-load prediction cache into memory (for instant /api/predictions/full responses)
    try:
        from production_dev.cache_service import init_memory_cache
        if init_memory_cache():
            print("✅ Prediction cache loaded into memory")
        else:
            print("⚠️ Prediction cache not available. Run: python -m production_dev.generate_cache")
    except Exception as e:
        print(f"⚠️ Cache not loaded: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown handler.
    """
    print("👋 Shutting down Bitcoin Astro Predictor")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=APP_CONFIG["port"],
        reload=True,  # Enable hot reload for development
        log_level="info",
    )
