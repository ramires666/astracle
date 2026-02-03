"""
Pydantic Schemas for Bitcoin Astro Prediction API

Defines request/response models for the API endpoints.
All models include detailed field descriptions for OpenAPI documentation.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """
    Single day prediction result.
    """
    date: str = Field(
        ...,
        description="Prediction date in ISO format (YYYY-MM-DD)",
        example="2024-02-04"
    )
    direction: str = Field(
        ...,
        description="Predicted price direction: UP or DOWN",
        example="UP"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the prediction (0.0 to 1.0)",
        example=0.65
    )
    simulated_price: float = Field(
        ...,
        ge=0.0,
        description="Simulated BTC price based on random walk with prediction drift",
        example=105234.56
    )


class ForecastSummary(BaseModel):
    """
    Summary statistics for the forecast period.
    """
    total_days: int = Field(..., description="Total number of days in forecast")
    up_predictions: int = Field(..., description="Number of UP predictions")
    down_predictions: int = Field(..., description="Number of DOWN predictions")
    up_ratio: float = Field(..., description="Ratio of UP predictions (0.0 to 1.0)")
    average_confidence: float = Field(..., description="Average prediction confidence")
    start_price: float = Field(..., description="Starting simulated price")
    end_price: float = Field(..., description="Ending simulated price")


class ForecastResponse(BaseModel):
    """
    Complete forecast response with predictions and metadata.
    """
    predictions: List[PredictionItem] = Field(
        ...,
        description="List of daily predictions"
    )
    summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics for the forecast"
    )
    model_info: Dict[str, Any] = Field(
        ...,
        description="Information about the prediction model"
    )
    generated_at: str = Field(
        ...,
        description="Timestamp when forecast was generated (UTC ISO format)"
    )


class HealthResponse(BaseModel):
    """
    Health check response.
    """
    status: str = Field(
        ...,
        description="Service status: healthy or unhealthy",
        example="healthy"
    )
    timestamp: str = Field(
        ...,
        description="Current server timestamp (UTC ISO format)"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether prediction model is loaded and ready"
    )
    natal_date: Optional[str] = Field(
        None,
        description="Bitcoin natal chart date used by model"
    )
    expected_accuracy: Optional[float] = Field(
        None,
        description="Expected model accuracy (R_MIN metric)"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if service is unhealthy"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response format.
    """
    detail: str = Field(
        ...,
        description="Error message describing what went wrong"
    )
    error_code: Optional[str] = Field(
        None,
        description="Machine-readable error code"
    )
