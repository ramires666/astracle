"""
Bitcoin Astro Predictor - Production Service

This package contains the production-ready web service for
Bitcoin price direction predictions using astrological analysis.

Run with:
    uvicorn production_dev.main:app --host 0.0.0.0 --port 9742

Or with Docker:
    docker-compose -f production_dev/docker-compose.yml up --build
"""

__version__ = "1.0.0"
__author__ = "Ostrofun Team"
