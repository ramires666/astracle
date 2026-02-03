# =============================================================================
# Bitcoin Astro Predictor - Production Dockerfile
# =============================================================================
#
# Multi-stage build for minimal production image.
# Includes ephemeris data and trained model artifacts.
#
# Build:   docker build -t btc-astro-predictor ./production_dev
# Run:     docker run -p 9742:9742 btc-astro-predictor
# =============================================================================

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=9742

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =============================================================================
# Dependencies Stage
# =============================================================================

FROM base AS dependencies

# Copy only requirements first for better caching
COPY production_dev/requirements.txt ./production_dev/

# Install Python dependencies
RUN pip install --no-cache-dir -r production_dev/requirements.txt

# =============================================================================
# Final Stage
# =============================================================================

FROM dependencies AS final

# Copy project source code
COPY src/ ./src/
COPY RESEARCH/ ./RESEARCH/
COPY configs/ ./configs/
COPY production_dev/ ./production_dev/

# Copy model artifacts (if they exist)
COPY models_artifacts/ ./models_artifacts/

# Copy ephemeris data (if available)
COPY data/ephe/ ./data/ephe/ 2>/dev/null || true

# Create non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/api/health')" || exit 1

# Run the application
CMD ["uvicorn", "production_dev.main:app", "--host", "0.0.0.0", "--port", "9742"]
