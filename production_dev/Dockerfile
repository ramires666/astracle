# =============================================================================
# Bitcoin Astro Predictor - Production Dockerfile
# =============================================================================
#
# Build from repo root:
#   docker build -f production_dev/Dockerfile -t ostrofun-backend:latest .
# =============================================================================

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=9742 \
    VENV_PATH=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin

WORKDIR /app


# -----------------------------------------------------------------------------
# Dependencies build stage
# -----------------------------------------------------------------------------
FROM base AS builder

# Build tools are kept only in this stage.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv ${VENV_PATH}

COPY production_dev/requirements.txt ./production_dev/
RUN ${VENV_PATH}/bin/pip install --no-cache-dir --upgrade pip && \
    ${VENV_PATH}/bin/pip install --no-cache-dir -r production_dev/requirements.txt


# -----------------------------------------------------------------------------
# Final runtime image
# -----------------------------------------------------------------------------
FROM base AS final

# Python deps from builder stage (no build toolchain in final image).
COPY --from=builder ${VENV_PATH} ${VENV_PATH}

# Runtime source
COPY --chown=1000:1000 src/ ./src/
COPY --chown=1000:1000 RESEARCH/ ./RESEARCH/
COPY --chown=1000:1000 RESEARCH2/ ./RESEARCH2/
COPY --chown=1000:1000 configs/ ./configs/
COPY --chown=1000:1000 production_dev/ ./production_dev/
COPY --chown=1000:1000 models_artifacts/ ./models_artifacts/

# Keep only bootstrap snapshots in image.
COPY --chown=1000:1000 data/market/ ./bootstrap/data/market/
COPY --chown=1000:1000 data/prediction_cache/ ./bootstrap/data/prediction_cache/

# Runtime writable dirs (can be bind-mounted on deploy).
RUN mkdir -p /app/data/market /app/data/prediction_cache /app/RESEARCH/cache && \
    chown -R 1000:1000 /app/data /app/RESEARCH/cache

USER 1000:1000

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/api/health')" || exit 1

CMD ["uvicorn", "production_dev.main:app", "--host", "0.0.0.0", "--port", "9742"]
