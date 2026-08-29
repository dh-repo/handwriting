# ===========================================================================
# Stage 1: Builder
# ===========================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy package definitions and install wheels
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# ===========================================================================
# Stage 2: Runtime Runner
# ===========================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install minimal runtime shared libraries (OpenCV headless & curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEVICE=cpu
ENV USE_MOCK_ENGINE=false

# Create unprivileged application user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy application source code
COPY --chown=appuser:appgroup app /app/app

# Switch to unprivileged user
USER appuser

EXPOSE 8000

# Healthcheck definition
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/v1/health || exit 1

# Launch FastAPI ASGI server via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
