# ===========================================================================
# Stage 1: Builder
# ===========================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY README.md .
COPY backend ./backend
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir ./backend

# ===========================================================================
# Stage 2: Runtime Runner
# ===========================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Install runtime shared libraries (OpenCV headless & curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEVICE=cpu
ENV USE_MOCK_ENGINE=false
ENV VOCAB_DIR="/app/data/reference_handwriting/vocabularies"

# Create unprivileged application user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /bin/bash -m appuser

# Copy codebases and vocabularies from repository root
COPY --chown=appuser:appgroup backend /app/backend
COPY --chown=appuser:appgroup pipeline /app/pipeline
COPY --chown=appuser:appgroup data/reference_handwriting/vocabularies /app/data/reference_handwriting/vocabularies

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
