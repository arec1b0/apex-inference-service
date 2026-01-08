# Build Stage
FROM python:3.11-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install dependencies first (caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Runtime Stage
FROM python:3.11-slim-bookworm

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY src /app/src
COPY pyproject.toml /app/
# --- NEW LINE HERE ---
COPY model_store /app/model_store

# Set environment path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Runtime configuration
ENV PORT=8000
EXPOSE 8000

USER appuser

# Use uvicorn directly or via a shell script
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]