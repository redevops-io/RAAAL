# Build a container that refreshes the dashboard daily and serves it via HTTP.
# Uses requirements-core.txt (no torch/transformers/RL) to keep the image lean.
# The ML workstream packages are optional and used only when installed.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install all dependencies including ML/RL stack
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure reports and data directories exist
RUN mkdir -p reports data/history data/models site

ENV START_DATE=2015-01-01 \
    WARMUP_DAYS=252 \
    STEP_DAYS=5 \
    REFRESH_INTERVAL=86400 \
    FORCE_REFRESH=1 \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

CMD ["python", "scripts/service.py"]
