# Build a container that refreshes the dashboard daily and serves it via HTTP
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV START_DATE=2015-01-01 \
    WARMUP_DAYS=252 \
    STEP_DAYS=5 \
    REFRESH_INTERVAL=86400 \
    FORCE_REFRESH=1 \
    PORT=8080

EXPOSE 8080

CMD ["python", "scripts/service.py"]
