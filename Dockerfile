# The production image. It serves the pilot API and nothing else.
#
# This ran `scripts/service.py` — the Bokeh dashboard — until Gate 3, while
# nothing anywhere served `src/api.py`. The pilot application, its gated
# routers and its entire startup preflight had no deployment entrypoint: every
# control was on a path the container never took. The dashboard now lives in
# `Dockerfile.dashboard` and is a development surface.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# The API surface does not need the ML/RL stack.
COPY requirements-core.txt ./
RUN pip install --no-cache-dir -r requirements-core.txt

COPY . .

RUN mkdir -p data

# Deliberately unset. `QUANTIFY_DATABASE_URL` has no production default — the
# preflight refuses rather than falling back to a local SQLite file, which
# would be a live path reading a database nobody authorised. The build stamps
# are supplied at deploy time and the preflight refuses without them.
ENV QUANTIFY_DEPLOYMENT_PROFILE=production \
    PORT=8000

EXPOSE 8000

# Readiness, not liveness. A process answering on a port is not evidence that
# it has a migrated database, a current schema or an observable build.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health/ready || exit 1

# The factory, not the module-level app. `create_app` runs the deployment
# preflight while the process is still starting, so a refusal prevents the
# server binding at all rather than being noticed by a lifespan hook after the
# socket is already open.
CMD ["uvicorn", "src.api:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]
