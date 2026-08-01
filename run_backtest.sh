#!/bin/bash
# Local backtest + dashboard + deploy.
#
# This mirrors .github/workflows/daily-deploy.yml, which is the canonical path.
# Both now publish `reports/` — this script previously staged into
# /tmp/raaal-deploy while CI deployed `reports/`, so the two could ship
# different artifacts from the same commit.
set -euo pipefail

cd /projects/RAAAL
source .venv/bin/activate

echo "Running tests at $(date)"
python -m pytest tests/ -q

echo "Starting backtest at $(date)"
python -m src.history --start 2016-01-01 --end 2025-11-20 --refresh

echo "Backtest complete at $(date)"
echo "Run manifest:"
cat data/history/run_manifest.json

echo "Rebuilding dashboard..."
python -m src.visualization.bokeh_app

# Cloudflare Pages serves index.html; the dashboard is generated under its own
# name so both live in the published directory.
cp reports/regime_dashboard.html reports/index.html

echo "Deploying to Cloudflare..."
CLOUDFLARE_EMAIL="$CLOUDFLARE_EMAIL" CLOUDFLARE_API_KEY="$GLOBAL_API_TOKEN" \
  npx wrangler pages deploy reports --project-name raaal-dashboard --commit-dirty=true

echo "All done at $(date)"
