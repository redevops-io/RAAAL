#!/bin/bash
# Run backtest with regime-aware HRP

cd /projects/RAAAL
source .venv/bin/activate

echo "Starting backtest at $(date)"
python -m src.history --start 2016-01-01 --end 2025-11-20 --refresh

echo "Backtest complete at $(date)"
echo "Rebuilding dashboard..."
python -m src.visualization.bokeh_app

echo "Deploying to Cloudflare..."
mkdir -p /tmp/raaal-deploy
cp reports/regime_dashboard.html /tmp/raaal-deploy/index.html
CLOUDFLARE_EMAIL="$CLOUDFLARE_EMAIL" CLOUDFLARE_API_KEY="$GLOBAL_API_TOKEN" npx wrangler pages deploy /tmp/raaal-deploy --project-name raaal-dashboard --commit-dirty=true

echo "All done at $(date)"
