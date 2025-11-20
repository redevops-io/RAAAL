# Regime-adjusted Asset Allocation MVP

This prototype follows the MVP brief from `Regime-adjusted asset allocation model.odt` and builds on the multi-asset modelling principles laid out by Vuletic (2025), focusing on clean ETF exposures, explainable regime logic, and fast constrained optimization.

## Features
- **ETF universe (9 tickers)**: SPY/SH (equity long/short), TLT/TBT (treasury long/short), LQD (investment grade credit), DBC (commodities), GLD (gold), HYG (high yield), BIL (cash proxy). Auxiliary series: TIP, VIX for signals.
- **Regime detection**: rule-based classifier (Risk-On, Risk-Off, Inflation Shock) using SPY trend, VIX level, credit-spread proxy, SPY–TLT correlation, commodity/TIP momentum.
- **Portfolio engine**: Sharpe-maximizing optimizer with per-ETF bounds, cash floors, inverse caps, leverage guardrails, and turnover penalty. Falls back to risk-parity if the solver fails.
- **Rebalancing signal**: change of regime triggers rebalance flag (state stored in `reports/state.json`).
- **Reporting**: console table + CSV/JSON snapshots under `reports/` for dashboard ingestion.

## Project layout
```
├── main.py                # CLI entrypoint
├── src/
│   ├── config.py          # Universe + constraints
│   ├── data_loader.py     # yfinance downloader with caching
│   ├── features.py        # Rolling stats & signals
│   ├── regime.py          # Regime classifier
│   ├── optimizer.py       # Sharpe-maximizing weights
│   ├── pipeline.py        # Orchestration & state mgmt
│   ├── portfolio_utils.py # Shared RF/metrics helpers
│   ├── history.py         # Historical backtest + storage
│   └── visualization/
│       └── bokeh_app.py   # Animated SPY/regime/weights dashboard
├── docs/mvp_plan.md       # Design plan & references
├── requirements.txt
└── tests/
    ├── test_regime.py
    └── test_optimizer.py
```

## Quick start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline (example window):
   ```bash
   python main.py --start 2015-01-01 --end 2025-11-15
   ```
3. Inspect outputs in `reports/` and the console summary.

### Historical analysis + Bokeh dashboard
1. Generate the historical regime/portfolio dataset (weekly steps, adjustable):
   ```bash
   python -m src.history --start 2018-01-01 --end 2025-11-15 --step 5
   ```
   Artifacts land in `data/history/` (`timeline.parquet`, `weights.parquet`, JSON summary).
2. Build the interactive dashboard (saves to `reports/regime_dashboard.html` by default):
   ```bash
    python -m src.visualization.bokeh_app --output reports/regime_dashboard.html
   ```
   The page shows SPY with regime shading, stacked allocations, and an animated slider/playback that highlights each regime change.

## Dockerized daily refresh service
Build and run the container (serves the static Bokeh page on port 8000 by default):
```bash
docker build -t regime-dashboard .
docker run -d \
   -e START_DATE=2015-01-01 \
   -e STEP_DAYS=5 \
   -e REFRESH_INTERVAL=86400 \
   -e PORT=8000 \
   -p 80:8000 \
   --name regime-dashboard \
   regime-dashboard
```
Environment knobs:
- `START_DATE` – earliest history to pull (ISO date).
- `WARMUP_DAYS` – lookback required before first optimization.
- `STEP_DAYS` – evaluation stride (e.g., 5 = weekly).
- `REFRESH_INTERVAL` – seconds between rebuilds (set to `0` to disable loop and rebuild only once at start).
- `FORCE_REFRESH` – `1` to force yfinance downloads each rebuild.
- `PORT` – HTTP port inside the container.

The container runs `scripts/service.py`, which:
1. Rebuilds the history + Bokeh HTML (`reports/regime_dashboard.html`).
2. Copies it to `/app/site/index.html`.
3. Repeats every `REFRESH_INTERVAL` seconds.
4. Serves `/app/site` over HTTP so the Bokeh HTML is the only page.

### Deploying to Cloudflare Pages
For static hosting of the dashboard HTML:

1. **Build the dashboard:**
   ```bash
   python -m src.history --start 2015-01-01 --end $(date +%Y-%m-%d) --step 5
   python -m src.visualization.bokeh_app --output reports/regime_dashboard.html
   ```

2. **Deploy to Cloudflare Pages:**
   ```bash
   # Using wrangler (requires CLOUDFLARE_API_KEY and CLOUDFLARE_EMAIL)
   CLOUDFLARE_API_KEY=$GLOBAL_API_TOKEN npx wrangler pages deploy reports --project-name raaal-dashboard
   
   # Or using the deployment script
   export DOMAIN="yourdomain.com"  # Set your domain
   ./deploy_cloudflare.sh
   ```
   
3. **DNS Configuration:**
   - Add a `CNAME` record: `@` → `raaal-dashboard.pages.dev`
   - Or add `www` subdomain: `www` → `raaal-dashboard.pages.dev`
   - Cloudflare will automatically provision SSL/TLS certificates

4. **Automated daily updates (GitHub Actions):**
   - Set `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, and `CLOUDFLARE_EMAIL` as repository secrets
   - Configure workflow to rebuild history + dashboard and push to Pages on schedule

Alternative: Deploy via Docker on VM with A record pointing to server IP.

## Tests
```bash
pytest
```

## References
- Milena Vuletic (2025), *Multi-asset financial markets: mathematical modelling and data-driven approaches* (Oxford DPhil thesis).
- Internal brief `Regime-adjusted asset allocation model.odt`.
