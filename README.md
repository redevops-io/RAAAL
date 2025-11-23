# RAAAL: Regime-Adjusted Asset Allocation with AI/ML Enhancements

This prototype follows the MVP brief from `Regime-adjusted asset allocation model.odt` and builds on the multi-asset modelling principles laid out by Vuletic (2025), enhanced with AI/ML techniques from the CFA Institute "AI in Asset Management" monograph (2025).

## Academic References

- **Vuletic, M. (2025).** Multi-asset portfolio modeling and regime detection.
- **CFA Institute Research Foundation. (2025).** *AI in Asset Management: Tools, Applications, and Frontiers.* 
  Editor: Joseph Simonian, PhD. 
  Available at: https://rpc.cfainstitute.org/sites/default/files/docs/research-reports/rf_aiinassetmanagement_full-monograph_online.pdf
  
  Key chapters implemented:
  - Chapter 1: Unsupervised Learning (Hierarchical Clustering for HRP)
  - Chapter 2: Network Theory (Correlation Networks & Centrality Measures)
  - Chapter 5: Ensemble Learning (Random Forest & Gradient Boosting)
  
- **López de Prado, M. (2016).** "Building Diversified Portfolios That Outperform Out of Sample." 
  *Journal of Portfolio Management*, 42(4), 59-69.

## Features

### Core Portfolio Engine
- **ETF universe (9 tickers)**: SPY/SH (equity long/short), TLT/TBT (treasury long/short), LQD (investment grade credit), DBC (commodities), GLD (gold), HYG (high yield), BIL (cash proxy). Auxiliary series: TIP, VIX for signals.
- **Regime detection**: rule-based classifier (Risk-On, Risk-Off, Inflation Shock) using SPY trend, VIX level, credit-spread proxy, SPY–TLT correlation, commodity/TIP momentum.
- **Portfolio engine**: Sharpe-maximizing optimizer with per-ETF bounds, cash floors, inverse caps, leverage guardrails, and turnover penalty. Falls back to risk-parity if the solver fails.
- **Rebalancing signal**: change of regime triggers rebalance flag (state stored in `reports/state.json`).
- **Reporting**: console table + CSV/JSON snapshots under `reports/` for dashboard ingestion.

### AI/ML Enhancements (NEW)
- **Hierarchical Risk Parity (HRP)**: Academic portfolio construction using correlation-based clustering (López de Prado 2016). Provides regime-agnostic diversification benchmark.
- **Network Analysis**: Correlation networks with centrality measures (degree, betweenness, eigenvector) to identify systemic risk assets. Includes community detection per regime.
- **Ensemble Learning**: Random Forest + Gradient Boosting classifiers for regime prediction. Trained on historical features with accuracy metrics and feature importance analysis.
- **Multi-tab Dashboard**: Interactive Bokeh visualization with:
   - **Main Dashboard**: Original regime timeline, allocation weights, VIX tracking
   - **Advanced Analysis**: HRP vs RAAAL comparison, ensemble regime predictions, network visualizations, feature importance charts
   - **Strategy Lab**: Multi-strategy growth comparison (momentum, relative-value, risk-based, factor sleeves) with rule-based, ML, or neutral regime inputs, plus a multi-select growth plot and buy/sell pressure bar chart for the chosen strategies.

### Strategy experimentation (NEW)
- **Strategy test harness**: `src/strategies.py` exposes a `StrategySuite` that can score additional trading approaches (momentum, relative-value/mean-reversion, risk-based, and factor portfolios) using rule-based regimes, ML regimes, or no regime overlay.
- **Regime flexibility**: Toggle `detection_mode` between `"rule_based"`, `"ml"`, or `"none"` to combine each strategy with rule-based signals, ensemble predictions, or neutral baselines.
- **Pytest coverage**: `tests/test_strategies.py` exercises all strategy families, including ML-backed regime detection, to guard against regressions.
- **Quick usage**:
   ```python
   from src.strategies import StrategySuite
   suite = StrategySuite()
   results = suite.evaluate(prices, returns, detection_mode="rule_based")
   print(results["dual_momentum"].metrics)
   ```
- **Strategy Lab dashboard tab**: after running `python -m src.history ...` the new Tab 4 visualizes normalized growth curves plus summary table. The "Visible strategies" multi-select now defaults to the union of the top Sharpe and top total-return strategies (≈6–10 series) so the stacked signal lanes and composites stay readable; you can still Ctrl/Cmd + click to toggle more. The diagnostics include per-strategy bar lanes (one row per strategy) driven by net cash redeployments so only one buy/sell marker fires per rebalance, and clicking any bar opens a ticker showing that rebalance's turnover, cash, and top-three holdings. A composite recommendation panel aggregates equal-weight signals from the top 5 Sharpe and top 5 total-return strategies, complete with normalized portfolio compositions sourced from each strategy's most recent rebalance snapshot. Use `python -m src.visualization.bokeh_app --timeline data/history/timeline.parquet --weights data/history/weights.parquet --prices data/history/prices.parquet --output reports/regime_dashboard.html` to regenerate manually.

### Performance Results (Annualized, 2016-2025)
- Standard (Restricted): **5.87%**
- Standard (Unrestricted): **8.10%**
- Regime (Restricted): **6.33%**
- Regime (Unrestricted): **13.00%** ⭐
- **HRP: 2.03%** (diversification-focused, lower volatility)

## Project layout
```
├── main.py                      # CLI entrypoint
├── src/
│   ├── config.py                # Universe + constraints
│   ├── data_loader.py           # yfinance downloader with caching
│   ├── features.py              # Rolling stats & signals
│   ├── regime.py                # Rule-based regime classifier
│   ├── optimizer.py             # Sharpe-maximizing weights
│   ├── hrp.py                   # Hierarchical Risk Parity optimizer (NEW)
│   ├── network.py               # Network analysis & centrality measures (NEW)
│   ├── ensemble_regime.py       # ML ensemble regime classifier (NEW)
│   ├── pipeline.py              # Orchestration & state mgmt
│   ├── portfolio_utils.py       # Shared RF/metrics helpers
│   ├── history.py               # Historical backtest (now includes HRP)
│   └── visualization/
│       ├── bokeh_app.py         # Multi-tab dashboard
│       └── advanced_analysis.py # Advanced ML visualizations (NEW)
├── docs/mvp_plan.md             # Design plan & references
├── requirements.txt
├── rf_aiinassetmanagement_full-monograph_online.pdf  # CFA Institute reference
└── tests/
    ├── test_regime.py
    ├── test_optimizer.py
    └── test_advanced_features.py  # Tests for HRP, network, ensemble (NEW)
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

### Local preview via Docker

To test locally with the new Strategy Lab tab enabled:

```bash
docker build -t regime-dashboard .
docker run --rm -it \
   -e START_DATE=2018-01-01 \
   -e STEP_DAYS=5 \
   -e REFRESH_INTERVAL=0 \
   -p 8000:8000 \
   regime-dashboard
```

Then open `http://localhost:8000` and navigate to the **Strategy Lab** tab to compare all strategy families side-by-side.

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
