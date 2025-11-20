# Regime-adjusted Asset Allocation MVP Plan

## Inputs & guiding references
- **Vuletic (2025)** highlights the need for multi-asset models that capture co-movements, adapt to regime shifts, and remain arbitrage-aware (Ch. 2–3). We leverage these ideas by focusing on a modular pipeline with explainable statistical blocks and stress-tested heuristics.
- **Product brief (`Regime-adjusted asset allocation model.odt`)** specifies a liquid ETF universe (SPY/SH, TLT/TBT, LQD, BIL cash proxy, DBC, GLD, HYG) plus a three-regime playbook (Risk-On, Risk-Off, Inflation Shock) with periodic rebalancing and constrained optimization (max position sizes, cash floors, inverse caps, leverage cap \<= 1.25x).

## System overview
1. **Data ingestion**
   - Pull daily bars for ETFs + proxies (`SPY`, `SH`, `TLT`, `TBT`, `LQD`, `BIL`, `DBC`, `HYG`, `GLD`, `TIP`, `^VIX`).
   - Cache locally (parquet) to avoid repeated downloads.
2. **Feature engineering**
   - Compute log returns, rolling volatility/Sharpe (21/63/252d lookbacks), beta to `SPY`, rolling correlations (`SPY`/`TLT`), credit-spread proxy (`LQD` minus `TLT` short-term momentum), commodity/TIP momentum.
3. **Regime detection**
   - Risk-On: `SPY` price > 200d SMA, `^VIX` < 18, credit spread narrowing, `corr(SPY, TLT)` \< 0.2.
   - Risk-Off: `SPY` < 200d SMA, `^VIX` > 22, credit spread widening, `corr(SPY, TLT)` \< -0.3.
   - Inflation Shock: `corr(SPY, TLT)` > 0, commodity momentum > 0, TIP momentum < 0.
   - Otherwise default to the closest matching regime (priority: risk-off > inflation > risk-on).
4. **Portfolio optimizer**
   - Target: maximize Sharpe = `(μᵀw - r_f) / sqrt(wᵀΣw)` using exponential-weighted returns/covariance (63d half-life) with `BIL` yield as `r_f`.
   - Two variants:
     - **Restricted**: Regime-specific guardrails (cash floors, inverse caps, turnover penalty)
     - **Unrestricted**: Long-only 0-100% bounds with budget constraint only
   - Constraints (restricted version):
     - Sum of weights = 1.
     - Per-ETF bounds (from brief; e.g., `SPY` \<= 0.5, `TLT` \<= 0.5, inverse ETFs \<= 0.2, `DBC` \<= 0.15, cash between regime-specific limits).
     - Optional leverage cap via `||w||₁ \<= 1.25`.
     - Simple turnover penalty when previous weights available (encourages stability per spec).
   - Solver: `scipy.optimize.minimize` with Sequential Least Squares Programming (SLSQP).
   - If optimizer fails, fall back to heuristic (risk-parity style) weights.
5. **Historical backtesting**
   - Evaluate 4 strategies over 2016-2025 history:
     - Standard (restricted): Fixed risk-on guardrails regardless of regime
     - Standard (unrestricted): Uniform-weighted stats with long-only bounds
     - Regime (restricted): Regime-aware guardrails with exponential weighting
     - Regime (unrestricted): Regime-aware inputs with long-only bounds
   - Store weights and performance metrics (annualized returns) in parquet format
6. **Visualization & deployment**
   - Interactive Bokeh dashboard with:
     - SPY price chart with regime band overlays
     - Stacked area chart showing allocation weights over time
     - VIX chart for volatility context
     - Performance comparison table (4 strategies with annualized returns)
     - Clickable regime segments showing allocation shifts
   - Deploy via Docker (daily refresh) or Cloudflare Pages (static HTML)

## Implementation status (✅ Complete)
1. ✅ `src/config.py`: ETF universe with BIL, regime definitions, constraints
2. ✅ `src/data_loader.py`: yfinance downloader with parquet caching
3. ✅ `src/features.py`: exponential-weighted returns/covariance, momentum signals
4. ✅ `src/regime.py`: rule-based classifier (Risk-On, Risk-Off, Inflation Shock)
5. ✅ `src/optimizer.py`: Sharpe optimizer with restricted/unrestricted variants
6. ✅ `src/pipeline.py`: end-to-end orchestration with state management
7. ✅ `src/history.py`: historical backtest runner for 4 strategies
8. ✅ `src/visualization/bokeh_app.py`: interactive dashboard with regime overlays
9. ✅ `main.py`: CLI entry for single-point analysis
10. ✅ `scripts/service.py`: Docker service with daily refresh + HTTP server
11. ✅ `deploy_cloudflare.sh`: Cloudflare Pages deployment script
12. ✅ Tests: `test_regime.py`, `test_optimizer.py` (passing)
13. ✅ `README.md`: comprehensive documentation with deployment guides

**Performance Results (2016-2025, annualized):**
- Standard (restricted): 5.9%
- Standard (unrestricted): 8.1%
- Regime (restricted): 6.3%
- Regime (unrestricted): 13.0%

## Data & dependency notes
- Dependencies: `pandas`, `numpy`, `yfinance`, `scipy`, `pydantic` (for config validation), `tabulate` (pretty tables), `pytest`.
- Store processed data under `data/cache/*.parquet` and reports under `reports/`.
- Maintain ASCII-only source per repo rules.

## Future enhancements
- Machine learning regime detection (vs rule-based heuristics)
- GAN-based scenario generator (VolGAN/Fin-GAN) for stress testing
- Graph-based ensemble weighting (GraFiN-Gen) for cross-asset signals
- Real-time data feeds and intraday rebalancing
- Multi-account portfolio aggregation
- Tax-loss harvesting optimization
- Transaction cost modeling
