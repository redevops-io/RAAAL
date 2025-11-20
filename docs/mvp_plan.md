# Regime-adjusted Asset Allocation MVP Plan

## Inputs & guiding references
- **Vuletic (2025)** highlights the need for multi-asset models that capture co-movements, adapt to regime shifts, and remain arbitrage-aware (Ch. 2–3). We leverage these ideas by focusing on a modular pipeline with explainable statistical blocks and stress-tested heuristics.
- **Product brief (`Regime-adjusted asset allocation model.odt`)** specifies a liquid ETF universe (SPY/SH, TLT/TBT, LQD, SHV-SGOV, DBC, optional GLD/HYG/UUP) plus a three-regime playbook (Risk-On, Risk-Off, Inflation Shock) with monthly rebalancing and constrained optimization (max position sizes, cash floors, leverage cap \<= 1.25x).

## System overview
1. **Data ingestion**
   - Pull daily bars for ETFs + proxies (`SPY`, `SH`, `TLT`, `TBT`, `LQD`, `SGOV`, `DBC`, optional `HYG`, `GLD`, `UUP`, `TIP`, `^VIX`).
   - Cache locally (parquet) to avoid repeated downloads.
2. **Feature engineering**
   - Compute log returns, rolling volatility/Sharpe (21/63/252d lookbacks), beta to `SPY`, rolling correlations (`SPY`/`TLT`), credit-spread proxy (`LQD` minus `TLT` short-term momentum), commodity/TIP momentum.
3. **Regime detection**
   - Risk-On: `SPY` price > 200d SMA, `^VIX` < 18, credit spread narrowing, `corr(SPY, TLT)` \< 0.2.
   - Risk-Off: `SPY` < 200d SMA, `^VIX` > 22, credit spread widening, `corr(SPY, TLT)` \< -0.3.
   - Inflation Shock: `corr(SPY, TLT)` > 0, commodity momentum > 0, TIP momentum < 0.
   - Otherwise default to the closest matching regime (priority: risk-off > inflation > risk-on).
4. **Portfolio optimizer**
   - Target: maximize Sharpe = `(μᵀw - r_f) / sqrt(wᵀΣw)` using exponential-weighted returns/ covariance (63d half-life) with `SGOV` yield as `r_f`.
   - Constraints:
     - Sum of weights = 1.
     - Per-ETF bounds (from brief; e.g., `SPY` \<= 0.5, `TLT` \<= 0.5, inverse ETFs \<= 0.2, `DBC` \<= 0.15, cash between regime-specific limits).
     - Optional leverage cap via `||w||₁ \<= 1.25`.
     - Simple turnover penalty when previous weights available (encourages stability per spec).
   - Solver: `scipy.optimize.minimize` with Sequential Least Squares Programming (SLSQP).
   - If optimizer fails, fall back to heuristic (risk-parity style) weights.
5. **Rebalancing logic**
   - Base schedule: end-of-month. Trigger immediate rebalance if regime flips, volatility spike (portfolio volatility > 2σ of trailing year), or weight drift > 10%.
6. **Reporting & UX hooks**
   - Generate Markdown/CSV summary with final weights, rationale (derived from signals), key metrics (exp return, vol, Sharpe, beta, drawdown proxy), and regime diagnostics.
   - Persist JSON snapshot for dashboard integration.

## Implementation roadmap
1. `src/config.py`: definitions for ETF metadata, regimes, constraints, lookbacks.
2. `src/data_loader.py`: download/cache ETF history via `yfinance` (with CLI args for start/end dates).
3. `src/features.py`: compute returns, vol, Sharpe, betas, correlations, credit-spread proxies.
4. `src/regime.py`: encapsulate rule-based regime classifier with explainability payload.
5. `src/optimizer.py`: Sharpe-maximizing optimizer with constraint set + fallback risk-parity.
6. `src/pipeline.py`: orchestrate ingestion → features → regime → optimization → reporting.
7. `main.py`: CLI entry (`python main.py --start 2015-01-01 --end 2025-11-01`).
8. `src/reporting.py`: format outputs (console table + CSV/JSON).
9. Tests (`tests/test_regime.py`, `tests/test_optimizer.py`) to ensure deterministic behavior on synthetic inputs.
10. `README.md`: usage, architecture summary, references to Vuletic (2025) + product brief.

## Data & dependency notes
- Dependencies: `pandas`, `numpy`, `yfinance`, `scipy`, `pydantic` (for config validation), `tabulate` (pretty tables), `pytest`.
- Store processed data under `data/cache/*.parquet` and reports under `reports/`.
- Maintain ASCII-only source per repo rules.

## Stretch ideas (post-MVP)
- Plug GAN-based scenario generator (VolGAN/Fin-GAN style) to stress-test allocations.
- Add graph-based ensemble weighting (GraFiN-Gen) for cross-asset signals.
- Web dashboard (Streamlit) for interactive allocation explanations.
