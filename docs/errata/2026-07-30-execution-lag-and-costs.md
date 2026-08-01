# Erratum 2026-07-30-01 — Execution lag and transaction costs

**Status:** published
**Affects:** all annualized performance figures published before 2026-07-30, in `README.md` and on the deployed dashboard
**Supersedes:** the "Performance Results (Annualized, 2016-2025)" block as published through 2026-07-29
**Severity:** material — the headline figure changes sign
**Superseded results:** preserved below, not deleted

---

## Summary

Backtested performance was overstated because portfolio weights earned the return of the day they were decided, and because no transaction costs were charged. Both defects are corrected. Every annualized figure changes; the most-promoted variant changes sign.

## What was wrong

**1. One-day look-ahead in return attribution.** Weights computed using data through date *d* were forward-filled from *d* and multiplied by *d*'s own return. A strategy therefore earned the return of the session it was still deciding how to trade. Correct behaviour is to lag weights by one executable period.

**2. No transaction costs.** No backtest path charged anything for turnover. The only cost model in the codebase (`transaction_cost=0.001`) applied solely inside the DRL training reward and never reached a published number.

Two further defects were corrected in the same release and affect the `ml` regime mode rather than these headline figures:

**3. Future-trained regime models fed historical backtests.** Random Forest and Gradient Boosting classifiers were trained on the *full* timeline by the dashboard, persisted to `data/models/`, then reloaded at the start of the next backtest and used to classify every historical date. CI purged `data/cache` and `data/history` but not `data/models`, so the contamination persisted across runs.

**4. Model accuracy was measured on a shuffled split.** `train_test_split(..., stratify=...)` on a time series with overlapping rolling-window features places near-duplicate observations on both sides of the split. **The ">80% accuracy" figure previously published for the ensemble regime classifier is withdrawn.** It was not out-of-sample and should not be cited.

## Corrected figures

Annualized, 2016-01-01 → 2025-11-20.

| Variant | Published (superseded) | Corrected | Change |
|---|---:|---:|---:|
| Standard (Restricted) | 5.87% | **2.17%** | −3.70pp |
| Standard (Unrestricted) | 8.10% | **6.13%** | −1.97pp |
| Regime (Restricted) | 6.33% | **2.22%** | −4.11pp |
| Regime (Unrestricted) ⭐ | 13.00% | **−2.83%** | **−15.83pp** |

## Attribution of the correction

Each defect isolated, same weights and same price data throughout:

| Variant | Defective (no lag, no cost) | + execution lag | + lag and costs |
|---|---:|---:|---:|
| Standard (Restricted) | 5.99% | 2.72% | 2.17% |
| Standard (Unrestricted) | 7.70% | 6.78% | 6.13% |
| Regime (Restricted) | 6.43% | 2.84% | 2.22% |
| Regime (Unrestricted) | 12.28% | 1.13% | −2.83% |

Two things follow.

**The look-ahead dominates, not the costs.** For Regime (Unrestricted), execution lag alone removes 11.15pp; costs remove a further 3.96pp. Adding costs without fixing the lag would have left most of the overstatement in place.

**The variant promoted as best was the one most dependent on the defect.** Regime (Unrestricted) carried the ⭐ in the README and lost 11.15pp to the lag fix, against 0.92pp for Standard (Unrestricted). This is the expected signature: among variants searched over the same data, the one that looks best is disproportionately likely to be the one exploiting the flaw hardest. It is also the reason the [redesign plan](../quantify_redesign_plan.md) puts deflated-Sharpe and trial-count reporting ahead of any leaderboard.

The `Defective` column reproduces the published figures to within 0.1–0.7pp. The residual gap is explained by data restatement since original publication (yfinance retroactively adjusts `Adj Close` for splits and dividends), the removal of contaminated `ml`-mode variants, and a refreshed price window — none of which was reproducible before this release, because no run manifest existed.

## Scope and limitations

- **The `ml` regime mode produces no results in this run.** The model-cutoff gate now refuses any artifact whose training data reaches the date it is asked to predict, and the only available artifacts were trained on the full timeline. Losing the mode is the correct outcome; restoring it requires training inside the walk-forward loop.
- **Transaction costs are set to a flat 10bps of notional traded** (`config.TRANSACTION_COST_BPS`), applied to turnover. This is a conservative placeholder for liquid US ETFs, not a calibrated microstructure model. It does not model market impact, and capacity is unmodelled.
- **These remain hypothetical, backtested results.** No capital was managed. They are not a track record, and under SEC Rule 206(4)-1(e) backtested performance is hypothetical performance.
- **Costs still are not applied to the ex-ante `sharpe`, `exp_return` and `exp_vol` columns**, which are forward-looking estimates from μ and Σ rather than realized performance. They should not be read as achieved results.

## What changed in code

| Defect | Fix |
|---|---|
| Execution lag | `src/history.py` — new `strategy_daily_returns()` shifts weights by `config.EXECUTION_LAG_DAYS` before meeting returns. Headline metric and dashboard growth curve now share this one implementation, so they cannot disagree. |
| Transaction costs | Same function charges `config.TRANSACTION_COST_BPS` on turnover, including the initial trade in from cash. |
| Future-trained models | `src/ensemble_regime.py` — models record a `train_cutoff`; `load_ensemble_models(as_of=...)` refuses any model trained through the requested date, and refuses models with no recorded cutoff. `.github/workflows/daily-deploy.yml` now purges `data/models`. |
| Shuffled split | Chronological split with a 21-row embargo; raises a clear error when the training window spans only one regime rather than silently shuffling. |
| Synthetic high/low | `src/features_alpha.py` — ADX features are emitted only when real OHLC is present. The previous `close*1.005` / `close*0.995` band made ADX a deterministic function of close while presenting as a volatility feature. |
| Non-reproducibility | New `src/reproducibility.py` — `seed_everything()` pins Python, NumPy and Torch RNGs plus cuDNN determinism; every run writes `data/history/run_manifest.json` with commit SHA, working-tree dirty flag, resolved package versions, parameters, and content digests of inputs and outputs. `requirements-core.lock` pins the environment. |

Guarded by `tests/test_leakage.py` (15 tests), now run in CI before any deploy. The suite includes a control test asserting that disabling the lag *reproduces* the original defect, so the regression test cannot silently stop testing anything.

## Reproducing

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-core.lock
.venv/bin/python -m src.history --start 2016-01-01 --end 2025-11-20 --refresh
cat data/history/run_manifest.json
```

Note that `--refresh` re-downloads from yfinance, whose adjusted history is restated over time. Exact reproduction of *these* figures requires the price snapshot digest recorded in the manifest; snapshot retention is Release 1 work.

---

*Raised and corrected 2026-07-30 during the Release 0 correctness pass. The superseded figures are retained in this document by design — see [quantify_redesign_plan.md §5.0](../quantify_redesign_plan.md), Release 0 exit criterion 5.*
