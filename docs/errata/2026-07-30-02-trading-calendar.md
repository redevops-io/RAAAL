# Erratum 2026-07-30-02 — Trading calendar contamination

**Status:** published
**Affects:** all performance figures computed from `data/history/prices.parquet`, including those in [Erratum 2026-07-30-01](2026-07-30-execution-lag-and-costs.md)
**Severity:** material — volatility understated, annualization mis-scaled
**Superseded results:** preserved, not deleted

---

## Summary

The joined price panel carries a seven-day index because `BTC-USD` trades at weekends. Equity and ETF prices were forward-filled across Saturday and Sunday, so **31.1% of "daily returns" were zeros that are not market data**. This deflated realized volatility, and the 252-periods-per-year annualization was applied to a series with roughly 365 observations per year.

## What was wrong

`data_loader.download_prices` concatenates `Adj Close` across all requested tickers and forward-fills. The universe is nine ETFs, but the auxiliary series include `BTC-USD`, which has no weekend gap. The union index is therefore seven-day, and every five-day instrument was padded.

Two consequences, in opposite directions:

1. **Volatility understated.** Injecting ~104 zero-return observations per year into a return series reduces its standard deviation without reducing its total return. Every ratio computed from it — Sharpe above all — was correspondingly inflated.
2. **Annualized return understated.** `(1 + total)^(252/n) − 1` with `n ≈ 365 × years` produces a smaller exponent than the data warrants.

## Corrected figures

`methodology/hrp@3 × protocol/long-warmup@1`, 2017-05-23 → 2025-11-19:

| Quantity | Contaminated | Corrected | Change |
|---|---:|---:|---:|
| Observations | 3,103 | 2,073 | −1,030 |
| Zero-return days | 31.1% | 3.4% | −27.7pp |
| Annualized return | 1.3664% | **2.2892%** | +0.92pp |
| Volatility | 3.1879% | **4.0939%** | +0.91pp |
| Sharpe | 0.4286 | **0.5592** | +0.13 |

Both the return and the volatility were understated. Sharpe rose because the return correction outweighed the volatility correction here; that is specific to this series and should not be assumed elsewhere.

**Erratum 2026-07-30-01's figures are affected by the same defect** and are superseded in turn. They were computed through `src/history.py` against the same panel. Their *relative* conclusion is unchanged — the look-ahead removal remains the dominant correction, and the Regime (Unrestricted) sign flip stands — but the absolute levels carry the calendar error and should be recomputed under a declared protocol before being cited.

## Why it was not caught earlier

The calendar was an **ambient choice**. Nothing declared that returns were daily-business-day, so nothing could check it. `_annualize` hard-coded 252 and the runner hard-coded `√252`, both silently assuming a calendar the data did not have.

This is the same class of defect as the covariance estimator (Release 1) and the fallback and precedence rules (Release 2): a decision that materially moves a published number while living nowhere in an artifact. It was found by inspecting an observation count that did not match the declared period — 3,103 observations over roughly 8.5 years — which was visible only because the assessment layer reports `observations` as a fact rather than folding it into a ratio.

## The fix

The trading calendar is now **declared protocol data**, not an assumption:

```yaml
walk_forward:
  calendar: business_days      # business_days | all_days
  periods_per_year: 252
```

- `evaluation.runner.apply_calendar` restricts the panel before execution, so a methodology cannot see padded observations.
- Annualization and volatility scaling both read `periods_per_year` from the protocol; `history._annualize` takes it as a parameter rather than assuming 252.
- A protocol wanting a seven-day index must say `calendar: all_days`, and its `periods_per_year` must agree.

Changing `calendar` or `periods_per_year` changes the protocol's content hash, so results measured under the contaminated settings and the corrected ones are distinguishable rather than silently conflated.

## Limitations of the fix

- `business_days` means Monday–Friday. It does **not** yet exclude exchange holidays, so a small number of padded observations remain (3.4% of the series are zero-return days, some of which are holidays and some genuinely flat). A proper exchange calendar is outstanding work.
- The fix applies to the evaluation path (`src/evaluation/runner.py`). The legacy engine path in `src/history.py` accepts `periods_per_year` but its callers still pass the default; figures produced directly by `python -m src.history` remain on the contaminated calendar until that path is migrated.

---

*Raised and corrected 2026-07-30 during Release 2. Superseded figures are retained here by design.*
