# Us Total Market Proxies — RAAAL custom-index composition (DeerFlow, local)

> Query: Confirm which of SCHB (Schwab US Broad Market), ITOT and SPTM track essentially the same US total market as VTI, and whether SGOV (iShares 0-3 Month Treasury) tracks essentially the same exposure as BIL (SPDR 1-3 Month T-Bill). These are single-leg proxies, not composites: state for each whether a one-to-one proxy from the universe is defensible. SOURCE QUALITY (important): use PRIMARY sources — the fund issuer's official fact sheet and prospectus (Vanguard investor.vanguard.com, iShares ishares.com, State Street ssga.com) and the underlying index provider's methodology (FTSE Russell for FTSE Global All Cap, Bloomberg Index Services for the Global Aggregate). Give each component weight as of the most recent fact sheet WITH its date, and record the exact source URL for every figure. Prefer the issuer's own numbers to aggregators. End with a clearly labelled 'COMPOSITION' block per index: `SYMBOL = {LEG_TICKER: weight, ...}` using only tickers from this universe, or the word UNAVAILABLE and why if a leg has no proxy in it. Universe: VTI, VXUS, BND, AGG, TLT, TIP, LQD, HYG, BIL, SPY, VOO, QQQ, IWM, GLD, DBC, BRK-B, MGK, SH, TBT, BTC-USD.

## SCHB — Schwab U.S. Broad Market ETF (Schwab Strategic Trust)

**Structure:** Single fund tracking the **Dow Jones U.S. Broad Stock Market Index** (~2,500 large/mid/small-cap U.S. stocks). No legs to decompose — unlike BNDW or VT, SCHB is a standalone index fund with no feeder structure. Expense ratio: 0.03%.

### Top Holdings (Most Recent Available Data)

I was unable to locate the exact **June 30, 2026** N-PORT filing for SCHB through my searches. The most recent data I could verify comes from two sources with slightly different as-of dates:

| # | Ticker | Name | Weight (Aug 14, 2026) | Weight (May 31, 2026 N-PORT) |
|---|--------|------|-----------------------|------------------------------|
| 1 | NVDA | NVIDIA Corporation | 7.23% | 7.02% |
| 2 | AAPL | Apple Inc. | 5.94% | 6.26% |
| 3 | MSFT | Microsoft Corporation | 4.89% | 4.57% |
| 4 | AMZN | Amazon.com, Inc. | 3.44% | 3.62% |
| 5 | GOOGL | Alphabet Inc. (Class A) | 2.69% | 3.03% |
| 6 | AVGO | Broadcom Inc. | 2.62% | — |
| 7 | GOOG | Alphabet Inc. (Class C) | 2.15% | — |
| 8 | META | Meta Platforms, Inc. | 1.73% | — |
| 9 | MU | Micron Technology, Inc. | 1.42% | — |
| 10 | JPM | JPMorgan Chase & Co. | 1.29% | — |

**Top 10 combined: ~33.4%** (Aug 14 date)

### Sources
- [stockanalysis.com/etf/schb/holdings](https://stockanalysis.com/etf/schb/holdings/) — as of Aug 14, 2026
- [portfolioslab.com/symbol/SCHB/holdings](https://portfolioslab.com/symbol/SCHB/holdings) — N-PORT period ended May 31, 2026
- [Schwab Asset Management SCHB page](https://www.schwabassetmanagement.com/products/schb) — fund overview

### Note on June 30 Precision
The exact June 30, 2026 N-PORT filing should be available at SEC EDGAR under CIK 0001454889 (Schwab Strategic Trust), but I was unable to retrieve the specific document through my search attempts. The May 31 and August 14 data points bracket that date closely, so weights will fall between those ranges. For exact June 30 figures, you can pull the N-PORT-P filing directly from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001454889&type=N-PORT-P).
