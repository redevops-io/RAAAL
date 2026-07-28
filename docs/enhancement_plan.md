# RAAAL Enhancement Plan — Leveraging Open-Source Quant/AI Frameworks

> Generated: April 9, 2026  
> Scope: Integration roadmap for microsoft/qlib, FinGPT/FinNLP, FinRL/FinRL-X, and insanely-fast-whisper into the RAAAL pipeline.

---

## Executive Summary

RAAAL is a **regime-adjusted asset allocation engine** that combines rule-based and ML regime detection (RF + GBM ensembles), Sharpe-maximizing optimization, HRP, network analysis, FOMO/FOBI sentiment, nowcasting, and a 17-strategy evaluation harness. It operates on a 10-ticker ETF universe with auxiliary signals and deploys via GitHub Actions → Cloudflare Pages.

After a deep analysis of the codebase and the five external repos, enhancements are grouped into **7 workstreams** ranked by impact and implementation effort.

---

## Current RAAAL Architecture (Capability Map)

| Layer | What RAAAL Has | Key Gaps |
|-------|---------------|----------|
| **Data** | yfinance daily bars, parquet cache | No news/sentiment data, no alt data, no streaming, no high-frequency |
| **Features** | Log returns, exponential mean/cov, rolling Sharpe, beta, credit spread, commodity/TIP momentum, FOMO/FOBI composite | No NLP-derived features, no deep-learned factors, no Alpha158/360 feature sets |
| **Regime Detection** | Rule-based (3 regimes) + RF/GBM ensemble | No concept drift adaptation, no market dynamics modeling, no temporal models |
| **Strategies** | 17 strategies (momentum, mean-reversion, risk-based, factor, sentiment overlay) | No RL-based strategies, no DRL portfolio allocation, no adaptive rotation |
| **Optimization** | Sharpe-max (SLSQP) + HRP + risk-parity fallback | No end-to-end differentiable optimization, no RL-based execution |
| **Execution** | None (offline backtest only) | No broker integration, no paper trading, no order execution |
| **Evaluation** | Walk-forward backtest, annualized returns, Sharpe/beta metrics | No IC analysis, no signal decay, no transaction cost models, no benchmark-relative analysis |
| **LLM/NLP** | None | No sentiment analysis, no earnings call analysis, no news-driven signals |
| **Audio/Voice** | None | No earnings call transcription, no Fed speech parsing |

---

## Workstream 1: Deep Learning Forecasting Models (from Qlib)
**Impact: 🔴 HIGH | Effort: 🟡 MEDIUM**

### What Qlib Offers
Qlib's **Model Zoo** contains 20+ forecasting models (LightGBM, LSTM, GRU, Transformer, TRA, HIST, ADARNN, TCN) trained on Alpha158/Alpha360 feature sets. It also provides a data-processing pipeline with high-performance caching and an auto-quant workflow (`qrun`).

### Integration Points for RAAAL

#### 1a. Alpha158 Feature Set → `src/features.py`
Add Qlib-style technical indicators to augment RAAAL's current log-return + momentum features:

```
New features to add:
- MACD (12/26/9), Bollinger Bands (20d), RSI (14/30), ADX
- VWAP ratios, OBV momentum (if volume data added)
- 60+ derived features from Alpha158 (rolling std, kurtosis, autocorrelation)
```

**Implementation:** New `src/features_alpha.py` module that computes Alpha158-style features from the existing `prices` DataFrame. These feed into both the regime ensemble and a new return-forecasting layer.

#### 1b. Return Forecasting Layer → New `src/forecaster.py`
Replace or augment the naive `exponential_mean()` return estimate with a learned model:

| Model | Purpose | Why |
|-------|---------|-----|
| **LightGBM** | Baseline ML forecaster | Fast, handles mixed features well, Qlib's top performer on Alpha158 |
| **LSTM/GRU** | Temporal pattern mining | Captures sequential dependencies in ETF returns |
| **Transformer** | Multi-horizon forecasting | Self-attention for cross-asset dependency learning |

**Data flow:**
```
prices → Alpha158 features → forecaster.predict() → μ̂ (forecasted returns)
                                                      ↓
                             optimizer.optimize_weights(μ̂, Σ, regime)
```

This replaces the exponentially-weighted mean `μ` in `pipeline.py` with a model-predicted `μ̂`, directly improving portfolio quality.

#### 1c. Market Dynamics Adaptation (DDG-DA / Rolling Retraining)
RAAAL's ensemble models are trained once and loaded from disk. Qlib's **DDG-DA** and **Rolling Retraining** address concept drift:

- Add a rolling retrain schedule in the daily CI pipeline
- Implement distribution-shift detection to trigger emergency retraining
- Add to `ensemble_regime.py`: `retrain_if_stale(timeline, max_age_days=30)`

#### 1d. Qlib Data Server for Performance
Replace yfinance with Qlib's columnar data format for 10-50x faster feature computation:
- Convert cached parquet to Qlib binary format
- Use Qlib's `ExpressionCache` for derived features
- Benefit: sub-second feature computation vs. current multi-second pandas operations

---

## Workstream 2: NLP Sentiment Signals (from FinNLP + FinGPT)
**Impact: 🔴 HIGH | Effort: 🟡 MEDIUM**

### What FinNLP/FinGPT Offers
- **FinNLP**: Crawlers for financial news (Finnhub/Yahoo/Reuters/SeekingAlpha), social media (Stocktwits, Reddit/WSB), SEC filings, and Google Trends
- **FinGPT**: LLM-based sentiment scoring with LoRA fine-tuning, using stock price changes as auto-labels

### Integration Points for RAAAL

#### 2a. News Sentiment Feature → New `src/sentiment.py`
Add a sentiment signal pipeline that feeds into regime detection and the FOMO/FOBI indicator:

```python
# New module: src/sentiment.py
class SentimentEngine:
    def fetch_news(self, tickers, start, end) -> pd.DataFrame
    def score_sentiment(self, headlines: List[str]) -> pd.Series  # -1 to +1
    def aggregate_market_sentiment(self) -> float  # composite score
```

**Data sources** (via FinNLP adapters):
| Source | What | Signal For |
|--------|------|-----------|
| Finnhub | Aggregated financial news headlines | Macro regime sentiment |
| Stocktwits | Retail trader sentiment | FOMO/FOBI enhancement |
| Reddit/WSB | Retail positioning signals | Contrarian indicator |
| SEC EDGAR | 10-K/10-Q filings | Credit risk signals for LQD/HYG |

#### 2b. FOMO/FOBI Sentiment Enhancement
The current `fomo_fobi.py` is purely price-based (10 components). Add NLP-derived components:

```
New FOMO/FOBI components:
- news_sentiment_momentum: Rolling 5d news sentiment zscore
- social_media_intensity: StockTwits/Reddit volume spike detection
- fear_language_ratio: NLP analysis of fear vs. greed keywords
- fed_speech_hawkishness: Hawkish/dovish scoring of Fed communications
```

Increase `FOMO_COMPONENT_WEIGHTS` in `config.py` to include these 4 new components (redistributing ~15% total weight from existing components).

#### 2c. LLM-as-Signal (FinGPT Integration)
Use FinGPT's fine-tuned models as a "virtual analyst":

- Daily: Feed top 50 financial headlines → FinGPT → sentiment label (positive/negative/neutral)
- Use the aggregate sentiment shift as an **overlay signal** in `optimize_weights()`:
  - Strong negative shift → increase cash floor by 5%
  - Strong positive shift → relax inverse cap by 5%

This can run in the daily CI pipeline via an API call (OpenAI, local LLM, or Hugging Face Inference API).

---

## Workstream 3: Reinforcement Learning Strategies (from FinRL / FinRL-X)
**Impact: 🔴 HIGH | Effort: 🔴 HIGH**

### What FinRL/FinRL-X Offers
- **FinRL**: DRL agents (A2C, DDPG, PPO, SAC, TD3) for portfolio allocation
- **FinRL-X**: Weight-centric pipeline with modular strategy → backtest → execution layers, adaptive rotation, and Alpaca broker integration

### Integration Points for RAAAL

#### 3a. DRL Portfolio Allocator → New Strategy in `src/strategies.py`
Add a deep RL-based allocation strategy that learns optimal weights from the environment:

```python
def drl_portfolio_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    regime: Optional[str],
    context: Dict[str, object],
) -> Dict[str, float]:
    """PPO/SAC agent that generates portfolio weights."""
    # State: [returns, volatility, regime_encoding, sentiment, macro_nowcasts]
    # Action: continuous weight vector for 10 assets
    # Reward: risk-adjusted return (Sharpe-like)
```

**Training approach:**
1. Define Gym environment with RAAAL's asset universe
2. State space: last 20 days of returns + regime features + FOMO score + nowcasts
3. Action space: 10-dimensional continuous (portfolio weights, softmax-constrained)
4. Reward: daily Sharpe contribution with turnover penalty
5. Train PPO/SAC using Stable Baselines 3

This becomes strategy #18 in the `DEFAULT_STRATEGIES` list and participates in the existing evaluation framework.

#### 3b. RL-Based Timing Overlay
Inspired by FinRL-X's KAMA timing overlay:
- Train a binary RL agent: "should we rebalance today?" (yes/no)
- State: regime diagnostics + days since last rebalance + VIX + sentiment
- This replaces the simple "regime changed → rebalance" logic in `pipeline.py`

#### 3c. Adaptive Multi-Asset Rotation (from FinRL-X Use Case 3)
FinRL-X's adaptive rotation strategy is highly relevant to RAAAL:

| FinRL-X Feature | RAAAL Mapping |
|----------------|---------------|
| Growth Tech / Real Assets / Defensive groups | Map to RAAAL's `REGIME_BUCKETS` |
| Information Ratio-based group selection | New group scoring in `strategies.py` |
| Residual momentum ranking | Enhance `_trailing_simple_returns()` |
| Slow + Fast regime detection | Augment `detect_regime()` with fast risk-off trigger |
| Trailing/absolute stop-loss | New risk control layer in `pipeline.py` |

Add this as `adaptive_rotation_strategy()` in `strategies.py`.

#### 3d. Live/Paper Trading Execution (from FinRL-X)
RAAAL currently has zero execution capability. Adopt FinRL-X's Alpaca integration:

```python
# New module: src/execution.py
class TradeExecutor:
    def __init__(self, broker: str = "alpaca"):
        ...
    def execute_rebalance(self, target_weights: Dict[str, float]):
        ...
    def get_current_positions(self) -> Dict[str, float]:
        ...
    def pre_trade_risk_check(self, target_weights) -> bool:
        ...
```

This enables the daily CI pipeline to optionally paper-trade the recommended allocation.

---

## Workstream 4: Earnings Call / Fed Speech Transcription (from Insanely-Fast-Whisper)
**Impact: 🟡 MEDIUM | Effort: 🟢 LOW**

### What Insanely-Fast-Whisper Offers
Blazingly fast Whisper transcription (150 min audio → 98 seconds on A100). Supports batched inference, Flash Attention 2, and speaker diarization.

### Integration Points for RAAAL

#### 4a. Fed Speech Transcription Pipeline
Fed speeches, FOMC press conferences, and earnings calls are leading indicators for regime shifts. Add a transcription → analysis pipeline:

```
Audio URL → insanely-fast-whisper → transcript.json
    → FinGPT sentiment scoring → hawkish/dovish signal
    → feed into regime detection & FOMO/FOBI
```

**Implementation:**
```python
# New module: src/audio_signals.py
def transcribe_audio(url: str) -> str:
    """Uses insanely-fast-whisper to transcribe earnings/Fed audio."""
    
def extract_monetary_policy_signal(transcript: str) -> float:
    """Score -1 (very dovish) to +1 (very hawkish)."""
    
def process_earnings_call(url: str, ticker: str) -> Dict[str, float]:
    """Transcribe → extract sentiment, guidance, key metrics."""
```

#### 4b. Earnings Season Signal
During earnings season, transcribe key ETF constituent earnings calls and aggregate:
- SPY/QQQ: top-10 holdings earnings sentiment
- LQD/HYG: credit-relevant issuers
- Feed aggregate into the nowcasting module as `earnings_momentum`

---

## Workstream 5: Enhanced Backtesting Framework
**Impact: 🟡 MEDIUM | Effort: 🟡 MEDIUM**

Drawing from both Qlib and FinRL-X:

#### 5a. Information Coefficient (IC) Analysis (from Qlib)
Add signal quality metrics to RAAAL's reporting:
- IC / Rank IC for each feature vs. forward returns
- IC decay curve (1d, 5d, 21d, 63d horizons)
- Monthly IC heatmap
- Auto-correlation of forecasting signals

New `src/signal_analysis.py` module.

#### 5b. Transaction Cost Model (from FinRL-X)
RAAAL's backtest assumes zero transaction costs. Add:
- Bid-ask spread estimation per ETF
- Impact model based on trade size
- Slippage estimation
- Update `_strategy_total_return()` in `history.py` to include costs

#### 5c. Multi-Benchmark Comparison (from FinRL-X)
FinRL-X uses the `bt` library for benchmark-relative analysis. Add:
- 60/40 benchmark (SPY/TLT)
- Risk-parity benchmark (ARC)
- Equal-weight benchmark
- S&P 500 buy-and-hold

#### 5d. Walk-Forward Validation (from Qlib)
Formalize RAAAL's existing rolling evaluation into proper walk-forward:
- Purged cross-validation for regime model training
- Embargo periods to prevent look-ahead bias
- Rolling window model retraining with performance decay alerts

---

## Workstream 6: Data Pipeline Upgrade
**Impact: 🟡 MEDIUM | Effort: 🟢 LOW**

#### 6a. Multi-Source Data Fetching (from FinRL-X)
Replace single-source yfinance dependency:

```python
# Enhanced src/data_loader.py
class DataManager:
    sources = ["yahoo", "fmp", "qlib"]  # priority order
    
    def get_price_data(self, tickers, start, end) -> pd.DataFrame:
        """Try sources in priority order with automatic fallback."""
```

Benefits: redundancy, better data quality, access to FMP fundamentals.

#### 6b. Google Trends Integration (from FinNLP)
Add Google Trends data as a retail attention signal:
- Search volume for "recession", "inflation", "stock market crash"
- Feed as a feature into regime detection ensemble

#### 6c. SQLite Cache (from FinRL-X)
Upgrade from parquet-file-per-ticker to a proper SQLite database:
- Atomic writes, concurrent reads
- Schema versioning
- Metadata tracking (last refresh, data quality flags)

---

## Workstream 7: Automated R&D (from Qlib/RD-Agent)
**Impact: 🟡 MEDIUM | Effort: 🔴 HIGH**

Qlib's **RD-Agent** is an LLM-driven autonomous system for:
- Automated factor mining from financial reports
- Model architecture search and optimization
- Hypothesis generation → backtesting → evaluation loop

### Integration Point:
- Configure RD-Agent to target RAAAL's feature set and strategy space
- Let it explore new factor definitions from RAAAL's price/sentiment data
- Automatically test new strategy variants against the existing 17 strategies
- Human-in-the-loop approval for strategies that pass significance tests

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
| Task | Source | Files Modified |
|------|--------|---------------|
| Alpha158 features | Qlib | New `src/features_alpha.py`, update `pipeline.py` |
| News sentiment scraping | FinNLP | New `src/sentiment.py`, update `config.py` |
| LightGBM return forecaster | Qlib | New `src/forecaster.py`, update `pipeline.py` |
| Transaction cost model | FinRL-X | Update `src/history.py` |
| Multi-benchmark comparison | FinRL-X | Update `src/reporting.py` |

### Phase 2: Core ML Upgrades (2-4 weeks)
| Task | Source | Files Modified |
|------|--------|---------------|
| FOMO/FOBI NLP enhancement | FinGPT/FinNLP | Update `src/fomo_fobi.py`, `config.py` |
| LSTM/Transformer forecaster | Qlib | Extend `src/forecaster.py` |
| DRL portfolio strategy | FinRL | New `src/drl_strategy.py`, update `strategies.py` |
| Adaptive rotation strategy | FinRL-X | Update `src/strategies.py` |
| Rolling model retraining | Qlib | Update `ensemble_regime.py`, CI pipeline |
| IC analysis + signal quality | Qlib | New `src/signal_analysis.py` |

### Phase 3: Production Pipeline (4-8 weeks)
| Task | Source | Files Modified |
|------|--------|---------------|
| Paper/live trading via Alpaca | FinRL-X | New `src/execution.py`, update `main.py` |
| Audio transcription pipeline | insanely-fast-whisper | New `src/audio_signals.py` |
| Fed speech hawkish/dovish scoring | FinGPT + Whisper | Update `src/sentiment.py` |
| Multi-source data manager | FinRL-X | Rewrite `src/data_loader.py` |
| Automated factor mining | Qlib/RD-Agent | New `src/rd_agent_config.py` |
| Walk-forward validation | Qlib | Update `src/history.py` |

---

## New Dependencies to Add to `requirements.txt`

```
# Phase 1
lightgbm>=4.0
finnlp>=0.1.0
ta>=0.11.0                 # Technical Analysis library for Alpha158 features
requests>=2.31              # For news API calls

# Phase 2
torch>=2.0                  # For LSTM/Transformer models
stable-baselines3>=2.0      # For DRL strategies
gymnasium>=0.29             # RL environment
transformers>=4.35          # For FinGPT sentiment scoring

# Phase 3
alpaca-trade-api>=3.0       # For live/paper trading
insanely-fast-whisper>=0.0.15  # For audio transcription
pyqlib>=0.9.0               # For Qlib data server + models (optional)
```

---

## Architectural Diagram (Enhanced RAAAL)

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ yfinance │ │ FMP API  │ │  FinNLP  │ │ Whisper  │           │
│  │ (prices) │ │(prices+  │ │(news/    │ │(earnings │           │
│  │          │ │ fundmtls)│ │ social)  │ │ calls)   │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       └─────────────┴────────────┴────────────┘                  │
│                         ↓                                        │
│              ┌─────────────────────┐                             │
│              │  DataManager (SQLite)│                             │
│              └──────────┬──────────┘                             │
├─────────────────────────┼───────────────────────────────────────┤
│                    FEATURE LAYER                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ features │ │features_ │ │sentiment │ │audio_    │           │
│  │ .py      │ │alpha.py  │ │.py       │ │signals.py│           │
│  │(returns, │ │(Alpha158,│ │(FinGPT,  │ │(Whisper +│           │
│  │ cov, mom)│ │ RSI,MACD)│ │ news NLP)│ │ hawkish) │           │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘           │
│       └─────────────┴────────────┴────────────┘                  │
│                         ↓                                        │
├─────────────────────────┼───────────────────────────────────────┤
│                 MODEL / REGIME LAYER                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ Rule-based   │ │ ML Ensemble  │ │ Forecaster   │             │
│  │ regime.py    │ │ ensemble_    │ │ forecaster.py│             │
│  │              │ │ regime.py    │ │ (LightGBM,   │             │
│  │              │ │ (RF+GBM+    │ │  LSTM, TFM)  │             │
│  │              │ │  rolling     │ │              │             │
│  │              │ │  retrain)    │ │              │             │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘             │
│         └────────────────┴────────────────┘                      │
│                         ↓                                        │
├─────────────────────────┼───────────────────────────────────────┤
│                   STRATEGY LAYER                                 │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │ 17 existing│ │ DRL-based  │ │ Adaptive   │ │ Sentiment  │   │
│  │ strategies │ │ PPO/SAC    │ │ Rotation   │ │ overlay    │   │
│  │            │ │(FinRL)     │ │(FinRL-X)   │ │(FinGPT)    │   │
│  └──────┬─────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘   │
│         └──────────────┴──────────────┴──────────────┘           │
│                         ↓                                        │
│              ┌─────────────────────┐                             │
│              │  optimizer.py       │                             │
│              │  (Sharpe-max + HRP  │                             │
│              │   + risk controls)  │                             │
│              └──────────┬──────────┘                             │
├─────────────────────────┼───────────────────────────────────────┤
│                 EXECUTION LAYER (NEW)                             │
│              ┌─────────────────────┐                             │
│              │  execution.py       │                             │
│              │  (Alpaca paper/live │                             │
│              │   + risk checks)    │                             │
│              └──────────┬──────────┘                             │
├─────────────────────────┼───────────────────────────────────────┤
│                  REPORTING LAYER                                 │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                  │
│  │ Bokeh      │ │ Signal     │ │ IC/Backtest│                  │
│  │ Dashboard  │ │ Analysis   │ │ Analytics  │                  │
│  └────────────┘ └────────────┘ └────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary: Expected Impact

| Enhancement | Expected Improvement | Confidence |
|-------------|---------------------|------------|
| Alpha158 features + LightGBM forecaster | +2-5% annualized return via better μ estimates | High |
| NLP sentiment signals | Better regime transition detection (lead time) | Medium-High |
| DRL portfolio strategy | +1-3% via learned non-linear allocation | Medium |
| Adaptive rotation | Better drawdown control, faster de-risking | Medium-High |
| Transaction cost model | More realistic backtests, avoid overfitting | High |
| Earnings call transcription | Early warning for sector rotation | Medium |
| Live trading execution | Move from research to production | High (operational) |
| Automated factor mining (RD-Agent) | Continuous alpha discovery | Medium (long-term) |

**Bottom line:** These integrations can transform RAAAL from a research-grade backtest engine into a **production-capable, multi-signal, AI-native portfolio management system** — while maintaining its existing modular architecture and regime-aware philosophy.
