"""NLP sentiment engine — direct API scrapers + FinGPT / VADER scoring.

Provides a unified ``SentimentEngine`` class that:
1. Scrapes financial news & social media via direct REST APIs
   (Finnhub, Stocktwits, Reddit JSON).
2. Scores text using transformer-based FinGPT/FinBERT or fallback VADER.
3. Aggregates into a single market-sentiment float and per-ticker scores.

No dependency on ``finnlp`` — all scraping uses ``requests`` directly.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

ScorerType = Literal["fingpt", "vader", "auto"]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SentimentItem:
    """Single scored text item."""

    source: str
    text: str
    timestamp: datetime
    score: float  # −1 (bearish) to +1 (bullish)
    ticker: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentimentSnapshot:
    """Aggregate snapshot of market sentiment."""

    timestamp: datetime
    market_score: float  # aggregate −1 … +1
    ticker_scores: Dict[str, float]
    item_count: int
    sources_used: List[str]
    raw_items: List[SentimentItem] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# Scrapers (direct REST API – no finnlp dependency)
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "RAAAL-SentimentBot/1.0"})

_FINNHUB_BASE = "https://finnhub.io/api/v1"
_STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
_REDDIT_BASE = "https://www.reddit.com"


def _scrape_finnhub(
    tickers: List[str],
    start: datetime,
    end: datetime,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scrape company-news headlines from Finnhub REST API.

    Requires a free Finnhub API key (env ``FINNHUB_API_KEY`` or arg).
    https://finnhub.io/docs/api/company-news
    """
    key = api_key or os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        logger.info(
            "No FINNHUB_API_KEY set — Finnhub scraping skipped"
        )
        return []

    items: List[Dict[str, Any]] = []
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    for ticker in tickers:
        try:
            resp = _SESSION.get(
                f"{_FINNHUB_BASE}/company-news",
                params={
                    "symbol": ticker,
                    "from": start_s,
                    "to": end_s,
                    "token": key,
                },
                timeout=10,
            )
            resp.raise_for_status()
            for art in resp.json():
                headline = art.get("headline", "")
                if not headline:
                    continue
                items.append({
                    "source": "finnhub",
                    "text": headline,
                    "timestamp": pd.Timestamp(
                        art.get("datetime", start.timestamp()),
                        unit="s",
                    ),
                    "ticker": ticker,
                })
        except Exception as exc:
            logger.debug("Finnhub fetch error for %s: %s", ticker, exc)
    logger.info("Finnhub: scraped %d headlines", len(items))
    return items


def _scrape_stocktwits(tickers: List[str]) -> List[Dict[str, Any]]:
    """Scrape latest Stocktwits messages via public REST API.

    No auth required for the public streams endpoint.
    https://api.stocktwits.com/developers/docs/api#streams-symbol
    """
    items: List[Dict[str, Any]] = []
    for ticker in tickers:
        try:
            resp = _SESSION.get(
                f"{_STOCKTWITS_BASE}/streams/symbol/{ticker}.json",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            messages = data.get("messages", [])
            for msg in messages:
                body = msg.get("body", "")
                if not body:
                    continue
                created = msg.get("created_at", "")
                try:
                    ts = pd.Timestamp(created)
                except Exception:
                    ts = pd.Timestamp(datetime.now(timezone.utc))
                items.append({
                    "source": "stocktwits",
                    "text": body,
                    "timestamp": ts,
                    "ticker": ticker,
                })
        except Exception as exc:
            logger.debug("Stocktwits error for %s: %s", ticker, exc)
    logger.info("Stocktwits: scraped %d messages", len(items))
    return items


def _scrape_reddit(
    subreddits: Optional[List[str]] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Scrape Reddit posts from finance subreddits via JSON API.

    Uses Reddit's public ``.json`` endpoint (no auth needed,
    rate-limited to ~60 req/min with a User-Agent header).
    """
    subreddits = subreddits or [
        "wallstreetbets", "investing", "stocks",
    ]
    items: List[Dict[str, Any]] = []
    for sub in subreddits:
        try:
            resp = _SESSION.get(
                f"{_REDDIT_BASE}/r/{sub}/new.json",
                params={"limit": limit},
                timeout=10,
            )
            resp.raise_for_status()
            posts = resp.json().get("data", {}).get("children", [])
            for post in posts:
                d = post.get("data", {})
                title = d.get("title", "")
                selftext = d.get("selftext", "")
                text = f"{title} {selftext}".strip()
                if not text:
                    continue
                created_utc = d.get("created_utc")
                if isinstance(created_utc, (int, float)):
                    ts = pd.Timestamp(created_utc, unit="s")
                else:
                    ts = pd.Timestamp(datetime.now(timezone.utc))
                items.append({
                    "source": f"reddit/{sub}",
                    "text": text,
                    "timestamp": ts,
                    "ticker": None,
                })
        except Exception as exc:
            logger.debug("Reddit error for r/%s: %s", sub, exc)
    logger.info("Reddit: scraped %d posts", len(items))
    return items


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


def _score_vader(texts: List[str]) -> List[float]:
    """Fallback VADER sentiment scoring."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        return [analyzer.polarity_scores(t)["compound"] for t in texts]
    except ImportError:
        logger.warning("vaderSentiment not installed — returning neutral scores")
        return [0.0] * len(texts)


def _score_fingpt(texts: List[str], model_name: str = "FinGPT/fingpt-sentiment_llama2-13b_lora") -> List[float]:
    """Score texts with a FinGPT sentiment model via HuggingFace transformers."""
    try:
        from transformers import pipeline as hf_pipeline

        # Use a sentiment-analysis pipeline with a finance-tuned model
        # Fall back to a smaller model if the large one is unavailable
        try:
            scorer = hf_pipeline("text-classification", model=model_name, truncation=True, max_length=512)
        except Exception:
            logger.info("FinGPT model unavailable, falling back to ProsusAI/finbert")
            scorer = hf_pipeline("text-classification", model="ProsusAI/finbert", truncation=True, max_length=512)

        results = scorer(texts, batch_size=16)
        scores = []
        for r in results:
            label = r["label"].lower()
            conf = r["score"]
            if "positive" in label or "bullish" in label:
                scores.append(conf)
            elif "negative" in label or "bearish" in label:
                scores.append(-conf)
            else:
                scores.append(0.0)
        return scores
    except ImportError:
        logger.warning("transformers not installed — falling back to VADER")
        return _score_vader(texts)


def _score_texts(
    texts: List[str],
    scorer_type: ScorerType = "auto",
) -> List[float]:
    """Route to the appropriate scorer."""
    if not texts:
        return []
    if scorer_type == "vader":
        return _score_vader(texts)
    if scorer_type == "fingpt":
        return _score_fingpt(texts)
    # auto: try FinGPT first, fall back to VADER
    try:
        return _score_fingpt(texts)
    except Exception:
        return _score_vader(texts)


# ---------------------------------------------------------------------------
# Sentiment Engine
# ---------------------------------------------------------------------------


class SentimentEngine:
    """End-to-end sentiment pipeline: scrape → score → aggregate.

    Usage
    -----
    >>> engine = SentimentEngine(scorer="auto")
    >>> snapshot = engine.fetch_and_score(["SPY", "QQQ"])
    >>> print(snapshot.market_score)
    """

    def __init__(
        self,
        scorer: ScorerType = "auto",
        lookback_days: int = 3,
        finnhub_api_key: Optional[str] = None,
        use_reddit: bool = True,
        use_stocktwits: bool = True,
        use_finnhub: bool = True,
    ) -> None:
        self.scorer = scorer
        self.lookback_days = lookback_days
        self.finnhub_api_key = finnhub_api_key
        self.use_reddit = use_reddit
        self.use_stocktwits = use_stocktwits
        self.use_finnhub = use_finnhub
        self._cache: Optional[SentimentSnapshot] = None

    def fetch_and_score(
        self,
        tickers: Optional[List[str]] = None,
        force_refresh: bool = False,
    ) -> SentimentSnapshot:
        """Fetch texts from all enabled sources, score, and aggregate."""
        if self._cache is not None and not force_refresh:
            age = (datetime.now(timezone.utc) - self._cache.timestamp).total_seconds()
            if age < 3600:  # 1-hour cache
                return self._cache

        tickers = tickers or ["SPY", "QQQ", "TLT", "GLD", "BTC-USD"]
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=self.lookback_days)
        end = now

        raw_items: List[Dict[str, Any]] = []
        sources_used: List[str] = []

        if self.use_finnhub:
            finnhub_items = _scrape_finnhub(tickers, start, end, self.finnhub_api_key)
            raw_items.extend(finnhub_items)
            if finnhub_items:
                sources_used.append("finnhub")

        if self.use_stocktwits:
            st_items = _scrape_stocktwits(tickers)
            raw_items.extend(st_items)
            if st_items:
                sources_used.append("stocktwits")

        if self.use_reddit:
            reddit_items = _scrape_reddit()
            raw_items.extend(reddit_items)
            if reddit_items:
                sources_used.append("reddit")

        # Score all texts
        texts = [item["text"] for item in raw_items if item.get("text")]
        scores = _score_texts(texts, self.scorer) if texts else []

        # Build SentimentItems
        scored_items: List[SentimentItem] = []
        for item, score in zip(raw_items, scores):
            scored_items.append(
                SentimentItem(
                    source=item["source"],
                    text=item["text"],
                    timestamp=item.get("timestamp", now),
                    score=score,
                    ticker=item.get("ticker"),
                )
            )

        # Aggregate per ticker
        ticker_scores: Dict[str, float] = {}
        for ticker in tickers:
            ticker_items = [si for si in scored_items if si.ticker == ticker]
            if ticker_items:
                ticker_scores[ticker] = float(np.mean([si.score for si in ticker_items]))
            else:
                ticker_scores[ticker] = 0.0

        # Overall market score
        if scored_items:
            market_score = float(np.mean([si.score for si in scored_items]))
        else:
            market_score = 0.0

        snapshot = SentimentSnapshot(
            timestamp=now,
            market_score=np.clip(market_score, -1.0, 1.0),
            ticker_scores=ticker_scores,
            item_count=len(scored_items),
            sources_used=sources_used,
            raw_items=scored_items,
        )
        self._cache = snapshot
        return snapshot

    def market_sentiment_score(self, tickers: Optional[List[str]] = None) -> float:
        """Convenience: return a single aggregate market sentiment float."""
        snapshot = self.fetch_and_score(tickers)
        return snapshot.market_score

    def as_fomo_components(self, tickers: Optional[List[str]] = None) -> Dict[str, float]:
        """Return NLP-derived component scores for FOMO/FOBI integration.

        Returns a dict with four keys that plug into the extended
        ``FOMO_COMPONENT_WEIGHTS`` in config:
            news_sentiment_momentum, social_media_intensity,
            fear_language_ratio, fed_hawkishness
        """
        snapshot = self.fetch_and_score(tickers)
        items = snapshot.raw_items

        # 1. News sentiment momentum: average score of news sources
        news_items = [i for i in items if i.source in ("finnhub",)]
        news_score = float(np.mean([i.score for i in news_items])) if news_items else 0.0

        # 2. Social media intensity: volume-weighted average of social sources
        social_items = [i for i in items if "reddit" in i.source or "stocktwits" in i.source]
        social_score = float(np.mean([i.score for i in social_items])) if social_items else 0.0

        # 3. Fear language ratio: proportion of negative items
        if items:
            negative_count = sum(1 for i in items if i.score < -0.3)
            fear_ratio = negative_count / len(items)
            # Map to −1..1 scale (more fear = more negative)
            fear_score = -(fear_ratio * 2 - 1)  # 0% negative → +1, 100% → −1
        else:
            fear_score = 0.0

        # 4. Fed hawkishness: scan for hawkish/dovish keywords
        hawk_words = {"hike", "tighten", "hawkish", "restrictive", "inflation"}
        dove_words = {"cut", "ease", "dovish", "accommodate", "stimulus"}
        hawk_count, dove_count = 0, 0
        for item in items:
            lower_text = item.text.lower()
            hawk_count += sum(1 for w in hawk_words if w in lower_text)
            dove_count += sum(1 for w in dove_words if w in lower_text)
        total_fd = hawk_count + dove_count
        if total_fd > 0:
            fed_score = (dove_count - hawk_count) / total_fd  # positive = dovish/bullish
        else:
            fed_score = 0.0

        return {
            "news_sentiment_momentum": np.clip(news_score, -1.0, 1.0),
            "social_media_intensity": np.clip(social_score, -1.0, 1.0),
            "fear_language_ratio": np.clip(fear_score, -1.0, 1.0),
            "fed_hawkishness": np.clip(fed_score, -1.0, 1.0),
        }
