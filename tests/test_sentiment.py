"""Tests for the sentiment engine module."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.sentiment import (
    SentimentEngine,
    SentimentItem,
    SentimentSnapshot,
    _scrape_finnhub,
    _scrape_reddit,
    _scrape_stocktwits,
    _score_texts,
)


class TestSentimentItem:
    def test_creation(self):
        item = SentimentItem(
            source="test",
            text="Markets rallied strongly today",
            timestamp=datetime.now(timezone.utc),
            score=0.8,
            ticker="SPY",
        )
        assert item.score == 0.8
        assert item.ticker == "SPY"


class TestSentimentSnapshot:
    def test_creation(self):
        snap = SentimentSnapshot(
            timestamp=datetime.now(timezone.utc),
            market_score=0.5,
            ticker_scores={"SPY": 0.6, "TLT": -0.1},
            item_count=10,
            sources_used=["finnhub"],
        )
        assert snap.market_score == 0.5
        assert len(snap.ticker_scores) == 2


class TestScoreTexts:
    def test_empty_input(self):
        assert _score_texts([], "vader") == []

    def test_vader_fallback(self):
        """VADER may or may not be installed — just verify no crash."""
        texts = ["The market is looking great!", "Terrible crash ahead."]
        scores = _score_texts(texts, "vader")
        assert len(scores) == 2
        for s in scores:
            assert isinstance(s, float)


class TestSentimentEngine:
    def test_init(self):
        engine = SentimentEngine(scorer="vader", use_finnhub=False)
        assert engine.scorer == "vader"

    def test_as_fomo_components_keys(self):
        """Verify the engine returns the 4 expected component keys."""
        engine = SentimentEngine(
            scorer="vader",
            use_finnhub=False,
            use_stocktwits=False,
            use_reddit=False,
        )
        components = engine.as_fomo_components(tickers=["SPY"])
        expected_keys = {
            "news_sentiment_momentum",
            "social_media_intensity",
            "fear_language_ratio",
            "fed_hawkishness",
        }
        assert set(components.keys()) == expected_keys
        for v in components.values():
            assert -1.0 <= v <= 1.0

    def test_market_sentiment_score(self):
        engine = SentimentEngine(
            scorer="vader",
            use_finnhub=False,
            use_stocktwits=False,
            use_reddit=False,
        )
        score = engine.market_sentiment_score(tickers=["SPY"])
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


# ------------------------------------------------------------------
# Scraper tests (mocked HTTP)
# ------------------------------------------------------------------


class TestScrapeFinnhub:
    @patch("src.sentiment._SESSION")
    def test_returns_items_on_success(self, mock_session):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"headline": "Stocks rally on earnings", "datetime": 1700000000},
            {"headline": "Fed holds rates steady", "datetime": 1700001000},
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        items = _scrape_finnhub(
            ["AAPL"], datetime(2024, 1, 1), datetime(2024, 1, 3),
            api_key="test_key",
        )
        assert len(items) == 2
        assert items[0]["source"] == "finnhub"
        assert items[0]["ticker"] == "AAPL"
        assert "rally" in items[0]["text"]

    def test_returns_empty_without_api_key(self):
        """No API key → should return empty, not crash."""
        with patch.dict("os.environ", {}, clear=True):
            items = _scrape_finnhub(
                ["SPY"], datetime(2024, 1, 1), datetime(2024, 1, 3),
            )
            assert items == []


class TestScrapeStocktwits:
    @patch("src.sentiment._SESSION")
    def test_returns_items_on_success(self, mock_session):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "messages": [
                {"body": "SPY to the moon!", "created_at": "2024-06-01T12:00:00Z"},
                {"body": "Bearish divergence forming", "created_at": "2024-06-01T13:00:00Z"},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        items = _scrape_stocktwits(["SPY"])
        assert len(items) == 2
        assert items[0]["source"] == "stocktwits"
        assert items[1]["ticker"] == "SPY"

    @patch("src.sentiment._SESSION")
    def test_handles_api_error(self, mock_session):
        mock_session.get.side_effect = Exception("429 rate limited")
        items = _scrape_stocktwits(["SPY"])
        assert items == []


class TestScrapeReddit:
    @patch("src.sentiment._SESSION")
    def test_returns_items_on_success(self, mock_session):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {
                "children": [
                    {"data": {"title": "Is now a good time to buy?",
                              "selftext": "Market seems oversold",
                              "created_utc": 1700000000}},
                    {"data": {"title": "NVDA earnings play",
                              "selftext": "",
                              "created_utc": 1700001000}},
                ]
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_resp

        items = _scrape_reddit(subreddits=["stocks"], limit=10)
        assert len(items) == 2
        assert items[0]["source"] == "reddit/stocks"
        assert "oversold" in items[0]["text"]

    @patch("src.sentiment._SESSION")
    def test_handles_api_error(self, mock_session):
        mock_session.get.side_effect = Exception("403 Forbidden")
        items = _scrape_reddit()
        assert items == []
