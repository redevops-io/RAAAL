"""Broker execution module — Alpaca integration for paper & live trading.

Provides ``AlpacaExecutor`` which translates RAAAL weight dicts into
market orders via the Alpaca Trade API, supporting both paper and live
trading modes.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

TradingMode = Literal["paper", "live"]


@dataclass
class OrderResult:
    """Result of a single order submission."""

    ticker: str
    side: str  # "buy" or "sell"
    qty: float
    notional: Optional[float] = None
    status: str = "pending"
    order_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RebalanceResult:
    """Aggregate result of a rebalance operation."""

    timestamp: datetime
    target_weights: Dict[str, float]
    orders: List[OrderResult]
    portfolio_value: float
    mode: TradingMode
    errors: List[str] = field(default_factory=list)


class AlpacaExecutor:
    """Execute portfolio rebalances via Alpaca Trade API.

    Environment variables used:
        ALPACA_API_KEY      — API key
        ALPACA_SECRET_KEY   — Secret key
        ALPACA_BASE_URL     — Override base URL (default: paper trading)
        ALPACA_TRADING_MODE — "paper" (default) or "live"

    Usage
    -----
    >>> executor = AlpacaExecutor(mode="paper")
    >>> result = executor.rebalance({"SPY": 0.6, "TLT": 0.3, "BIL": 0.1})
    """

    # Tickers that need special handling for Alpaca
    TICKER_MAP: Dict[str, str] = {
        "BTC-USD": "BTC/USD",  # Alpaca crypto symbol format
    }

    # Tickers that cannot be traded on Alpaca
    UNTRADEABLE: set = {"^VIX", "^VVIX"}

    def __init__(
        self,
        mode: TradingMode = "paper",
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        min_order_value: float = 1.0,
        dry_run: bool = False,
    ) -> None:
        self.mode = mode
        self.dry_run = dry_run
        self.min_order_value = min_order_value

        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")

        if base_url:
            self._base_url = base_url
        elif mode == "live":
            self._base_url = "https://api.alpaca.markets"
        else:
            self._base_url = "https://paper-api.alpaca.markets"

        self._api: Any = None

    def _connect(self) -> Any:
        """Lazily connect to Alpaca API."""
        if self._api is not None:
            return self._api

        if not self._api_key or not self._secret_key:
            raise RuntimeError(
                "Alpaca API credentials not configured. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )

        try:
            import alpaca_trade_api as tradeapi

            self._api = tradeapi.REST(
                self._api_key,
                self._secret_key,
                self._base_url,
                api_version="v2",
            )
            return self._api
        except ImportError:
            raise ImportError(
                "alpaca-trade-api package not installed. "
                "Install with: pip install alpaca-trade-api"
            )

    def _map_ticker(self, ticker: str) -> str:
        """Map RAAAL ticker to Alpaca-compatible symbol."""
        return self.TICKER_MAP.get(ticker, ticker)

    def get_portfolio_value(self) -> float:
        """Fetch current portfolio equity from Alpaca."""
        api = self._connect()
        account = api.get_account()
        return float(account.portfolio_value)

    def get_current_positions(self) -> Dict[str, float]:
        """Fetch current position values as a fraction of portfolio."""
        api = self._connect()
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)
        if portfolio_value <= 0:
            return {}

        positions = api.list_positions()
        pos_dict: Dict[str, float] = {}
        for pos in positions:
            symbol = str(pos.symbol)
            market_value = float(pos.market_value)
            # Reverse map Alpaca symbols back to RAAAL tickers
            raaal_ticker = symbol
            for raaal, alpaca in self.TICKER_MAP.items():
                if alpaca == symbol:
                    raaal_ticker = raaal
                    break
            pos_dict[raaal_ticker] = market_value / portfolio_value

        return pos_dict

    def rebalance(
        self,
        target_weights: Dict[str, float],
        portfolio_value: Optional[float] = None,
    ) -> RebalanceResult:
        """Rebalance portfolio to match target weights.

        Calculates the difference between current and target positions,
        then submits market orders to close the gap.
        """
        now = datetime.now(timezone.utc)
        orders: List[OrderResult] = []
        errors: List[str] = []

        if self.dry_run:
            logger.info("[DRY RUN] Would rebalance to: %s", target_weights)
            return RebalanceResult(
                timestamp=now,
                target_weights=target_weights,
                orders=[],
                portfolio_value=0.0,
                mode=self.mode,
                errors=[],
            )

        try:
            api = self._connect()
            if portfolio_value is None:
                portfolio_value = self.get_portfolio_value()

            current_positions = self.get_current_positions()

            for ticker, target_pct in target_weights.items():
                if ticker in self.UNTRADEABLE:
                    continue

                alpaca_symbol = self._map_ticker(ticker)
                current_pct = current_positions.get(ticker, 0.0)
                diff_pct = target_pct - current_pct
                diff_value = diff_pct * portfolio_value

                if abs(diff_value) < self.min_order_value:
                    continue

                side = "buy" if diff_value > 0 else "sell"
                notional = abs(diff_value)

                try:
                    order = api.submit_order(
                        symbol=alpaca_symbol,
                        notional=round(notional, 2),
                        side=side,
                        type="market",
                        time_in_force="day",
                    )
                    orders.append(
                        OrderResult(
                            ticker=ticker,
                            side=side,
                            qty=0.0,  # notional order, qty determined by market
                            notional=notional,
                            status=str(order.status),
                            order_id=str(order.id),
                        )
                    )
                    logger.info(
                        "Order submitted: %s %s $%.2f (%s)",
                        side.upper(),
                        alpaca_symbol,
                        notional,
                        order.status,
                    )
                except Exception as exc:
                    error_msg = f"Order failed for {ticker}: {exc}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    orders.append(
                        OrderResult(
                            ticker=ticker,
                            side=side,
                            qty=0.0,
                            notional=notional,
                            status="failed",
                            error=str(exc),
                        )
                    )

        except Exception as exc:
            error_msg = f"Rebalance failed: {exc}"
            logger.error(error_msg)
            errors.append(error_msg)
            portfolio_value = portfolio_value or 0.0

        return RebalanceResult(
            timestamp=now,
            target_weights=target_weights,
            orders=orders,
            portfolio_value=portfolio_value,
            mode=self.mode,
            errors=errors,
        )

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        api = self._connect()
        cancelled = api.cancel_all_orders()
        count = len(cancelled) if cancelled else 0
        logger.info("Cancelled %d open orders", count)
        return count

    def close_all_positions(self) -> int:
        """Liquidate all positions. Returns count of closed positions."""
        api = self._connect()
        closed = api.close_all_positions()
        count = len(closed) if closed else 0
        logger.info("Closed %d positions", count)
        return count
