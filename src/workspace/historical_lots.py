"""Refuse a described historical holding rather than coercing it.

A user may describe a position they already own even when nothing asked them
to. Before this, "I already own 500 shares of AAPL that I bought in 2019 at
$50" compiled to an ordinary plan with no material blocker and no stated
fields — the holding was dropped. "I bought 10 shares of NVDA in May 2024"
went further and asked *how much are you starting with*, which folds a share
count into a cash amount.

Both are financially material. A share quantity recorded before a corporate
action is in units that no longer exist, and a cost basis is not starting
capital: one is what was paid, the other is what is available to invest.
`src/holdings/` can resolve the first correctly and is not wired to any user
surface, so the honest answer here is a refusal that names itself.

**Detection is deliberately broad and the consequence is a block, not a
guess.** A false positive costs a user a sentence of rephrasing. A false
negative puts a wrong number in front of somebody who will believe it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence, Tuple

#: The public code a refusal carries. Stable, quotable, and distinct from the
#: market-data block: this plan's instruments may be perfectly priceable.
HISTORICAL_LOTS_NOT_AVAILABLE = "HISTORICAL_LOTS_NOT_AVAILABLE"

#: Ownership stated in units. "I own 500 shares", "I hold 100 shares of NVDA".
_UNITS = re.compile(
    r"\b(?:i|we)\s+(?:already\s+)?(?:own|hold|have)\b[^.]{0,40}?"
    r"\b\d[\d,]*(?:\.\d+)?\s*(?:shares?|units?)\b",
    re.IGNORECASE)

#: A past purchase of a security. "I bought 10 shares in 2019", "purchased
#: $80,000 of VTI over the last five years".
_PAST_PURCHASE = re.compile(
    r"\b(?:bought|purchased|acquired|been\s+buying|have\s+been\s+holding)\b",
    re.IGNORECASE)

#: Cost basis by any of its usual names.
_BASIS = re.compile(
    r"\b(?:cost\s+basis|basis\s+of|average\s+cost|paid\s+\$|"
    r"bought\s+(?:it\s+|them\s+)?at|purchase\s+price)\b",
    re.IGNORECASE)

#: An existing balance in an instrument rather than in cash.
_EXISTING_POSITION = re.compile(
    r"\b(?:i|we)\s+(?:already\s+)?(?:own|hold|have)\b[^.]{0,30}?"
    r"\$[\d,]+(?:\.\d+)?\s+(?:of|in|worth\s+of)\s+[A-Z]{1,5}\b")

#: Equity compensation lots, which are historical acquisitions with a vest
#: date and a basis whether or not the user calls them that.
_VESTED = re.compile(
    r"\b(?:vested|rsus?\s+(?:that\s+)?vested|already\s+vested|"
    r"shares?\s+from\s+(?:my\s+)?vesting)\b",
    re.IGNORECASE)


@dataclass(frozen=True)
class HistoricalLotSignal:
    """Why a description was read as naming an existing holding."""

    matched: str
    excerpt: str

    @property
    def question(self) -> str:
        return ("This describes a holding you already own. Quantify cannot "
                "model existing positions yet.")

    @property
    def why_it_matters(self) -> str:
        return (
            "A share count recorded before a split is in units that no longer "
            "exist, and a cost basis is not the same as money available to "
            "invest. Treating either as starting capital would produce a "
            "figure that looks reasonable and is wrong by whatever the "
            "instrument has done since. Describe only what you contribute "
            "from here, and the plan will run.")


def detect(text: str) -> Tuple[HistoricalLotSignal, ...]:
    """Every reason this description names an existing holding."""
    if not text:
        return ()
    found = []
    for name, pattern in (("units_held", _UNITS),
                          ("past_purchase", _PAST_PURCHASE),
                          ("cost_basis", _BASIS),
                          ("existing_position", _EXISTING_POSITION),
                          ("vested_equity", _VESTED)):
        match = pattern.search(text)
        if match:
            start = max(0, match.start() - 20)
            found.append(HistoricalLotSignal(name, text[start:match.end() + 20]))
    return tuple(found)


def blocks(text: str) -> bool:
    return bool(detect(text))
