"""Canonical instrument identity, and a refusal when the words do not pin one.

    resolve("VTI")                     -> VTI, an ETF
    resolve("Vanguard Total Stock")    -> VTI, by alias
    resolve("the S&P 500")             -> AMBIGUOUS: an index and four funds
    resolve("^VIX")                    -> an INDEX, and not tradable

**Never guesses.** An alias that names more than one instrument produces an
`AMBIGUOUS` resolution listing what it could have meant, and an alias that names
none produces `UNRESOLVED`. Neither returns the closest ticker, because "closest"
is a similarity score standing in for a decision — and a plan priced against the
wrong instrument is wrong in a way that looks entirely correct: right shape,
right dates, wrong company.

**An index is not a fund.** `^GSPC` is a number computed from constituents; SPY
is a thing you can buy. A backtest that bought the index is a backtest of an
instrument that does not exist, and quoting it beside one that does is the
comparison this system refuses everywhere else. `tradable` is therefore part of
the identity rather than a note in a description.

**This is not a search.** It is a lookup against declared identities. A resolver
that ranked candidates would be making a selection on the user's behalf, and the
whole architecture treats an unreported selection as the most damaging thing a
platform can do quietly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple


class InstrumentKind(str, Enum):
    ETF = "ETF"
    EQUITY = "EQUITY"
    INDEX = "INDEX"
    """Computed from constituents. Not purchasable, and priced anyway by any
    engine that is not told the difference."""

    CRYPTO = "CRYPTO"


class Outcome(str, Enum):
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class Instrument:
    """One canonical identity, and the words people use for it."""

    symbol: str
    kind: InstrumentKind
    name: str
    aliases: Tuple[str, ...] = ()

    @property
    def tradable(self) -> bool:
        return self.kind is not InstrumentKind.INDEX

    def to_json(self) -> Dict[str, str]:
        return {"symbol": self.symbol, "kind": self.kind.value,
                "name": self.name, "tradable": self.tradable}


@dataclass(frozen=True)
class Resolution:
    """What a query resolved to, or why it did not."""

    outcome: Outcome
    query: str
    instrument: Optional[Instrument] = None
    candidates: Tuple[Instrument, ...] = ()
    detail: str = ""

    @property
    def resolved(self) -> bool:
        return self.outcome is Outcome.RESOLVED


#: The instruments this build knows, with the words people use for them.
#:
#: Small and declared rather than fetched. Every entry is an instrument the
#: synthetic snapshot actually carries, plus the indices people name when they
#: mean a fund — which is where the ambiguity lives and is the reason this
#: module exists at all.
INSTRUMENTS: Tuple[Instrument, ...] = (
    Instrument("VTI", InstrumentKind.ETF, "Vanguard Total Stock Market ETF",
               ("vanguard total stock market", "total stock market",
                "us total market fund", "total market")),
    Instrument("VOO", InstrumentKind.ETF, "Vanguard S&P 500 ETF",
               ("vanguard s&p 500", "vanguard 500")),
    Instrument("SPY", InstrumentKind.ETF, "SPDR S&P 500 ETF Trust",
               ("spdr s&p 500", "spider")),
    Instrument("RSP", InstrumentKind.ETF,
               "Invesco S&P 500 Equal Weight ETF",
               ("equal weight s&p 500", "s&p 500 equal weight")),
    Instrument("QQQ", InstrumentKind.ETF, "Invesco QQQ Trust",
               ("nasdaq 100 etf", "invesco qqq")),
    Instrument("BND", InstrumentKind.ETF,
               "Vanguard Total Bond Market ETF",
               ("vanguard total bond market", "total bond market")),
    Instrument("AGG", InstrumentKind.ETF,
               "iShares Core U.S. Aggregate Bond ETF", ("aggregate bond",)),
    Instrument("TLT", InstrumentKind.ETF,
               "iShares 20+ Year Treasury Bond ETF", ("long treasuries",)),
    Instrument("GLD", InstrumentKind.ETF, "SPDR Gold Shares", ("gold etf",)),
    Instrument("IWM", InstrumentKind.ETF, "iShares Russell 2000 ETF",
               ("russell 2000 etf",)),
    Instrument("VXUS", InstrumentKind.ETF,
               "Vanguard Total International Stock ETF",
               ("total international stock",)),
    # Carried by the synthetic snapshot at full coverage over its 2016-2025 span
    # (tests/fixtures/prices_synthetic.parquet) but previously unnamed here, so a
    # strategy that used them refused for an asset the data already holds. Naming
    # them turns that refusal into a run: cash, inflation-protected and corporate
    # credit, broad commodities, mega-cap growth, and the inverse hedges.
    Instrument("BIL", InstrumentKind.ETF,
               "SPDR Bloomberg 1-3 Month T-Bill ETF",
               ("t-bills", "treasury bills", "1-3 month treasury", "cash etf")),
    Instrument("TIP", InstrumentKind.ETF,
               "iShares TIPS Bond ETF",
               ("tips", "inflation-protected treasuries",
                "inflation protected bonds", "treasury inflation protected")),
    Instrument("LQD", InstrumentKind.ETF,
               "iShares iBoxx $ Investment Grade Corporate Bond ETF",
               ("investment grade corporate bonds", "corporate bonds",
                "ig corporates")),
    Instrument("HYG", InstrumentKind.ETF,
               "iShares iBoxx $ High Yield Corporate Bond ETF",
               ("high yield bonds", "high-yield corporates", "junk bonds")),
    Instrument("DBC", InstrumentKind.ETF,
               "Invesco DB Commodity Index Tracking Fund",
               ("commodities", "broad commodities", "commodity etf")),
    Instrument("MGK", InstrumentKind.ETF,
               "Vanguard Mega Cap Growth ETF",
               ("mega cap growth", "mega-cap growth", "large-cap growth")),
    Instrument("SH", InstrumentKind.ETF,
               "ProShares Short S&P 500 ETF",
               ("inverse s&p 500", "short s&p 500", "s&p 500 short")),
    Instrument("TBT", InstrumentKind.ETF,
               "ProShares UltraShort 20+ Year Treasury ETF",
               ("ultrashort treasuries", "inverse long treasuries",
                "short long-term treasuries")),
    # A custom index: no series of its own in the snapshot, computed from its
    # components (US total market + total international) at pricing time. See
    # market_data/custom_indices.py for the composition and evaluation/
    # index_calc.py for the calculation. Named here so a plan can hold it.
    Instrument("VT", InstrumentKind.ETF,
               "Vanguard Total World Stock (composed of VTI + VXUS)",
               ("total world", "total world stock", "world stock market",
                "global stock market")),
    # Single-component composites — proxies made explicit in custom_indices.py.
    Instrument("SCHB", InstrumentKind.ETF,
               "Schwab U.S. Broad Market ETF (tracks the US total market, as VTI)",
               ("schwab us broad market", "schwab total market")),
    Instrument("SGOV", InstrumentKind.ETF,
               "iShares 0-3 Month Treasury Bond ETF (as BIL)",
               ("ishares 0-3 month treasury", "0-3 month treasury")),
    Instrument("BRK-B", InstrumentKind.EQUITY, "Berkshire Hathaway Inc. Class B",
               ("berkshire", "berkshire hathaway", "berkshire b")),
    Instrument("BTC-USD", InstrumentKind.CRYPTO, "Bitcoin in US dollars",
               ("bitcoin", "btc")),
    # Indices. Named because people say them, and kept distinct because an
    # engine handed one would price something nobody can hold.
    Instrument("^GSPC", InstrumentKind.INDEX, "S&P 500 Index",
               ("s&p 500 index", "sp500 index")),
    Instrument("^VIX", InstrumentKind.INDEX, "CBOE Volatility Index",
               ("vix", "volatility index")),
    Instrument("^VVIX", InstrumentKind.INDEX, "CBOE VVIX Index", ("vvix",)),
)

#: Words that name a family rather than an instrument.
#:
#: "the S&P 500" is the clearest case: it is an index, and it is also what
#: somebody means when they say they hold a fund tracking it. Resolving it to
#: any single one of those is a choice about the user's portfolio made by a
#: lookup table, so it resolves to all of them and refuses.
FAMILIES: Mapping[str, Tuple[str, ...]] = {
    "s&p 500": ("^GSPC", "VOO", "SPY", "RSP"),
    "sp500": ("^GSPC", "VOO", "SPY", "RSP"),
    "the s&p": ("^GSPC", "VOO", "SPY", "RSP"),
    "nasdaq 100": ("QQQ",),
    "total bond": ("BND", "AGG"),
}

_BY_SYMBOL = {one.symbol.upper(): one for one in INSTRUMENTS}
_BY_ALIAS: Dict[str, Tuple[Instrument, ...]] = {}
for _one in INSTRUMENTS:
    for _alias in (_one.name.lower(), *_one.aliases):
        _BY_ALIAS.setdefault(_alias, ())
        _BY_ALIAS[_alias] = _BY_ALIAS[_alias] + (_one,)


def _normalise(query: str) -> str:
    text = str(query or "").strip().lower()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return re.sub(r"\s+", " ", text)


def resolve(query: str) -> Resolution:
    """One instrument, several, or none — and never a guess."""
    raw = str(query or "").strip()
    if not raw:
        return Resolution(Outcome.UNRESOLVED, raw,
                          detail="nothing was named")

    exact = _BY_SYMBOL.get(raw.upper())
    if exact is not None:
        return Resolution(Outcome.RESOLVED, raw, instrument=exact)

    text = _normalise(raw)

    family = FAMILIES.get(text)
    if family is not None:
        found = tuple(_BY_SYMBOL[one] for one in family)
        if len(found) == 1:
            return Resolution(Outcome.RESOLVED, raw, instrument=found[0])
        return Resolution(
            Outcome.AMBIGUOUS, raw, candidates=found,
            detail=f"{raw!r} names a family, not an instrument: "
                   + ", ".join(f"{one.symbol} ({one.kind.value})"
                               for one in found)
                   + ". Picking one would choose a holding on your behalf, and "
                     "the index among them cannot be bought at all")

    by_alias = _BY_ALIAS.get(text, ())
    if len(by_alias) == 1:
        return Resolution(Outcome.RESOLVED, raw, instrument=by_alias[0])
    if len(by_alias) > 1:
        return Resolution(
            Outcome.AMBIGUOUS, raw, candidates=by_alias,
            detail=f"{raw!r} matches {len(by_alias)} instruments: "
                   + ", ".join(one.symbol for one in by_alias))

    return Resolution(
        Outcome.UNRESOLVED, raw,
        detail=f"{raw!r} names no instrument this build knows. It is not "
               "resolved to the closest match, because a plan priced against "
               "the wrong instrument is wrong in a way that looks correct")


def resolve_all(queries: Sequence[str]) -> Tuple[Tuple[Instrument, ...],
                                                 Tuple[Resolution, ...]]:
    """Resolve several, and report every one that did not resolve.

    Both halves returned. A caller given only the successes would build a
    portfolio quietly missing whatever could not be named, which is the
    silently-dropped-instruction defect wearing market-data clothes.
    """
    found, failed = [], []
    for query in queries:
        outcome = resolve(query)
        (found if outcome.resolved else failed).append(
            outcome.instrument if outcome.resolved else outcome)
    return tuple(found), tuple(failed)
