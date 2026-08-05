"""What asset did the user intend?

Not "which ticker" — the question is semantic. Somebody who writes "SPX ETF"
has named an *index* and asked for a *fund*, which is internally inconsistent
and is exactly why "there is no price history for SPX" is a true and useless
answer: the plan would not run with SPX priced either, because SPX is not a
thing you can buy.

So identity is an unresolved field like any other. It observes a phrase,
offers candidates, and the user's choice becomes a `ScenarioAmendment`. The
description is never edited: "SPX ETF" stays "SPX ETF" forever, and the plan
records that the user meant SPY.

Confidence decides the interaction, not the outcome:

    high     one candidate, and no serious rival -> state the reading, offer
             to change it
    medium   two or more plausible readings      -> ask
    low      nothing recognisable                -> ask, and do not guess

The tiers only change what is said. A high-confidence reading is still an
amendment the user can see and overturn, never a silent rewrite.
"""
from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

import yaml

_MAPPING_FILE = (pathlib.Path(__file__).resolve().parents[2]
                 / "data" / "catalog_instruments.yaml")


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Candidate:
    symbol: str
    name: str
    score: float
    #: Why it ranked where it did. A list of tickers in an order nobody can
    #: account for is a recommendation wearing a resolution's clothes.
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Identification:
    """One observed phrase, and what it might be."""

    observed: str
    candidates: Tuple[Candidate, ...]
    confidence: Confidence
    #: Why this is being asked, in the user's terms. "No price history" is
    #: true of SPX and explains nothing; "that is the index, not a fund you
    #: can buy" is the actual problem.
    reason: str = ""
    #: The registry that produced this reading. Pinned on the plan, so a
    #: stored interpretation can say which catalogue it came from.
    registry_digest: str = ""
    concept_id: str = ""

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None


#: Names for the symbols the pilot can price. A dropdown reading "SPY / VOO"
#: asks the user to already know the answer; "SPY — SPDR S&P 500 ETF Trust"
#: does not.
NAMES: Mapping[str, str] = {
    "SPY": "SPDR S&P 500 ETF Trust",
    "VOO": "Vanguard S&P 500 ETF",
    "IVV": "iShares Core S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "QQQ": "Invesco Nasdaq-100 ETF",
    "DIA": "SPDR Dow Jones Industrial Average ETF",
    "IWM": "iShares Russell 2000 ETF",
    "IWB": "iShares Russell 1000 ETF",
    "VXUS": "Vanguard Total International Stock ETF",
    "VEA": "Vanguard Developed Markets ETF",
    "VWO": "Vanguard Emerging Markets ETF",
    "BND": "Vanguard Total Bond Market ETF",
    "AGG": "iShares Core U.S. Aggregate Bond ETF",
    "TLT": "iShares 20+ Year Treasury Bond ETF",
    "IEF": "iShares 7-10 Year Treasury Bond ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "BIL": "SPDR 1-3 Month T-Bill ETF",
    "TIP": "iShares TIPS Bond ETF",
    "MUB": "iShares National Muni Bond ETF",
    "GLD": "SPDR Gold Shares",
    "DBC": "Invesco DB Commodity Index Fund",
    "VNQ": "Vanguard Real Estate ETF",
    "VTV": "Vanguard Value ETF",
    "VBR": "Vanguard Small-Cap Value ETF",
    "QUAL": "iShares MSCI USA Quality Factor ETF",
    "MTUM": "iShares MSCI USA Momentum Factor ETF",
    "USMV": "iShares MSCI USA Min Vol Factor ETF",
    "VIG": "Vanguard Dividend Appreciation ETF",
    "RSP": "Invesco S&P 500 Equal Weight ETF",
    "SGOV": "iShares 0-3 Month Treasury Bond ETF",
    "BTC-USD": "Bitcoin",
}

#: Phrases that name an index rather than something purchasable, and the funds
#: that track them. The mismatch is the thing worth saying out loud.
_INDEX_FUNDS: Mapping[str, Tuple[str, ...]] = {
    "SPX": ("SPY", "VOO", "IVV"),
    "S&P 500": ("SPY", "VOO", "IVV"),
    "S&P500": ("SPY", "VOO", "IVV"),
    "SP500": ("SPY", "VOO", "IVV"),
    "GSPC": ("SPY", "VOO", "IVV"),
    "NDX": ("QQQ",),
    "NASDAQ 100": ("QQQ",),
    "NASDAQ": ("QQQ",),
    "DJIA": ("DIA",),
    "DOW": ("DIA",),
    "DOW JONES": ("DIA",),
    "RUSSELL 2000": ("IWM",),
    "RUSSELL 1000": ("IWB",),
}

#: Descriptive phrases with more than one reasonable fund behind them.
_THEMES: Mapping[str, Tuple[str, ...]] = {
    "TOTAL MARKET": ("VTI",),
    "TOTAL STOCK MARKET": ("VTI",),
    "TOTAL BOND": ("BND", "AGG"),
    "TREASURIES": ("IEF", "TLT", "SHY"),
    "T BILLS": ("BIL", "SGOV"),
    "TBILLS": ("BIL", "SGOV"),
    "CASH": ("BIL", "SGOV"),
    "GOLD": ("GLD",),
    "COMMODITIES": ("DBC",),
    "REAL ESTATE": ("VNQ",),
    "REITS": ("VNQ",),
    "INTERNATIONAL": ("VXUS", "VEA"),
    "EMERGING MARKETS": ("VWO",),
    "DEVELOPED MARKETS": ("VEA",),
    "MUNIS": ("MUB",),
    "MUNICIPAL BONDS": ("MUB",),
    "BITCOIN": ("BTC-USD",),
}

_ETF_WORDS = re.compile(r"\b(etf|fund|index fund|tracker)\b", re.IGNORECASE)
_NOISE = re.compile(r"\((?:[^)]*)\)|\b(etf|fund|index|the|a|an)\b", re.IGNORECASE)


def _clean(observed: str) -> str:
    """The phrase with its decoration removed.

    A model hands back "SP500 etf (no literal ticker given)" and the
    parenthetical is the model talking to us, not the user naming an asset.
    """
    without_notes = re.sub(r"\([^)]*\)", " ", observed)
    return re.sub(r"[^A-Za-z0-9&\s.-]", " ", without_notes).strip()


_ALIAS_CACHE: Optional[Mapping[str, str]] = None


def _alias_table() -> Mapping[str, str]:
    global _ALIAS_CACHE
    if _ALIAS_CACHE is None:
        _ALIAS_CACHE = aliases()
    return _ALIAS_CACHE


def identify(observed: str, *, priceable: Sequence[str] = ()) -> Identification:
    """What this phrase might be, and how sure we are.

    A thin adapter over `resolver.resolve` now. The tables that used to live
    here — phrase to funds, symbol to name — were a flat map answering several
    questions at once, and the registry separates them: a concept is what the
    user meant, an instrument is what can satisfy it, and an alias observation
    carries the facets (vehicle, issuer) that a `phrase -> ticker` dictionary
    could not hold.

    `priceable` filters candidates to what the deployment can value. Offering
    a fund the pilot cannot price would replace one dead end with a politer
    one.
    """
    from . import resolver

    found = resolver.resolve(observed, priceable=priceable)
    if not found.candidates:
        return Identification(
            observed=observed, candidates=(), confidence=Confidence.LOW,
            reason=found.mismatch or (
                "This did not match any instrument the pilot can price."),
            registry_digest=found.registry_digest)

    candidates = tuple(
        Candidate(one.symbol, one.name, one.score, one.reasons)
        for one in found.candidates)
    confidence = (Confidence.HIGH if len(candidates) == 1
                  else Confidence.MEDIUM)
    return Identification(
        observed=observed, candidates=candidates, confidence=confidence,
        reason=found.mismatch, registry_digest=found.registry_digest,
        concept_id=found.concept_id or "")


def aliases() -> Mapping[str, str]:
    """The catalog's own alias table, when it is available.

    Read rather than duplicated: `data/catalog_instruments.yaml` already maps
    the way people write things to real tickers, and a second copy here would
    be the one that stops matching.
    """
    try:
        loaded = yaml.safe_load(_MAPPING_FILE.read_text()) or {}
    except OSError:
        return {}
    return {str(k).upper(): str(v) for k, v in (loaded.get("aliases") or {}).items()}
