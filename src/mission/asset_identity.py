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

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple



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
