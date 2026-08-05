"""Phrase to candidates, deterministically, with the reasons attached.

    "SP500 etf"  ->  concept INDEX:SP500 + vehicle ETF
                 ->  SPY, VOO, IVV  (RSP excluded: equal-weight, not the index)
                 ->  filtered to what this deployment can price
                 ->  ranked, each with why

The model's job ended before this function started. It said a phrase could not
be placed; everything here is table lookup and arithmetic over the compiled
registry, so the same phrase produces the same candidates in the same order on
every run and in every deployment holding the same registry digest.

**Ranking is explained, not merely produced.** A candidate carries the reasons
it scored — issuer matched, tracks the concept, priceable here — because a
list of tickers in an order nobody can account for is a recommendation
wearing a resolution's clothes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

from .registry import (
    AliasObservation,
    Instrument,
    Registry,
    Vehicle,
    loaded,
    normalize,
)

#: Ranking weights. Ordered, additive and small: the point is that two people
#: reading the reasons agree about the order, not that the numbers are
#: meaningful on their own.
_ISSUER_MATCH = 40
_VEHICLE_MATCH = 20
_TRACKS_CONCEPT = 15
_PRICEABLE = 10
_CATALOG_DEFAULT = 5


@dataclass(frozen=True)
class InstrumentCandidate:
    instrument: Instrument
    score: int
    reasons: Tuple[str, ...] = ()

    @property
    def symbol(self) -> str:
        return self.instrument.symbol

    @property
    def name(self) -> str:
        return self.instrument.name


@dataclass(frozen=True)
class Resolution:
    """What a phrase turned out to mean, and what could satisfy it."""

    observed: str
    concept_id: Optional[str] = None
    concept_name: str = ""
    vehicle_requested: Optional[Vehicle] = None
    candidates: Tuple[InstrumentCandidate, ...] = ()
    registry_digest: str = ""
    mismatch: str = ""
    """Set when the user named one kind of thing and asked for another — an
    index and a fund. The reason "no price history for SPX" never gave."""

    @property
    def certain(self) -> bool:
        return len(self.candidates) == 1

    @property
    def unresolved(self) -> bool:
        return not self.candidates


#: Instruments that track an index without reproducing it. RSP holds the S&P
#: 500 equally weighted, which is a different strategy with the same
#: constituents — offering it as "the S&P 500 fund" would answer a question
#: about exposure with a fund that deliberately differs from it.
_NOT_A_PLAIN_TRACKER = {"RSP"}


def _alias_for(registry: Registry, phrase: str) -> Optional[AliasObservation]:
    normalized = normalize(phrase)
    for one in registry.aliases:
        if one.normalized_phrase == normalized:
            return one
    return None


def resolve(phrase: str, *, priceable: Sequence[str] = (),
            registry: Optional[Registry] = None) -> Resolution:
    """What did the user mean, and what could deliver it?"""
    registry = registry or loaded()
    normalized = normalize(phrase)
    observation = _alias_for(registry, phrase)
    target = registry.phrase_index.get(normalized)

    if target is None:
        return Resolution(observed=phrase, registry_digest=registry.digest)

    kind, target_id = target

    # A phrase that names one instrument outright is not a question.
    if kind == "INSTRUMENT":
        one = registry.instruments.get(target_id)
        if one is None:
            return Resolution(observed=phrase, registry_digest=registry.digest)
        return Resolution(
            observed=phrase, registry_digest=registry.digest,
            candidates=(InstrumentCandidate(
                one, _PRICEABLE + _CATALOG_DEFAULT,
                ("named directly in the description",)),))

    concept = registry.concepts.get(target_id)
    if concept is None:
        return Resolution(observed=phrase, registry_digest=registry.digest)

    vehicle = observation.vehicle_hint if observation else None
    issuer = observation.issuer_hint if observation else None

    mismatch = ""
    if concept.kind.value == "INDEX":
        mismatch = (
            f"{concept.canonical_name} is an index — a measurement, not "
            f"something you can buy."
            + (" You asked for a fund, so this is which fund."
               if vehicle else " These funds track it."))

    available = set(priceable)
    candidates = []
    for one in registry.tracking(concept.concept_id):
        if one.symbol in _NOT_A_PLAIN_TRACKER and not issuer:
            continue
        if available and one.symbol not in available:
            continue
        score = 0
        reasons = []
        if issuer and issuer.lower() in one.issuer.lower():
            score += _ISSUER_MATCH
            reasons.append(f"issued by {one.issuer}, which you named")
        if vehicle and one.instrument_type == vehicle.value:
            score += _VEHICLE_MATCH
            reasons.append(f"is an {one.instrument_type}, which you asked for")
        if one.tracks_index == concept.concept_id:
            score += _TRACKS_CONCEPT
            reasons.append(f"tracks {concept.canonical_name}")
        elif one.delivers_exposure == concept.concept_id:
            score += _TRACKS_CONCEPT
            reasons.append(f"gives {concept.canonical_name} exposure")
        if not available or one.symbol in available:
            score += _PRICEABLE
            reasons.append("can be priced in this deployment")
        if concept.default_instrument == one.instrument_id:
            score += _CATALOG_DEFAULT
            reasons.append("the catalogue's default for this concept")
        candidates.append(InstrumentCandidate(one, score, tuple(reasons)))

    # Symbol is the tie-break, never file position: an editorial reshuffle of
    # the YAML must not silently change which fund is offered first.
    candidates.sort(key=lambda one: (-one.score, one.symbol))

    return Resolution(
        observed=phrase, concept_id=concept.concept_id,
        concept_name=concept.canonical_name, vehicle_requested=vehicle,
        candidates=tuple(candidates), registry_digest=registry.digest,
        mismatch=mismatch)
