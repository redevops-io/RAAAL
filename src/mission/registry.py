"""The instrument registry: authored as YAML, compiled, then read-only.

The flat alias map was answering seven questions at once — what phrase was
used, what kind of thing was meant, which securities satisfy it, which are
available here, which comes first, and what the user chose. Those are separate
facts with separate lifetimes, and a `phrase -> ticker` dictionary can hold
none of them.

    data/instruments/*.yaml  ->  compile()  ->  Registry(digest=...)

**Compiled, not read live.** An uncompiled dictionary consulted at runtime has
no version, so a plan cannot say which interpretation produced it, and no
validation, so a dangling relationship is discovered by whoever hits it. The
compiler refuses a registry that does not hold together, and stamps a digest a
stored plan can pin.

Identity is `instrument_id`, never the ticker. Tickers collide across venues,
get reissued after a delisting, and differ between data providers — SPY is one
label for one instrument, and `provider_symbols` is how each vendor spells it.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple

import yaml

SOURCE_DIR = pathlib.Path(__file__).resolve().parents[2] / "data" / "instruments"


class ConceptKind(str, Enum):
    INDEX = "INDEX"
    ASSET_CLASS = "ASSET_CLASS"


class Vehicle(str, Enum):
    ETF = "ETF"
    FUND = "FUND"
    ANY = "ANY"


class AliasSource(str, Enum):
    CURATED = "CURATED"
    """Written by a person and reviewed."""

    PROPOSED = "PROPOSED"
    """Observed from a user's clarification and not yet promoted. Never used
    for resolution — one user's answer must not change everyone's reading."""


class RegistryError(ValueError):
    """The registry does not hold together and will not be compiled."""


@dataclass(frozen=True)
class Instrument:
    instrument_id: str
    symbol: str
    name: str
    instrument_type: str
    exchange: str
    currency: str
    issuer: str
    provider_symbols: Mapping[str, str] = field(default_factory=dict)
    tracks_index: Optional[str] = None
    delivers_exposure: Optional[str] = None


@dataclass(frozen=True)
class AssetConcept:
    concept_id: str
    kind: ConceptKind
    canonical_name: str
    aliases: Tuple[str, ...] = ()
    #: Which instrument to offer first when the phrase itself does not decide.
    default_instrument: Optional[str] = None


@dataclass(frozen=True)
class AliasObservation:
    normalized_phrase: str
    target_kind: str
    target_id: str
    source: AliasSource = AliasSource.CURATED
    confidence: str = "HIGH"
    vehicle_hint: Optional[Vehicle] = None
    issuer_hint: Optional[str] = None


@dataclass(frozen=True)
class Registry:
    version: int
    digest: str
    instruments: Mapping[str, Instrument]
    concepts: Mapping[str, AssetConcept]
    aliases: Tuple[AliasObservation, ...]
    #: normalized phrase -> (kind, id). Built once, not searched linearly.
    phrase_index: Mapping[str, Tuple[str, str]] = field(default_factory=dict)

    def instrument_by_symbol(self, symbol: str) -> Optional[Instrument]:
        for one in self.instruments.values():
            if one.symbol == symbol:
                return one
        return None

    def tracking(self, concept_id: str) -> Tuple[Instrument, ...]:
        """Instruments that deliver a concept, in a stable order.

        Sorted by symbol rather than by file position. Deriving order from
        where a line happens to sit in YAML makes an editorial reshuffle a
        behaviour change nobody intended.
        """
        return tuple(sorted(
            (one for one in self.instruments.values()
             if one.tracks_index == concept_id
             or one.delivers_exposure == concept_id),
            key=lambda one: one.symbol))


def normalize(phrase: str) -> str:
    """One spelling for lookup.

    Case, punctuation and the parenthetical a model adds — "SP500 etf (no
    literal ticker given)" — are noise around the thing the user typed.
    """
    without_notes = re.sub(r"\([^)]*\)", " ", phrase or "")
    cleaned = re.sub(r"[^a-z0-9&\s-]", " ", without_notes.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _load(directory: pathlib.Path, name: str) -> dict:
    path = directory / name
    if not path.exists():
        raise RegistryError(f"{path} is missing")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def compile_registry(source_dir: Optional[pathlib.Path] = None) -> Registry:
    """Validate the authored sources and produce the runtime artifact.

    Every failure here is one that would otherwise surface as a wrong answer:
    a duplicate identity means two instruments share one, a dangling
    relationship means a concept resolves to nothing, and a phrase claimed by
    two targets means the resolution depends on iteration order.
    """
    # The directory is a parameter, not a global reassignment. Setting the
    # module-level SOURCE_DIR made compiling one registry redirect every later
    # load in the process — a test registry silently became the production
    # one, which is the shape of bug that only shows up in whatever ran next.
    directory = source_dir if source_dir is not None else SOURCE_DIR

    concepts: Dict[str, AssetConcept] = {}
    for raw in _load(directory, "concepts.yaml").get("concepts", []):
        concept_id = raw["concept_id"]
        if concept_id in concepts:
            raise RegistryError(f"duplicate concept {concept_id}")
        concepts[concept_id] = AssetConcept(
            concept_id=concept_id,
            kind=ConceptKind(raw["kind"]),
            canonical_name=raw["canonical_name"],
            aliases=tuple(normalize(a) for a in raw.get("aliases", ())),
            default_instrument=raw.get("default_instrument"),
        )

    instruments: Dict[str, Instrument] = {}
    for raw in _load(directory, "instruments.yaml").get("instruments", []):
        instrument_id = raw["instrument_id"]
        if instrument_id in instruments:
            raise RegistryError(f"duplicate instrument {instrument_id}")
        instruments[instrument_id] = Instrument(
            instrument_id=instrument_id,
            symbol=raw["symbol"], name=raw["name"],
            instrument_type=raw["instrument_type"], exchange=raw["exchange"],
            currency=raw["currency"], issuer=raw.get("issuer", "unknown"),
            provider_symbols=dict(raw.get("provider_symbols", {})),
            tracks_index=raw.get("tracks_index"),
            delivers_exposure=raw.get("delivers_exposure"),
        )

    # No dangling relationships. An ETF pointing at a concept nobody defined
    # is a candidate list that silently comes back empty.
    for concept in concepts.values():
        if (concept.default_instrument
                and concept.default_instrument not in instruments):
            raise RegistryError(
                f"{concept.concept_id} defaults to unknown instrument "
                f"{concept.default_instrument}")

    for one in instruments.values():
        for reference in (one.tracks_index, one.delivers_exposure):
            if reference and reference not in concepts:
                raise RegistryError(
                    f"{one.instrument_id} references unknown concept {reference}")

    # Provider symbols unique per provider: two instruments claiming one
    # vendor symbol makes a price lookup ambiguous.
    seen_provider: Dict[Tuple[str, str], str] = {}
    for one in instruments.values():
        for provider, symbol in one.provider_symbols.items():
            key = (provider, symbol)
            if key in seen_provider:
                raise RegistryError(
                    f"{provider}:{symbol} claimed by {seen_provider[key]} "
                    f"and {one.instrument_id}")
            seen_provider[key] = one.instrument_id

    aliases: list[AliasObservation] = []
    for raw in _load(directory, "aliases.yaml").get("aliases", []):
        aliases.append(AliasObservation(
            normalized_phrase=normalize(raw["phrase"]),
            target_kind=raw["target_kind"], target_id=raw["target_id"],
            source=AliasSource(raw.get("source", "CURATED")),
            confidence=raw.get("confidence", "HIGH"),
            vehicle_hint=Vehicle(raw["vehicle_hint"])
            if raw.get("vehicle_hint") else None,
            issuer_hint=raw.get("issuer_hint"),
        ))

    phrase_index: Dict[str, Tuple[str, str]] = {}
    for concept in concepts.values():
        for phrase in concept.aliases:
            claim = ("CONCEPT", concept.concept_id)
            existing = phrase_index.get(phrase)
            # Two spellings collapsing to one — "spx" and "^spx" — is not a
            # conflict when they name the same thing. Only a different target
            # makes resolution depend on iteration order.
            if existing is not None and existing != claim:
                raise RegistryError(
                    f"phrase {phrase!r} claimed by {existing[1]} "
                    f"and {concept.concept_id}")
            phrase_index[phrase] = claim
    for observation in aliases:
        if observation.source is not AliasSource.CURATED:
            continue
        target = (observation.target_kind, observation.target_id)
        if observation.target_kind == "CONCEPT" and \
                observation.target_id not in concepts:
            raise RegistryError(
                f"alias {observation.normalized_phrase!r} targets unknown "
                f"concept {observation.target_id}")
        if observation.target_kind == "INSTRUMENT" and \
                observation.target_id not in instruments:
            raise RegistryError(
                f"alias {observation.normalized_phrase!r} targets unknown "
                f"instrument {observation.target_id}")
        # A faceted alias may refine a concept alias; it may not contradict a
        # different target.
        existing = phrase_index.get(observation.normalized_phrase)
        if existing and existing != target:
            raise RegistryError(
                f"phrase {observation.normalized_phrase!r} claimed by "
                f"{existing[1]} and {observation.target_id}")
        phrase_index[observation.normalized_phrase] = target

    # Over the content, not the keys.
    #
    # The first version hashed `sorted(concepts)` and `sorted(instruments)` —
    # the identifiers alone. Changing a concept's default instrument, which is
    # precisely the drift a pinned digest exists to detect, moved no key and
    # left the digest identical. A fingerprint that misses the most likely
    # change is a fingerprint that certifies nothing.
    payload = json.dumps({
        "concepts": [
            {"id": c.concept_id, "kind": c.kind.value,
             "name": c.canonical_name, "aliases": sorted(c.aliases),
             "default": c.default_instrument}
            for c in sorted(concepts.values(), key=lambda c: c.concept_id)],
        "instruments": [
            {"id": i.instrument_id, "symbol": i.symbol, "name": i.name,
             "type": i.instrument_type, "exchange": i.exchange,
             "currency": i.currency, "issuer": i.issuer,
             "providers": dict(sorted(i.provider_symbols.items())),
             "tracks": i.tracks_index, "delivers": i.delivers_exposure}
            for i in sorted(instruments.values(),
                            key=lambda i: i.instrument_id)],
        "aliases": [
            {"phrase": a.normalized_phrase, "kind": a.target_kind,
             "target": a.target_id, "source": a.source.value,
             "vehicle": a.vehicle_hint.value if a.vehicle_hint else None,
             "issuer": a.issuer_hint}
            for a in sorted(aliases, key=lambda a: (a.normalized_phrase,
                                                    a.target_id))],
    }, sort_keys=True).encode()
    digest = "reg1:" + hashlib.sha256(payload).hexdigest()[:32]

    return Registry(version=1, digest=digest, instruments=instruments,
                    concepts=concepts, aliases=tuple(aliases),
                    phrase_index=phrase_index)


_COMPILED: Optional[Registry] = None


def loaded() -> Registry:
    """The compiled registry, once per process."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = compile_registry()
    return _COMPILED
