"""Every persisted artifact that could carry a market-derived number.

The reader inventory closed one class: no production request obtains market data
outside the shared gate. This closes the other: no persisted figure sits outside
the provenance graph.

    DIRECT           the artifact stores its own `MarketDataProvenance`
    REFERENCED       it cites another persisted artifact whose provenance is
                     authoritative — the chain resolves, it does not copy
    NOT_APPLICABLE   it cannot contain a market-derived value
    LEGACY_UNKNOWN   it may, and provenance was not recorded

**Omission is not `NOT_APPLICABLE`.** A type absent from this registry fails the
structural check rather than being assumed harmless, which is the difference
between an enumerated class and a list of the ones somebody thought of.

**Provenance is not duplicated into every consumer.** A worksheet block cites a
run; the run holds the provenance. Copying the object into each block would give
one figure several sources of truth, and they would diverge at the first
correction.

**A priced field does not make a type market-derived.** An observed vest carries
a value the *payroll system* reported; a reconciliation may compare reported
quantities, priced values, or both. Classifying the whole type as market-derived
would make an observed payroll figure appear to depend on a market snapshot it
never touched, so those types declare a pricing basis per record instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple


class ProvenanceOwnership(str, Enum):
    DIRECT = "DIRECT"
    REFERENCED = "REFERENCED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    LEGACY_UNKNOWN = "LEGACY_UNKNOWN"


class PricingBasis(str, Enum):
    """Where a stored value's number came from.

    Kept separate from provenance because they answer different questions.
    Provenance says *which market data*; this says *whether market data was
    involved at all*.
    """

    NOT_APPLICABLE = "NOT_APPLICABLE"
    """No monetary value, or a quantity with no price attached."""

    OBSERVED_VALUE = "OBSERVED_VALUE"
    """A figure someone reported — a payroll statement, a broker confirmation.
    It has a source, and that source is not a market snapshot."""

    MARKET_SNAPSHOT = "MARKET_SNAPSHOT"
    """Priced from market data. Requires provenance."""


@dataclass(frozen=True)
class Producer:
    """One persisted type, and where its provenance lives."""

    table: str
    ownership: ProvenanceOwnership
    reason: str
    provenance_path: Optional[str] = None
    """For DIRECT: where in the payload the provenance sits."""

    reference_path: Optional[str] = None
    """For REFERENCED: where the citation of the authoritative artifact sits."""

    reference_table: Optional[str] = None
    pricing_basis_path: Optional[str] = None
    """For types whose records differ: where each record states its basis."""

    def to_json(self) -> dict:
        return {"table": self.table, "ownership": self.ownership.value,
                "reason": self.reason,
                "provenance_path": self.provenance_path,
                "reference_path": self.reference_path,
                "reference_table": self.reference_table,
                "pricing_basis_path": self.pricing_basis_path}


PRODUCERS: Mapping[str, Producer] = {
    one.table: one for one in (
        Producer(
            table="plan_run", ownership=ProvenanceOwnership.DIRECT,
            provenance_path="result.market_data",
            reason="The figures. A run is where market data becomes a number, "
                   "so it is where the record of which data belongs."),
        Producer(
            table="market_data_access_event",
            ownership=ProvenanceOwnership.DIRECT,
            provenance_path="provenance_digest",
            reason="Not a figure, but the record of the delivery a figure came "
                   "from — it names its own provenance by digest rather than "
                   "citing another artifact's, because it is the artifact a "
                   "run cites. Classifying it REFERENCED would point at the "
                   "run, and the run points here: the pair would each claim "
                   "the other held the answer."),
        Producer(
            table="worksheet", ownership=ProvenanceOwnership.REFERENCED,
            reference_path="payload.benchmark_run_refs",
            reference_table="plan_run",
            reason="A worksheet presents runs; it does not compute. Copying "
                   "provenance into each block would give one figure several "
                   "sources of truth, and they would diverge at the first "
                   "correction."),
        Producer(
            table="worksheet_proposal", ownership=ProvenanceOwnership.REFERENCED,
            reference_path="result_runs", reference_table="plan_run",
            reason="An accepted proposal cites the runs it produced. The "
                   "provenance is theirs."),
        Producer(
            table="planned_event", ownership=ProvenanceOwnership.NOT_APPLICABLE,
            pricing_basis_path="payload.pricing_basis",
            reason="An expectation stated by the user from a grant document. "
                   "`expected_value` is what they said it would be worth, not "
                   "a price this system looked up."),
        Producer(
            table="observed_event", ownership=ProvenanceOwnership.NOT_APPLICABLE,
            pricing_basis_path="payload.pricing_basis",
            reason="A report. `value` came from a payroll statement or broker "
                   "confirmation — it has a source, and that source is not a "
                   "market snapshot. Classifying it as market-derived would "
                   "make a reported figure appear to depend on data it never "
                   "touched."),
        Producer(
            table="event_reconciliation",
            ownership=ProvenanceOwnership.NOT_APPLICABLE,
            pricing_basis_path="payload.pricing_basis",
            reason="Derived by comparing an expectation with a report, both of "
                   "which are declarations. A reconciliation that ever prices "
                   "a difference must declare MARKET_SNAPSHOT on that record "
                   "and carry a run reference; today none does."),
        Producer(
            table="plan", ownership=ProvenanceOwnership.NOT_APPLICABLE,
            reason="A compiled scenario specification. It states what to do, "
                   "and holds no result."),
        Producer(
            table="plan_migration", ownership=ProvenanceOwnership.NOT_APPLICABLE,
            reason="A recompiled scenario specification and the authorisation "
                   "for it. Like `plan`, it states what to do and holds no "
                   "result — the provenance question belongs to the runs on "
                   "either side of it, which carry their own."),
        Producer(
            table="run_invalidation",
            ownership=ProvenanceOwnership.NOT_APPLICABLE,
            reason="A judgement about a run, not a figure. It says that a "
                   "stored result must not be read as a strategy result and "
                   "why, and holds no price, value or quantity of its own. "
                   "The provenance question belongs to the run it names, "
                   "which keeps its own — and keeps it precisely so a "
                   "withdrawn figure stays checkable."),
        Producer(
            table="worksheet_intent", ownership=ProvenanceOwnership.NOT_APPLICABLE,
            reason="A classified request. It records what was asked before "
                   "anything ran."),
        Producer(
            table="proposal", ownership=ProvenanceOwnership.NOT_APPLICABLE,
            reason="A forward-tracking artifact describing a suggested action. "
                   "It carries no computed figure."),
        Producer(
            table="observation", ownership=ProvenanceOwnership.NOT_APPLICABLE,
            reason="A mission observation record, sourced from its own "
                   "evidence rather than from market data."),
        Producer(
            table="confirmation_event", ownership=ProvenanceOwnership.NOT_APPLICABLE,
            reason="Confirmation-screen telemetry. It records an interaction, "
                   "not a figure."),
    )
}


def unclassified(tables: Sequence[str]) -> Tuple[str, ...]:
    """Tables the schema reports and this registry does not classify."""
    return tuple(sorted(set(tables) - set(PRODUCERS)))


def direct_producers() -> Tuple[Producer, ...]:
    return tuple(one for one in PRODUCERS.values()
                 if one.ownership is ProvenanceOwnership.DIRECT)


def referencing_producers() -> Tuple[Producer, ...]:
    return tuple(one for one in PRODUCERS.values()
                 if one.ownership is ProvenanceOwnership.REFERENCED)


def dig(payload: Any, path: str) -> Any:
    """Follow a dotted path into a stored payload, or return None."""
    current = payload
    for part in path.split("."):
        if part in ("result", "payload") and not isinstance(current, dict):
            return None
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


class CallerKind(str, Enum):
    """What a production caller of `record_run` is entitled to store.

    The distinction that matters is the third one. Without it a live path can
    satisfy the write guard by labelling its own omission as legacy — the
    record would say "nobody recorded this" when in fact this code declined to,
    and the two are indistinguishable afterwards.
    """

    MARKET_DERIVED = "MARKET_DERIVED"
    """Must supply a RECORDED provenance that identifies its data and carries
    an allowed access decision."""

    NON_MARKET = "NON_MARKET"
    """Must supply NOT_APPLICABLE explicitly."""

    LEGACY_IMPORT = "LEGACY_IMPORT"
    """May supply NOT_RECORDED, and only on an import or migration path. A
    live request may never choose this."""


@dataclass(frozen=True)
class RunCaller:
    """One production path that persists a run."""

    name: str
    module: str
    kind: CallerKind
    reason: str

    @property
    def may_claim_legacy(self) -> bool:
        return self.kind is CallerKind.LEGACY_IMPORT


#: Every production caller of `record_run`. Compared against the call graph by
#: `tests/test_run_callers.py`, so a new one fails until it is classified.
#:
#: `apply_import` is deliberately absent. It was listed here first, and the
#: call-graph scan showed it never calls `record_run` at all — the transfer
#: tool writes rows with raw SQL, below the store. That is correct for a
#: migration (it must be able to carry a legacy row through unchanged) and it
#: means the LEGACY_IMPORT kind currently classifies nothing. The kind stays,
#: because the distinction it draws is what stops a live caller labelling its
#: own omission as legacy, and `TestNoLiveCallerMayClaimLegacyAbsence` asserts
#: no live caller has taken it.
RUN_CALLERS: Mapping[str, RunCaller] = {
    one.name: one for one in (
        RunCaller(
            name="generate", module="src/workspace/generate.py",
            kind=CallerKind.MARKET_DERIVED,
            reason="Persists the run behind a saved scenario. The figures come "
                   "from a resolved frame, so the record of which frame is "
                   "required rather than optional."),
        RunCaller(
            name="main", module="src/workspace/migrate_plan.py",
            kind=CallerKind.MARKET_DERIVED,
            reason="Persists the replacement run for a plan recompiled under a "
                   "newer compiler. Operator-invoked rather than reached by a "
                   "request, and it writes exactly the same kind of artifact — "
                   "so it carries the same obligation to say which frame "
                   "produced the figures. A caller that persists runs outside "
                   "the request path is the one most likely to be forgotten by "
                   "a rule written for the request path."),
        RunCaller(
            name="_apply", module="src/workspace/apply.py",
            kind=CallerKind.MARKET_DERIVED,
            reason="Persists each candidate run produced by an accepted "
                   "proposal. Named `_apply` rather than `accept`: `accept` "
                   "validates and delegates, and the call graph is what says "
                   "which function actually writes. Every candidate is an "
                   "independent artifact carrying the access it was computed "
                   "from, so a worksheet citing three of them can say which "
                   "data each used."),
    )
}


def unclassified_callers(found: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(set(found) - set(RUN_CALLERS)))


def live_callers() -> Tuple[RunCaller, ...]:
    """Callers a user request can reach. None of them may claim legacy."""
    return tuple(one for one in RUN_CALLERS.values()
                 if one.kind is not CallerKind.LEGACY_IMPORT)
