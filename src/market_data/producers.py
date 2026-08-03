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
