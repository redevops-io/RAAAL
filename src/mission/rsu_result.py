"""What an RSU result must carry with it, whatever the figures say.

    Anything that can materially qualify, delay, invalidate or reinterpret a
    result travels on the result itself. It cannot remain only in an
    engine-local diagnostic.

A warning computed inside a function and never returned is a warning nobody
receives. Every stage of this pipeline now produces one — unpriced arrivals,
unsettled dispositions, unfilled allocation targets, missing concentration
prices, benchmark verdicts — and each was, until now, available at the point of
computation and nowhere afterwards.

**Three things kept apart**, because they lead a reader somewhere different:

    LIMITATION          a known boundary of the model
    UNSETTLED           the model knows what should happen; it has not yet
    DATA_GAP            the computation could not be performed
    EXECUTION_FAILURE   it was attempted and did not complete
    PARTIAL_RESULT      some of it completed

"Capital gains are not modelled", "the sale is still pending" and "the price was
missing" are all "not fully modelled" only if you stop reading.

**Absence is never completeness.** A result with no RSU context is not a clean
result; it is a result that never declared one. `Presentability` is derived from
what was established, not from the lack of complaints.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Bumped when the presentability rules change. A stored result says which rules
#: judged it rather than being re-judged by whatever is current.
CONTEXT_VERSION = "rsu-result-context@1"


class DiagnosticKind(str, Enum):
    LIMITATION = "LIMITATION"
    UNSETTLED = "UNSETTLED"
    DATA_GAP = "DATA_GAP"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"
    PARTIAL_RESULT = "PARTIAL_RESULT"


class Presentability(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    BLOCKED = "BLOCKED"


class ScopeStatus(str, Enum):
    DECLARED = "DECLARED"
    NOT_DECLARED = "NOT_DECLARED"
    """No RSU context was attached. Not the same as a context reporting nothing
    wrong — one is a clean result, the other is an unexamined one."""


#: The sentence that travels with every performance figure from an RSU run.
POST_WITHHOLDING_BASIS = (
    "Results measure the account value actually delivered after employer share "
    "withholding. They do not represent gross compensation or final tax "
    "liability.")

PENDING_DISPOSITION_NOTE = (
    "The sale was instructed but had not yet filled. Allocation and realized "
    "concentration are therefore incomplete.")

UNFILLED_TARGET_NOTE = (
    "Part of the proceeds remained in cash because one or more requested "
    "purchases could not be completed.")

HOUSEHOLD_SCOPE_NOTE = (
    "Concentration is measured within the modelled portfolio only; external "
    "assets and liabilities are excluded.")


@dataclass(frozen=True)
class Diagnostic:
    """One thing a reader needs, with the kind that says what to do about it."""

    kind: DiagnosticKind
    code: str
    detail: str
    refs: Sequence[str] = ()

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind.value, "code": self.code,
                "detail": self.detail, "refs": list(self.refs)}


# --- the destination registry ----------------------------------------------
#
# Every material engine diagnostic has exactly one place it lands. The registry
# is what `tests/test_rsu_result.py` checks an *independently discovered*
# inventory against, so a new engine diagnostic with nowhere to go fails rather
# than evaporating.

DESTINATIONS: Mapping[str, str] = {
    "unpriced_in_kind_arrivals": "vest_accounting.unpriced_arrivals",
    "cash_remainder": "vest_accounting.cash_remainder",
    "unsettled_report": "disposition.unsettled_report",
    "failed_instructions": "disposition.failed_instructions",
    "pending_instructions": "disposition.pending_instructions",
    "unfilled_targets": "allocation.unfilled_targets",
    "residual_cash": "allocation.residual_cash",
    "unallocated_weight": "allocation.unallocated_weight",
    "missing_prices": "concentration.missing_prices",
    "unresolved_inputs": "concentration.unresolved_inputs",
    "excluded_components": "concentration.excluded_components",
    "projected_post_sale_concentration": "concentration.projected",
    "realized_concentration": "concentration.realized",
    "unchecked_dimensions": "comparisons.verdict_rows",
    "differing_dimensions": "comparisons.verdict_rows",
    "verdict_rows": "comparisons.verdict_rows",
}


@dataclass(frozen=True)
class VestAccountingContext:
    basis: str = "POST_WITHHOLDING_ACCOUNT_VALUE"
    gross_vest_value: Optional[float] = None
    withheld_value: Optional[float] = None
    delivered_value: Optional[float] = None
    cash_remainder: Optional[float] = None
    unpriced_arrivals: Sequence[Mapping[str, Any]] = ()

    def to_json(self) -> Dict[str, Any]:
        return {"basis": self.basis, "basis_note": POST_WITHHOLDING_BASIS,
                "gross_vest_value": self.gross_vest_value,
                "withheld_value": self.withheld_value,
                "delivered_value": self.delivered_value,
                "cash_remainder": self.cash_remainder,
                "unpriced_arrivals": [dict(one) for one in self.unpriced_arrivals]}


@dataclass(frozen=True)
class DispositionContext:
    status: str = ""
    pending_instructions: Sequence[Mapping[str, Any]] = ()
    failed_instructions: Sequence[Mapping[str, Any]] = ()
    unsettled_report: Sequence[Mapping[str, Any]] = ()

    def to_json(self) -> Dict[str, Any]:
        return {"status": self.status,
                "pending_instructions": [dict(o) for o in self.pending_instructions],
                "failed_instructions": [dict(o) for o in self.failed_instructions],
                "unsettled_report": [dict(o) for o in self.unsettled_report]}


@dataclass(frozen=True)
class AllocationContext:
    requested_targets: Mapping[str, float] = field(default_factory=dict)
    executed_targets: Mapping[str, float] = field(default_factory=dict)
    unfilled_targets: Sequence[Mapping[str, Any]] = ()
    residual_cash: Optional[float] = None
    unallocated_weight: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        return {"requested_targets": dict(self.requested_targets),
                "executed_targets": dict(self.executed_targets),
                "unfilled_targets": [dict(o) for o in self.unfilled_targets],
                "residual_cash": self.residual_cash,
                "unallocated_weight": self.unallocated_weight}


@dataclass(frozen=True)
class ConcentrationContext:
    """Projected and realized stay apart all the way to the worksheet.

    A projected figure is what a plan expected; a realized one is what a fill
    produced. Only the second can say the cap was reached.
    """

    current: Optional[float] = None
    target: Optional[float] = None
    projected: Optional[float] = None
    realized: Optional[float] = None
    missing_prices: Sequence[str] = ()
    unresolved_inputs: Sequence[str] = ()
    denominator_scope: Sequence[str] = ()
    excluded_components: Sequence[str] = ()

    @property
    def cap_achieved(self) -> Optional[bool]:
        """None while only a projection exists. A plan cannot report a target
        met on the strength of an order it placed."""
        if self.realized is None or self.target is None:
            return None
        return self.realized <= self.target + 1e-9

    def to_json(self) -> Dict[str, Any]:
        return {"current": self.current, "target": self.target,
                "projected": self.projected, "realized": self.realized,
                "cap_achieved": self.cap_achieved,
                "missing_prices": list(self.missing_prices),
                "unresolved_inputs": list(self.unresolved_inputs),
                "denominator_scope": list(self.denominator_scope),
                "excluded_components": list(self.excluded_components),
                "scope_note": HOUSEHOLD_SCOPE_NOTE}


@dataclass(frozen=True)
class ComparisonContext:
    verdict_rows: Sequence[Mapping[str, Any]] = ()

    @property
    def unchecked_present(self) -> bool:
        return any(row.get("unchecked_dimensions") for row in self.verdict_rows)

    def to_json(self) -> Dict[str, Any]:
        return {"verdict_rows": [dict(row) for row in self.verdict_rows],
                "unchecked_present": self.unchecked_present}


@dataclass(frozen=True)
class RSUResultContext:
    """Everything that qualifies an RSU figure, carried by the figure."""

    vest_accounting: VestAccountingContext = field(
        default_factory=VestAccountingContext)
    disposition: DispositionContext = field(default_factory=DispositionContext)
    allocation: AllocationContext = field(default_factory=AllocationContext)
    concentration: ConcentrationContext = field(
        default_factory=ConcentrationContext)
    comparisons: ComparisonContext = field(default_factory=ComparisonContext)
    modelling_scope: Mapping[str, Sequence[str]] = field(default_factory=dict)
    context_version: str = CONTEXT_VERSION

    # ---- derived --------------------------------------------------------

    def diagnostics(self) -> List[Diagnostic]:
        """Every qualification, typed. Order is stable for display."""
        out: List[Diagnostic] = []

        for arrival in self.vest_accounting.unpriced_arrivals:
            out.append(Diagnostic(
                kind=DiagnosticKind.DATA_GAP, code="unpriced_arrival",
                detail=(f"{arrival.get('asset')} could not be priced on its "
                        f"arrival session, so no shares and no flow were "
                        f"recorded for it"),
                refs=(str(arrival.get("source_ref", "")),)))

        for one in self.disposition.failed_instructions:
            out.append(Diagnostic(
                kind=DiagnosticKind.EXECUTION_FAILURE,
                code="disposition_failed",
                detail=str(one.get("why") or "the sale did not complete"),
                refs=(str(one.get("instruction_id", "")),)))

        for one in self.disposition.pending_instructions:
            out.append(Diagnostic(
                kind=DiagnosticKind.UNSETTLED, code="disposition_pending",
                detail=PENDING_DISPOSITION_NOTE,
                refs=(str(one.get("instruction_id", "")),)))

        for one in self.allocation.unfilled_targets:
            out.append(Diagnostic(
                kind=DiagnosticKind.PARTIAL_RESULT, code="target_unfilled",
                detail=f"{one.get('asset')}: {one.get('why')}",
                refs=(str(one.get("asset", "")),)))

        if self.concentration.missing_prices:
            out.append(Diagnostic(
                kind=DiagnosticKind.DATA_GAP, code="concentration_uncomputable",
                detail=("concentration could not be measured because these "
                        "holdings had no price: "
                        + ", ".join(self.concentration.missing_prices)),
                refs=tuple(self.concentration.missing_prices)))

        if self.concentration.projected is not None \
                and self.concentration.realized is None:
            out.append(Diagnostic(
                kind=DiagnosticKind.UNSETTLED, code="concentration_projected",
                detail=("this concentration is PROJECTED from the planned sale; "
                        "the realized figure is not yet available")))

        if self.comparisons.unchecked_present:
            out.append(Diagnostic(
                kind=DiagnosticKind.PARTIAL_RESULT,
                code="comparison_not_isolated",
                detail=("one or more comparisons are shown without the strategy "
                        "effect being isolated")))

        for entry in self.modelling_scope.get("out_of_scope", ()):
            out.append(Diagnostic(kind=DiagnosticKind.LIMITATION,
                                  code="out_of_scope", detail=str(entry)))
        return out

    @property
    def scope_status(self) -> ScopeStatus:
        return (ScopeStatus.DECLARED if self.modelling_scope
                else ScopeStatus.NOT_DECLARED)

    @property
    def presentability(self) -> Presentability:
        """Derived from what was established, never from a lack of complaints.

        BLOCKED means a figure would misrepresent the run: a vest that could not
        be priced, a required sale that never filled, or a cap claimed without a
        realized measurement. PARTIAL means the figure stands with something
        outstanding beside it.
        """
        if self.scope_status is ScopeStatus.NOT_DECLARED:
            # An undeclared scope is not a clean result. It is one nobody
            # examined, and the two must never render alike.
            return Presentability.BLOCKED

        kinds = {one.kind for one in self.diagnostics()}
        if DiagnosticKind.DATA_GAP in kinds \
                or DiagnosticKind.EXECUTION_FAILURE in kinds:
            return Presentability.BLOCKED

        # A cap that was targeted but never realized cannot be reported as met.
        if self.concentration.target is not None \
                and self.concentration.realized is None:
            return Presentability.BLOCKED

        if kinds - {DiagnosticKind.LIMITATION}:
            return Presentability.PARTIAL
        return Presentability.COMPLETE

    def to_json(self) -> Dict[str, Any]:
        return {
            "context_version": self.context_version,
            "vest_accounting": self.vest_accounting.to_json(),
            "disposition": self.disposition.to_json(),
            "allocation": self.allocation.to_json(),
            "concentration": self.concentration.to_json(),
            "comparisons": self.comparisons.to_json(),
            "modelling_scope": {k: list(v)
                                for k, v in self.modelling_scope.items()},
            "diagnostics": [one.to_json() for one in self.diagnostics()],
            "scope_status": self.scope_status.value,
            "presentability": self.presentability.value,
        }


def from_json(payload: Mapping[str, Any]) -> RSUResultContext:
    """Rebuild a stored context. Derived fields are recomputed, never read."""
    vest = payload.get("vest_accounting") or {}
    disposition = payload.get("disposition") or {}
    allocation = payload.get("allocation") or {}
    concentration = payload.get("concentration") or {}
    comparisons = payload.get("comparisons") or {}

    return RSUResultContext(
        vest_accounting=VestAccountingContext(
            basis=vest.get("basis", "POST_WITHHOLDING_ACCOUNT_VALUE"),
            gross_vest_value=vest.get("gross_vest_value"),
            withheld_value=vest.get("withheld_value"),
            delivered_value=vest.get("delivered_value"),
            cash_remainder=vest.get("cash_remainder"),
            unpriced_arrivals=tuple(vest.get("unpriced_arrivals") or ())),
        disposition=DispositionContext(
            status=disposition.get("status", ""),
            pending_instructions=tuple(
                disposition.get("pending_instructions") or ()),
            failed_instructions=tuple(
                disposition.get("failed_instructions") or ()),
            unsettled_report=tuple(disposition.get("unsettled_report") or ())),
        allocation=AllocationContext(
            requested_targets=dict(allocation.get("requested_targets") or {}),
            executed_targets=dict(allocation.get("executed_targets") or {}),
            unfilled_targets=tuple(allocation.get("unfilled_targets") or ()),
            residual_cash=allocation.get("residual_cash"),
            unallocated_weight=allocation.get("unallocated_weight")),
        concentration=ConcentrationContext(
            current=concentration.get("current"),
            target=concentration.get("target"),
            projected=concentration.get("projected"),
            realized=concentration.get("realized"),
            missing_prices=tuple(concentration.get("missing_prices") or ()),
            unresolved_inputs=tuple(
                concentration.get("unresolved_inputs") or ()),
            denominator_scope=tuple(
                concentration.get("denominator_scope") or ()),
            excluded_components=tuple(
                concentration.get("excluded_components") or ())),
        comparisons=ComparisonContext(
            verdict_rows=tuple(comparisons.get("verdict_rows") or ())),
        modelling_scope={k: tuple(v) for k, v
                         in (payload.get("modelling_scope") or {}).items()},
        context_version=payload.get("context_version", CONTEXT_VERSION))
