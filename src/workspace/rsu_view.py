"""The RSU result, arranged for a page. A projection, never a second engine.

    engine computes -> MissionResult stores -> WorkspaceStore preserves
                    -> this reads

`from_result` takes a stored payload and nothing else. No prices, no runtimes,
no calculators, no scenario. The constructor shape is the guard: given nothing
to recompute with, recomputation is not an option a future edit can quietly
take. A test additionally patches every engine calculator to raise and requires
the page to render unchanged.

A figure recomputed at render time is a second implementation of the engine
living in the view layer, and the two will disagree on exactly the runs where
something went wrong — which are the runs a reader most needs to trust.

**Nothing collapses into a badge.** Pending and failed are different, projected
and realized are different, and incomparable and not-evaluated are different. A
single summary status would be the one thing a reader remembers, and each of
those pairs would land on the same side of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class ContextState(str, Enum):
    PRESENT = "RSU_CONTEXT_PRESENT"
    NOT_DECLARED = "RSU_CONTEXT_NOT_DECLARED"
    """No structured context was recorded. For a historical run this is a fact
    about the record, not about the run."""

    CORRUPT = "RSU_CONTEXT_CORRUPT"
    """Stored and unverifiable. The shell and provenance still render; the
    financial blocks say the result could not be verified rather than showing
    figures from a payload that failed its own checks."""


NOT_DECLARED_NOTE = (
    "Structured RSU result scope was not recorded for this historical run. "
    "That is an absence of record, not evidence that nothing was excluded.")

CORRUPT_NOTE = (
    "The stored result could not be verified, so its figures are not shown. "
    "The record was altered after it was written.")


@dataclass(frozen=True)
class VestAccountingView:
    gross_value: Optional[float] = None
    withheld_value: Optional[float] = None
    delivered_value: Optional[float] = None
    cash_remainder: Optional[float] = None
    basis_note: str = ""
    unpriced_arrivals: Sequence[Mapping[str, Any]] = ()

    def to_json(self) -> Dict[str, Any]:
        return {"gross_value": self.gross_value,
                "withheld_value": self.withheld_value,
                "delivered_value": self.delivered_value,
                "cash_remainder": self.cash_remainder,
                "basis_note": self.basis_note,
                "unpriced_arrivals": [dict(o) for o in self.unpriced_arrivals]}


@dataclass(frozen=True)
class DispositionView:
    status: str = ""
    pending: Sequence[Mapping[str, Any]] = ()
    failed: Sequence[Mapping[str, Any]] = ()
    unsettled: Sequence[Mapping[str, Any]] = ()
    executions: Sequence[Mapping[str, Any]] = ()

    @property
    def has_outstanding(self) -> bool:
        return bool(self.pending or self.failed or self.unsettled)

    def to_json(self) -> Dict[str, Any]:
        return {"status": self.status,
                "pending": [dict(o) for o in self.pending],
                "failed": [dict(o) for o in self.failed],
                "unsettled": [dict(o) for o in self.unsettled],
                "executions": [dict(o) for o in self.executions],
                "has_outstanding": self.has_outstanding}


@dataclass(frozen=True)
class AllocationView:
    requested: Mapping[str, float] = field(default_factory=dict)
    executed: Mapping[str, float] = field(default_factory=dict)
    unfilled: Sequence[Mapping[str, Any]] = ()
    residual_cash: Optional[float] = None
    unallocated_weight: Optional[float] = None
    purchase_costs: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        return {"requested": dict(self.requested),
                "executed": dict(self.executed),
                "unfilled": [dict(o) for o in self.unfilled],
                "residual_cash": self.residual_cash,
                "unallocated_weight": self.unallocated_weight,
                "purchase_costs": self.purchase_costs}


@dataclass(frozen=True)
class ConcentrationView:
    """Projected and realized stay in separate fields to the page.

    Rendered into one "concentration" figure, a plan's expectation and a fill's
    outcome become indistinguishable, and the expectation is the one that looks
    like an achievement.
    """

    current: Optional[float] = None
    declared_target: Optional[float] = None
    projected: Optional[float] = None
    realized: Optional[float] = None
    cap_achieved: Optional[bool] = None
    denominator_scope: Sequence[str] = ()
    missing_prices: Sequence[str] = ()
    scope_note: str = ""

    @property
    def projected_only(self) -> bool:
        return self.projected is not None and self.realized is None

    def to_json(self) -> Dict[str, Any]:
        return {"current": self.current,
                "declared_target": self.declared_target,
                "projected": self.projected, "realized": self.realized,
                "projected_only": self.projected_only,
                "cap_achieved": self.cap_achieved,
                "denominator_scope": list(self.denominator_scope),
                "missing_prices": list(self.missing_prices),
                "scope_note": self.scope_note}


@dataclass(frozen=True)
class ComparisonRowView:
    benchmark_id: str
    status: str
    reason: str = ""
    unchecked_dimensions: Sequence[str] = ()
    differing_dimensions: Sequence[str] = ()
    isolates: str = ""

    @property
    def attribution_isolated(self) -> bool:
        return self.status == "COMPARABLE" and not self.unchecked_dimensions

    def to_json(self) -> Dict[str, Any]:
        return {"benchmark_id": self.benchmark_id, "status": self.status,
                "reason": self.reason,
                "unchecked_dimensions": list(self.unchecked_dimensions),
                "differing_dimensions": list(self.differing_dimensions),
                "isolates": self.isolates,
                "attribution_isolated": self.attribution_isolated}


@dataclass(frozen=True)
class ModellingScopeView:
    modelled: Sequence[str] = ()
    not_applicable: Sequence[str] = ()
    out_of_scope: Sequence[str] = ()
    unresolved: Sequence[str] = ()

    def to_json(self) -> Dict[str, Any]:
        return {"modelled": list(self.modelled),
                "not_applicable": list(self.not_applicable),
                "out_of_scope": list(self.out_of_scope),
                "unresolved": list(self.unresolved)}


#: Display order. Scope sits directly beneath the figures it qualifies rather
#: than at the foot of the page, where it is read after the number has been
#: believed.
BLOCK_ORDER = ("result_status", "vest_accounting", "disposition", "allocation",
               "concentration", "benchmark_comparability", "modelling_scope",
               "provenance")


@dataclass(frozen=True)
class RSUWorksheetView:
    """Everything the RSU page shows, decided here and computed nowhere."""

    context_state: ContextState
    presentability: str = ""
    diagnostics: Sequence[Mapping[str, Any]] = ()
    vest_accounting: VestAccountingView = field(
        default_factory=VestAccountingView)
    disposition: DispositionView = field(default_factory=DispositionView)
    allocation: AllocationView = field(default_factory=AllocationView)
    concentration: ConcentrationView = field(default_factory=ConcentrationView)
    comparisons: Sequence[ComparisonRowView] = ()
    modelling_scope: ModellingScopeView = field(
        default_factory=ModellingScopeView)
    scope_disclosure: Optional[Mapping[str, Any]] = None
    """Read from the stored result, never rebuilt.

    Rebuilt from today's runtimes, a historical worksheet would show figures
    from one run beside the scope of another — and nothing on the page would
    say the scope had moved."""

    note: str = ""
    block_order: Sequence[str] = BLOCK_ORDER

    @property
    def financial_blocks_available(self) -> bool:
        """Whether the figures may be shown at all.

        False for a corrupt context: the shell and provenance still render, and
        the financial blocks say the result could not be verified rather than
        displaying numbers from a payload that failed its own checks.
        """
        return self.context_state is ContextState.PRESENT

    @classmethod
    def from_result(cls, result: Optional[Mapping[str, Any]]
                    ) -> "RSUWorksheetView":
        """The only constructor.

        Takes a stored result payload. Not a scenario, not prices, not a
        store — nothing that could be used to recompute a figure. The signature
        is the guard: a future edit that wanted to recalculate would have to add
        a parameter, which is visible in review in a way that a quiet call to a
        calculator is not.
        """
        payload = (result or {}).get("rsu_context")
        if not payload:
            return cls(context_state=ContextState.NOT_DECLARED,
                       note=NOT_DECLARED_NOTE)

        try:
            _verify(payload)
        except Exception:                                       # noqa: BLE001
            # No recovery attempt. A partially rendered corrupt result is an
            # edited record presented as an original.
            return cls(context_state=ContextState.CORRUPT, note=CORRUPT_NOTE)

        vest = payload.get("vest_accounting") or {}
        disposition = payload.get("disposition") or {}
        allocation = payload.get("allocation") or {}
        concentration = payload.get("concentration") or {}
        comparisons = (payload.get("comparisons") or {}).get("verdict_rows") or ()
        scope = payload.get("modelling_scope") or {}

        return cls(
            context_state=ContextState.PRESENT,
            presentability=payload.get("presentability", ""),
            diagnostics=tuple(payload.get("diagnostics") or ()),
            vest_accounting=VestAccountingView(
                gross_value=vest.get("gross_vest_value"),
                withheld_value=vest.get("withheld_value"),
                delivered_value=vest.get("delivered_value"),
                cash_remainder=vest.get("cash_remainder"),
                basis_note=vest.get("basis_note", ""),
                unpriced_arrivals=tuple(vest.get("unpriced_arrivals") or ())),
            disposition=DispositionView(
                status=disposition.get("status", ""),
                pending=tuple(disposition.get("pending_instructions") or ()),
                failed=tuple(disposition.get("failed_instructions") or ()),
                unsettled=tuple(disposition.get("unsettled_report") or ()),
                executions=tuple(disposition.get("executions") or ())),
            allocation=AllocationView(
                requested=dict(allocation.get("requested_targets") or {}),
                executed=dict(allocation.get("executed_targets") or {}),
                unfilled=tuple(allocation.get("unfilled_targets") or ()),
                residual_cash=allocation.get("residual_cash"),
                unallocated_weight=allocation.get("unallocated_weight"),
                purchase_costs=allocation.get("purchase_costs")),
            concentration=ConcentrationView(
                current=concentration.get("current"),
                declared_target=concentration.get("target"),
                projected=concentration.get("projected"),
                realized=concentration.get("realized"),
                cap_achieved=concentration.get("cap_achieved"),
                denominator_scope=tuple(
                    concentration.get("denominator_scope") or ()),
                missing_prices=tuple(concentration.get("missing_prices") or ()),
                scope_note=concentration.get("scope_note", "")),
            comparisons=tuple(
                ComparisonRowView(
                    benchmark_id=row.get("benchmark_id", ""),
                    status=row.get("status", ""),
                    reason=row.get("reason", ""),
                    unchecked_dimensions=tuple(
                        row.get("unchecked_dimensions") or ()),
                    differing_dimensions=tuple(
                        row.get("differing_dimensions") or ()),
                    isolates=row.get("isolates", ""))
                for row in comparisons),
            scope_disclosure=payload.get("scope_disclosure"),
            modelling_scope=ModellingScopeView(
                modelled=tuple(scope.get("modelled") or ()),
                not_applicable=tuple(scope.get("not_applicable") or ()),
                out_of_scope=tuple(scope.get("out_of_scope") or ()),
                unresolved=tuple(scope.get("unresolved") or ())))

    def to_json(self) -> Dict[str, Any]:
        return {"context_state": self.context_state.value,
                "financial_blocks_available": self.financial_blocks_available,
                "presentability": self.presentability,
                "diagnostics": [dict(one) for one in self.diagnostics],
                "vest_accounting": self.vest_accounting.to_json(),
                "disposition": self.disposition.to_json(),
                "allocation": self.allocation.to_json(),
                "concentration": self.concentration.to_json(),
                "comparisons": [row.to_json() for row in self.comparisons],
                "modelling_scope": self.modelling_scope.to_json(),
                "scope_disclosure": (dict(self.scope_disclosure)
                                     if self.scope_disclosure else None),
                "scope_recorded": self.scope_disclosure is not None,
                "note": self.note, "block_order": list(self.block_order)}


def _verify(payload: Mapping[str, Any]) -> None:
    """Integrity only. Imported inside the function so the module's own imports
    stay free of anything that computes a financial value."""
    from ..mission.rsu_result import validate

    validate(payload)
