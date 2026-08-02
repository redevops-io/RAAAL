"""A planner classification, turned into a reviewable change.

    WorksheetIntent -> typed diff -> impact -> confirmation -> new revision

**Every proposed edit names the exact block field or artifact it changes, or is
marked unsupported with a reason.** A proposal that classified successfully and
produced prose would be recognition without representation at the worksheet
layer — the same defect as a compiler that reads "hold dividends as cash" and
compiles a scenario with no trace of it.

So the generator refuses. A classification it cannot translate becomes an
`Unsupported` entry naming what it could not express, and the proposal reports
itself as not applicable rather than looking ready.

**Layout and scenario changes live in separate fields.** A request that touches
both must say so, because a financial change hidden inside a presentation edit
is the one a reviewer skims past.

Nothing here writes. A proposal is an offer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .intent import EditEffect, SelectionBasis, WorksheetIntent
from .worksheet import REGISTRY, Block, ResearchWorksheet


@dataclass(frozen=True)
class Change:
    """One typed edit. Names its target, never describes it."""

    target: str
    """`layout`, a block id, or a dotted path into the scenario."""

    operation: str          # "reorder" | "add_block" | "set" | "activate"
    value: Any = None
    previous: Any = None

    def to_json(self) -> Dict[str, Any]:
        return {"target": self.target, "operation": self.operation,
                "value": self.value, "previous": self.previous}


@dataclass(frozen=True)
class Unsupported:
    """A classification the generator could not express as a typed change.

    Kept rather than dropped, and it makes the whole proposal inapplicable. A
    partially-expressible edit applied partially is worse than one refused: the
    user asked for something and got most of it, with no way to see which part.
    """

    what: str
    why: str

    def to_json(self) -> Dict[str, str]:
        return {"what": self.what, "why": self.why}


@dataclass
class WorksheetProposal:
    """What would change, and what it would cost."""

    intent_ref: str
    source_revision: int
    edit_effect: str
    selection_basis: str
    repetition_signature: Mapping[str, str]

    changes: Sequence[Change] = ()
    unsupported: Sequence[Unsupported] = ()

    affected_blocks: Sequence[str] = ()
    affected_artifacts: Sequence[str] = ()

    rerun_required: bool = False
    trial_effect: int = 0
    comparability_impact: str = ""
    warnings: Sequence[str] = ()

    #: Kept apart deliberately. A request that changes both must say so rather
    #: than hiding a financial change inside a presentation one.
    proposed_layout: Optional[Sequence[str]] = None
    proposed_scenario_patch: Optional[Mapping[str, Any]] = None

    @property
    def applicable(self) -> bool:
        """Whether this can be applied at all. False while anything is
        unsupported — a proposal that applies most of a request is worse than
        one that refuses it."""
        return bool(self.changes) and not self.unsupported

    @property
    def touches_money(self) -> bool:
        return self.proposed_scenario_patch is not None

    def to_json(self) -> Dict[str, Any]:
        return {
            "intent_ref": self.intent_ref,
            "source_revision": self.source_revision,
            "edit_effect": self.edit_effect,
            "selection_basis": self.selection_basis,
            "repetition_signature": dict(self.repetition_signature),
            "changes": [c.to_json() for c in self.changes],
            "unsupported": [u.to_json() for u in self.unsupported],
            "affected_blocks": list(self.affected_blocks),
            "affected_artifacts": list(self.affected_artifacts),
            "rerun_required": self.rerun_required,
            "trial_effect": self.trial_effect,
            "comparability_impact": self.comparability_impact,
            "warnings": list(self.warnings),
            "proposed_layout": (list(self.proposed_layout)
                                if self.proposed_layout is not None else None),
            "proposed_scenario_patch": (dict(self.proposed_scenario_patch)
                                        if self.proposed_scenario_patch is not None
                                        else None),
            "applicable": self.applicable,
            "touches_money": self.touches_money,
        }


#: Where a layout request wants a block moved. Narrow on purpose: an instruction
#: naming a block the registry does not have is unsupported, not approximated.
_BLOCK_WORDS = {
    "scope": Block.MODELING_SCOPE, "modelling scope": Block.MODELING_SCOPE,
    "modeling scope": Block.MODELING_SCOPE,
    "risk": Block.PERFORMANCE_SUMMARY, "results": Block.PERFORMANCE_SUMMARY,
    "performance": Block.PERFORMANCE_SUMMARY,
    "benchmark": Block.BENCHMARK_COMPARISON,
    "benchmarks": Block.BENCHMARK_COMPARISON,
    "comparability": Block.BENCHMARK_COMPARISON,
    "strategy": Block.STRATEGY_DEFINITION,
    "provenance": Block.ARTIFACT_CHAIN,
    "trials": Block.TRIAL_ACCOUNTING,
    "alternatives": Block.TRIAL_ACCOUNTING,
}

#: Scenario fields a substitution can name, and where they live. A dimension
#: outside this map is unsupported rather than guessed at.
_SCENARIO_PATHS = {
    "holdings": "methodology.allocation_rule.assets",
    "inputs": "flows",
}


def _blocks_named(instruction: str) -> List[Block]:
    lowered = instruction.lower()
    found: List[Block] = []
    for word, block in _BLOCK_WORDS.items():
        if word in lowered and block not in found:
            found.append(block)
    return found


def _layout_change(intent: WorksheetIntent,
                   worksheet: ResearchWorksheet) -> tuple:
    """Reorder the layout, or say why it cannot be expressed."""
    named = _blocks_named(intent.instruction)
    if not named:
        return [], [Unsupported(
            what=intent.instruction,
            why=("no block in the registry matches this. Naming the nearest "
                 "one would move a panel the request never mentioned"))], None

    layout = list(worksheet.layout)
    moving = named[0]
    if moving not in layout:
        return [], [Unsupported(
            what=f"move {moving.value}",
            why="this worksheet does not contain that block")], None

    layout.remove(moving)
    anchor = next((b for b in named[1:] if b in layout), None)
    if anchor is None:
        layout.append(moving)
    else:
        index = layout.index(anchor)
        # "below X" places it after; anything else places it before. Stated
        # rather than inferred from the verb, because "move the scope panel"
        # with no anchor is a different request from "move it below risk".
        after = "below" in intent.instruction.lower() or \
            "under" in intent.instruction.lower()
        layout.insert(index + 1 if after else index, moving)

    if layout == list(worksheet.layout):
        # Already in that order. Reported rather than proposed: a revision that
        # changes nothing still costs a revision, and a diff showing two
        # identical lists asks a reviewer to spot a difference that is not
        # there.
        return [], [Unsupported(
            what=intent.instruction,
            why=("the worksheet is already in that order, so there is nothing "
                 "to change"))], None

    change = Change(target="layout", operation="reorder",
                    value=[b.value for b in layout],
                    previous=[b.value for b in worksheet.layout])
    return [change], [], [b.value for b in layout]


def _analysis_changes(intent: WorksheetIntent) -> tuple:
    """Add one analytical block per requested parameter.

    Every variant becomes its own block. Folding a search into one block that
    shows the chosen window would delete the alternatives, which is exactly the
    record trial accounting exists to keep.
    """
    metric = intent.repetition_signature.metric
    if not metric:
        return [], [Unsupported(
            what=intent.instruction,
            why=("no statistic was named, and a chart of nothing in particular "
                 "cannot be produced"))]

    parameters = list(intent.requested_parameters) or [""]
    changes = [
        Change(target=f"analysis/{metric}/{parameter or 'default'}",
               operation="activate" if intent.selection_basis
               is SelectionBasis.AFTER_RESULTS else "add_block",
               value={"metric": metric, "parameter": parameter,
                      "family": intent.repetition_signature.parameter_family})
        for parameter in parameters
    ]
    return changes, []


def _scenario_changes(intent: WorksheetIntent) -> tuple:
    """A typed patch into the scenario, or a refusal."""
    family = intent.repetition_signature.parameter_family
    path = _SCENARIO_PATHS.get(family)
    if path is None:
        return [], [Unsupported(
            what=intent.instruction,
            why=(f"the planner classified this as a {family!r} change and no "
                 "typed path expresses it"))], None

    values = list(intent.requested_parameters)
    if not values:
        return [], [Unsupported(
            what=intent.instruction,
            why="no instrument or value was named, so nothing can be set")], None

    searching = intent.selection_basis in {SelectionBasis.VARIANT_EXPLORATION,
                                           SelectionBasis.AFTER_RESULTS}
    if searching and len(values) > 1:
        # Each candidate is its own scenario, not one portfolio of three.
        # Setting the holdings to [SPY, VTI, VT] would propose a three-asset
        # strategy nobody described, and would record one trial where three
        # were run.
        changes = [Change(target=path, operation="set", value=[value],
                          previous=None)
                   for value in values]
        return changes, [], {path: [[v] for v in values]}

    patch = {path: values}
    changes = [Change(target=path, operation="set", value=values)]
    return changes, [], patch


def propose(intent: WorksheetIntent,
            worksheet: ResearchWorksheet) -> WorksheetProposal:
    """Turn one classification into a reviewable change, or refuse."""
    changes: List[Change] = []
    unsupported: List[Unsupported] = []
    layout: Optional[Sequence[str]] = None
    patch: Optional[Mapping[str, Any]] = None
    warnings: List[str] = []

    if intent.edit_effect is EditEffect.LAYOUT_ONLY:
        changes, unsupported, layout = _layout_change(intent, worksheet)
    elif intent.edit_effect is EditEffect.DERIVED_ANALYSIS:
        changes, unsupported = _analysis_changes(intent)
    else:
        changes, unsupported, patch = _scenario_changes(intent)

    if intent.selection_basis is SelectionBasis.AFTER_RESULTS:
        warnings.append(
            "This choice was made after seeing the results. Every alternative "
            "evaluated stays in the record and counts toward the trial total; "
            "keeping only the one that looked best is how an ordinary result "
            "comes to look remarkable.")
    if intent.rerun_required:
        warnings.append(
            "A new run is required before the worksheet revision, so the "
            "revision cites a run that already exists.")
    if patch is not None and layout is not None:
        warnings.append(
            "This changes both what is simulated and how it is displayed.")

    return WorksheetProposal(
        intent_ref=intent.intent_id,
        source_revision=intent.source_revision,
        edit_effect=intent.edit_effect.value,
        selection_basis=intent.selection_basis.value,
        repetition_signature=intent.repetition_signature.to_json(),
        changes=tuple(changes), unsupported=tuple(unsupported),
        affected_blocks=tuple(intent.target_blocks),
        affected_artifacts=(("scenario", "run") if patch is not None else ()),
        rerun_required=intent.rerun_required,
        trial_effect=intent.trial_effect,
        comparability_impact=intent.comparability_impact,
        warnings=tuple(warnings),
        proposed_layout=layout, proposed_scenario_patch=patch,
    )
