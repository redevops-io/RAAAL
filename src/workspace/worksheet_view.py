"""Resolving a stored worksheet into something a page can render.

The single rule: **this reads.** It resolves references and arranges what it
finds. It does not compile, simulate, revise, refresh, or resolve a concept id
to the newest version of anything.

That last one matters most. A route that helpfully upgrades `plan-1` to the
latest run silently changes what a saved worksheet means — the figures move, the
user never asked, and nothing in the interface says so. A newer run may be
*offered*; the stored worksheet keeps opening against the version it cited.

Derived display state is allowed. Derived financial values are not: every number
on the page comes out of a stored run, because a figure recomputed at render
time is a second implementation of the engine living in the view layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .worksheet import REGISTRY, Block, ResearchWorksheet


@dataclass(frozen=True)
class ResolvedBlock:
    """One section, and whether it could be filled."""

    block: str
    title: str
    available: bool
    payload: Mapping[str, Any] = field(default_factory=dict)
    unavailable_because: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"block": self.block, "title": self.title,
                "available": self.available,
                "payload": dict(self.payload),
                "unavailable_because": self.unavailable_because}


@dataclass
class WorksheetView:
    worksheet: Mapping[str, Any]
    blocks: Sequence[ResolvedBlock] = ()
    unresolved_references: Sequence[str] = ()

    @property
    def title(self) -> str:
        return self.worksheet.get("title") or self.worksheet["worksheet_id"]

    @property
    def is_complete(self) -> bool:
        return all(b.available for b in self.blocks)

    def to_json(self) -> Dict[str, Any]:
        return {"worksheet": dict(self.worksheet),
                "blocks": [b.to_json() for b in self.blocks],
                "unresolved_references": list(self.unresolved_references)}


def _scope_of(run: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return (run or {}).get("result", {}).get("modelling_scope") or {}


def build(worksheet: ResearchWorksheet, *, store, owner: str) -> WorksheetView:
    """Resolve every reference the worksheet names. Nothing else.

    A reference that cannot be resolved produces a *named* unavailable block
    rather than a dropped section. An omitted panel is invisible; a panel that
    says what is missing is a fact a reader can act on.
    """
    unresolved: List[str] = []

    plan = store.get_plan(worksheet.scenario_ref, owner)
    if plan is None and worksheet.scenario_ref:
        unresolved.append(f"scenario {worksheet.scenario_ref}")

    run = None
    if worksheet.primary_run_ref:
        run = store.get_run(worksheet.primary_run_ref, owner)
        if run is None:
            unresolved.append(f"run {worksheet.primary_run_ref}")

    benchmark_runs = []
    for reference in worksheet.benchmark_run_refs:
        found = store.get_run(reference, owner)
        if found is None:
            unresolved.append(f"benchmark run {reference}")
        else:
            benchmark_runs.append(found)

    scenario = (plan or {}).get("scenario") or {}
    unmet = worksheet.unavailable_blocks()

    blocks: List[ResolvedBlock] = []
    for spec in worksheet.blocks():
        name = spec.block.value
        if name in unmet:
            blocks.append(ResolvedBlock(name, spec.title, False,
                                        unavailable_because=unmet[name]))
            continue

        missing = [reference for reference in spec.requires
                   if reference == "scenario_ref" and plan is None
                   or reference == "primary_run_ref" and run is None]
        if missing:
            blocks.append(ResolvedBlock(
                name, spec.title, False,
                unavailable_because=(
                    f"{', '.join(missing)} could not be resolved; the artifact "
                    "it names is not in this workspace")))
            continue

        blocks.append(ResolvedBlock(name, spec.title, True,
                                    payload=_payload(spec.block, scenario, run,
                                                     benchmark_runs, worksheet)))
    return WorksheetView(worksheet=worksheet.to_json(), blocks=tuple(blocks),
                         unresolved_references=tuple(unresolved))


def _payload(block: Block, scenario: Mapping[str, Any],
             run: Optional[Mapping[str, Any]],
             benchmark_runs: Sequence[Mapping[str, Any]],
             worksheet: ResearchWorksheet) -> Dict[str, Any]:
    """What one block shows, read from stored artifacts.

    Every figure here came out of a stored run. Nothing is recomputed: a number
    derived at render time is a second implementation of the engine living in
    the view layer, and it would drift from the one that produced the record.
    """
    methodology = scenario.get("methodology") or {}
    flows = scenario.get("flows") or {}
    protocol = scenario.get("protocol") or {}
    result = (run or {}).get("result") or {}

    if block is Block.STRATEGY_DEFINITION:
        return {"assets": (methodology.get("allocation_rule") or {}).get("assets", []),
                "weighting": (methodology.get("allocation_rule") or {}).get("weighting"),
                "holdings_policy": methodology.get("holdings_policy") or {},
                "event_program": methodology.get("event_program") or [],
                "cadence": flows.get("cadence"), "amount": flows.get("amount"),
                "funding_source": flows.get("funding_source"),
                "day_rule": flows.get("day_rule"),
                "account": protocol.get("tax_treatment")}

    if block is Block.INTERPRETATION_SUMMARY:
        return {"inferred": scenario.get("inferred") or [],
                "stated_text": None}

    if block is Block.BENCHMARK_COMPARISON:
        comparison = (run or {}).get("comparison") or {}
        return {"comparability": comparison.get("comparability"),
                "members": comparison.get("members", []),
                "benchmark_runs": [b["run_id"] for b in benchmark_runs],
                # Comparability is shown before any side-by-side figure, so a
                # reader cannot rank two results before learning whether they
                # are comparable at all.
                "verdict_first": True}

    if block is Block.PERFORMANCE_SUMMARY:
        return {"time_weighted_annualized": result.get("time_weighted_annualized"),
                "money_weighted": result.get("money_weighted"),
                "final_value": result.get("final_value"),
                "ran_at": (run or {}).get("ran_at")}

    if block is Block.MODELING_SCOPE:
        return {"scope": _scope_of(run)}

    if block is Block.TRIAL_ACCOUNTING:
        return {"variants_evaluated": len(worksheet.benchmark_run_refs),
                "selection_basis": "STATED_PREFERENCE"}

    if block is Block.ARTIFACT_CHAIN:
        return {"scenario_ref": worksheet.scenario_ref,
                "primary_run_ref": worksheet.primary_run_ref,
                "benchmark_run_refs": list(worksheet.benchmark_run_refs),
                "mission_ref": worksheet.mission_ref,
                "context_view_ref": worksheet.context_view_ref,
                "worksheet_hash": worksheet.canonical_hash,
                "revision": worksheet.revision}
    return {}
