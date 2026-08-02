"""What an accepted proposal will execute, stated as Quantify's own declaration.

    accepted WorksheetProposal
        -> WorksheetExecutionDeclaration
        -> (optional) an adapter, into whatever orchestrator runs it

Deliberately neutral. It inherits from no SDK type and imports no contract
package, because the moment a financial declaration inherits an orchestration
type, the orchestrator's vocabulary starts deciding what Quantify can say.

    Quantify core     knows financial meaning
    an adapter        translates it
    the orchestrator  knows generic orchestration

This exists whether or not anything consumes it. Its immediate value is that the
execution semantics of `apply.py` — three candidates run before one revision,
every candidate retained, the proposal resolved last — become a *statement* that
can be compared against another representation, rather than an ordering implied
by the sequence of lines in a function.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .proposal import WorksheetProposal


@dataclass(frozen=True)
class CandidateExecution:
    """One thing that must run before the revision can exist."""

    candidate_id: str
    kind: str
    """`scenario` or `analysis`. A scenario candidate simulates; an analysis
    candidate computes over a run that already exists."""

    inputs: Mapping[str, Any]
    produces: str
    """The artifact this candidate must persist. Named, because a candidate that
    produces nothing cannot be a precondition for anything."""

    side_effecting: bool = True

    def to_json(self) -> Dict[str, Any]:
        return {"candidate_id": self.candidate_id, "kind": self.kind,
                "inputs": dict(self.inputs), "produces": self.produces,
                "side_effecting": self.side_effecting}


@dataclass(frozen=True)
class WorksheetExecutionDeclaration:
    """The execution an accepted proposal implies, and its ordering."""

    proposal_id: str
    source_revision: int
    candidates: Sequence[CandidateExecution]
    selected_candidate: Optional[str]
    """Which candidate a result-aware selection kept. Distinct from the
    candidate list on purpose: the evaluated set and the chosen one are
    different facts, and collapsing them is how a search becomes a single
    confident answer."""

    trial_effect: int
    required_terminal_artifacts: Sequence[str]
    edit_effect: str
    selection_basis: str

    #: The ordering the apply path must honour, stated rather than implied.
    #: Each entry must complete before the next begins.
    ordering: Sequence[str] = (
        "candidates", "worksheet_revision", "proposal_resolution")

    @property
    def fan_out(self) -> int:
        """How many candidates run in parallel before the join.

        One search over three instruments is three candidates, never one
        candidate holding three instruments — the distinction a flattened graph
        would lose, and the one trial accounting depends on.
        """
        return len(self.candidates)

    @property
    def requires_runs(self) -> bool:
        return any(c.kind == "scenario" for c in self.candidates)

    def to_json(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "source_revision": self.source_revision,
            "candidates": [c.to_json() for c in self.candidates],
            "selected_candidate": self.selected_candidate,
            "trial_effect": self.trial_effect,
            "required_terminal_artifacts": list(self.required_terminal_artifacts),
            "edit_effect": self.edit_effect,
            "selection_basis": self.selection_basis,
            "ordering": list(self.ordering),
            "fan_out": self.fan_out,
            "requires_runs": self.requires_runs,
        }


def declare(proposal: WorksheetProposal) -> WorksheetExecutionDeclaration:
    """State what accepting this proposal would execute.

    Refuses an inapplicable proposal rather than declaring an execution for
    something that will not be applied — a declaration nothing can run is a
    plan for a thing that does not happen.
    """
    if not proposal.applicable:
        raise ValueError(
            f"proposal {proposal.intent_ref} is not applicable, so there is no "
            "execution to declare: "
            + "; ".join(u.why for u in proposal.unsupported))

    candidates: List[CandidateExecution] = []
    if proposal.proposed_scenario_patch is not None:
        for index, change in enumerate(proposal.changes):
            candidates.append(CandidateExecution(
                candidate_id=f"candidate/{'-'.join(map(str, change.value))}",
                kind="scenario",
                inputs={"target": change.target, "value": change.value},
                produces="run"))
    elif proposal.edit_effect == "DERIVED_ANALYSIS":
        for change in proposal.changes:
            parameter = change.value.get("parameter") or "default"
            candidates.append(CandidateExecution(
                candidate_id=f"analysis/{change.value.get('metric')}/{parameter}",
                kind="analysis",
                inputs=dict(change.value), produces="derived_analysis"))
    else:
        candidates.append(CandidateExecution(
            candidate_id="layout", kind="layout",
            inputs={"layout": list(proposal.proposed_layout or ())},
            produces="worksheet_revision", side_effecting=False))

    # A selection names one of the candidates it was chosen from. Read from the
    # activated change rather than assumed to be the last, because the order a
    # search was written in says nothing about which one was kept.
    selected = None
    if proposal.selection_basis == "AFTER_RESULTS":
        activated = [c for c in proposal.changes if c.operation == "activate"]
        if activated:
            value = activated[0].value
            parameter = (value.get("parameter") if isinstance(value, dict)
                         else "-".join(map(str, value)))
            selected = candidates[proposal.changes.index(activated[0])].candidate_id
            del parameter

    terminal = ["worksheet_revision", "proposal_resolution"]
    if any(c.kind == "scenario" for c in candidates):
        terminal.insert(0, "run")

    return WorksheetExecutionDeclaration(
        proposal_id=proposal.intent_ref,
        source_revision=proposal.source_revision,
        candidates=tuple(candidates), selected_candidate=selected,
        trial_effect=proposal.trial_effect,
        required_terminal_artifacts=tuple(terminal),
        edit_effect=proposal.edit_effect,
        selection_basis=proposal.selection_basis)
