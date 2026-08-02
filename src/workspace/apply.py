"""Accepting a proposal: durable state, in an order that cannot orphan anything.

    validate source revision
        -> persist artifacts and runs
        -> persist the worksheet revision citing them
        -> resolve the proposal
        -> commit

**No run means no revision.** The revision cites the runs, so the runs must
exist first — and all of it commits together, because an accepted edit that
produced three runs and no revision leaves history belonging to nothing.

**A stale proposal is refused, never rebased.** Applying a diff computed against
revision 2 to revision 5 would apply changes to state the reviewer never saw.
Replanning against the current revision is the safe answer, and it is the user's
to ask for.

The proposal itself is immutable. Acceptance records an outcome beside it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .proposal import WorksheetProposal
from .worksheet import Block, ResearchWorksheet, from_json, revise


class ProposalStatus(str, Enum):
    PROPOSED = "PROPOSED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    SUPERSEDED = "SUPERSEDED"


class ApplyRefused(RuntimeError):
    """The proposal was not applied, and why."""


class StaleProposal(ApplyRefused):
    """The worksheet advanced after the proposal was reviewed.

    Not rebased on purpose: an old diff applied to new state changes things the
    reviewer never saw.
    """


@dataclass
class ApplyResult:
    proposal_id: str
    revision: int
    runs: Sequence[str] = ()
    derived: Sequence[str] = ()
    status: str = ProposalStatus.ACCEPTED.value

    def to_json(self) -> Dict[str, Any]:
        return {"proposal_id": self.proposal_id, "revision": self.revision,
                "runs": list(self.runs), "derived": list(self.derived),
                "status": self.status}


def accept(store, *, proposal_id: str, owner: str, worksheet_id: str,
           proposal: WorksheetProposal, at: str, actor: str = "pilot",
           run_candidate: Optional[Callable[[Sequence[str]], Mapping[str, Any]]] = None
           ) -> ApplyResult:
    """Apply a reviewed proposal, or refuse with a reason.

    `run_candidate` simulates one candidate and returns its result payload. It
    is injected rather than imported so the apply path can be tested against a
    failing run without a price file, and so a caller cannot accidentally get a
    scenario change applied without one.
    """
    if not proposal.applicable:
        raise ApplyRefused(
            f"proposal {proposal_id} is not applicable: "
            + "; ".join(u.why for u in proposal.unsupported))

    record = store.get_worksheet_proposal(proposal_id, owner)
    if record is not None and record["status"] != ProposalStatus.PROPOSED.value:
        raise ApplyRefused(
            f"proposal {proposal_id} is already {record['status']}. Accepting "
            "it twice would produce a second revision for one review")

    latest = store.get_worksheet(worksheet_id, owner)
    if latest is None:
        raise ApplyRefused(f"no worksheet {worksheet_id!r}")
    worksheet = from_json(latest["payload"])

    if worksheet.revision != proposal.source_revision:
        raise StaleProposal(
            f"proposal {proposal_id} was reviewed against revision "
            f"{proposal.source_revision} and the worksheet is now at "
            f"{worksheet.revision}. Re-plan against the current revision rather "
            "than applying a diff to state nobody reviewed")

    runs: List[str] = []
    derived: List[str] = []

    with store.transaction():
        # Artifacts first, always. The revision cites them, and a revision that
        # cites a run which does not exist is the dangling reference this whole
        # ordering exists to prevent.
        if proposal.proposed_scenario_patch is not None:
            if run_candidate is None:
                raise ApplyRefused(
                    "a scenario change requires a run, and no runner was "
                    "supplied. No run means no revision")
            for index, change in enumerate(proposal.changes):
                candidate = change.value
                result = run_candidate(candidate)
                run_id = f"{proposal_id}-run-{index}"
                store.record_run(run_id=run_id, plan_id=worksheet.scenario_ref,
                                 ran_at=at, result=dict(result),
                                 comparison={"candidate": list(candidate)})
                runs.append(run_id)

        elif proposal.edit_effect == "DERIVED_ANALYSIS":
            for index, change in enumerate(proposal.changes):
                # Every variant is recorded, including the ones a selection did
                # not keep. Persisting only the chosen one would leave a
                # worksheet that shows a winner and no search.
                identifier = f"{proposal_id}-analysis-{index}"
                store.record_confirmation_event(
                    event_id=identifier, owner=owner, occurred_at=at,
                    kind="derived_analysis_recorded",
                    field=str(change.value.get("metric")),
                    final_value=str(change.value.get("parameter")),
                    provenance=proposal.selection_basis)
                derived.append(identifier)

        updated = revise(
            worksheet,
            reason=f"accepted proposal {proposal_id}",
            created_at=at,
            layout=(tuple(Block(b) for b in proposal.proposed_layout)
                    if proposal.proposed_layout else worksheet.layout),
            # Every candidate stays referenced. Citing only the activated one
            # would hide the alternatives that were actually evaluated, which
            # is the record trial accounting exists to keep.
            benchmark_run_refs=tuple(worksheet.benchmark_run_refs) + tuple(runs),
        )
        store.save_worksheet(updated)
        store.resolve_worksheet_proposal(proposal_id, owner,
                               status=ProposalStatus.ACCEPTED.value,
                               resolved_at=at, actor=actor,
                               result_revision=updated.revision,
                               result_runs=runs)

    return ApplyResult(proposal_id=proposal_id, revision=updated.revision,
                       runs=tuple(runs), derived=tuple(derived))


def reject(store, *, proposal_id: str, owner: str, at: str,
           actor: str = "pilot") -> None:
    """Record that a proposal was declined. It stays readable."""
    store.resolve_worksheet_proposal(proposal_id, owner,
                           status=ProposalStatus.REJECTED.value,
                           resolved_at=at, actor=actor)
