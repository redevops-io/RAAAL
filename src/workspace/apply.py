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

from ..db.errors import PublicCode, Retry, is_conflict
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


class ProposalConflict(ApplyRefused):
    """Another request resolved this proposal while we were applying it.

    Carries `PublicCode.STALE_TRANSITION`: the statement succeeded and moved no
    row, so what failed is the assumption it was issued under. The caller must
    re-read before retrying, which is a different instruction from "the service
    was busy" — and the reason this is not folded into contention.

    A subtype of `ApplyRefused` so callers that handle refusal handle this too,
    and distinct so one that wants to say "someone else got there first" can.
    Raised in place of whatever the database would otherwise have thrown, which
    would have carried constraint names and table structure into an application
    error path.
    """


class TransitionIntegrityError(RuntimeError):
    """A state transition changed a number of rows that cannot be right.

    Carries `PublicCode.TRANSITION_INTEGRITY_FAILURE` and is never retryable.

    Deliberately *not* an `ApplyRefused`. A refusal means the request was
    understood and declined; this means the stored state no longer matches what
    this code can reason about, and a caller that retried would be retrying
    against a database nobody has explained.
    """


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


#: Which public category each refusal becomes. Declared beside the exceptions
#: rather than in the HTTP layer: the layer that knows a transition moved zero
#: rows is the layer that knows the caller must re-read, and a route
#: reconstructing that from a class name would be deciding it a second time.
#:
#: Without this, `STALE_TRANSITION` and `TRANSITION_INTEGRITY_FAILURE` were two
#: declared categories nothing produced — a declaration with no reachable
#: consumer, which is the defect this codebase keeps finding in other people's
#: work and had just reintroduced in its own.
ProposalConflict.public_code = PublicCode.STALE_TRANSITION
ProposalConflict.retry_disposition = Retry.AFTER_REREAD
TransitionIntegrityError.public_code = PublicCode.TRANSITION_INTEGRITY_FAILURE
TransitionIntegrityError.retry_disposition = Retry.NEVER


def accept(store, *, proposal_id: str, owner: str, worksheet_id: str,
           proposal: WorksheetProposal, at: str, actor: str = "pilot",
           run_candidate: Optional[Callable[[Sequence[str]], Mapping[str, Any]]] = None,
           access=None) -> ApplyResult:
    """Apply a reviewed proposal, or refuse with a reason.

    `run_candidate` simulates one candidate and returns its result payload. It
    is injected rather than imported so the apply path can be tested against a
    failing run without a price file, and so a caller cannot accidentally get a
    scenario change applied without one.

    `access` is the one `MarketDataAccess` every candidate was simulated
    against. It is passed here rather than left inside the runner's closure
    because the delivery must be *recorded* once and cited by every candidate
    run: a fan-out where each candidate carried its own event would say the
    candidates were measured on different data, and comparing them is the only
    thing the fan-out is for.
    """
    if not proposal.applicable:
        raise ApplyRefused(
            f"proposal {proposal_id} is not applicable: "
            + "; ".join(u.why for u in proposal.unsupported))

    runs: List[str] = []
    derived: List[str] = []

    try:
        return _apply(
            store, proposal_id=proposal_id, owner=owner,
            worksheet_id=worksheet_id, proposal=proposal, at=at, actor=actor,
            run_candidate=run_candidate, runs=runs, derived=derived,
            access=access)
    except ApplyRefused:
        raise
    except Exception as exc:
        # Contention is meant to be settled by the row lock and the conditional
        # transition above. If either is lost, the database reports the
        # collision as a deadlock, a serialization failure or a unique
        # violation — carrying constraint names, table structure and process
        # ids that do not belong in an application error path. Found by
        # removing the lock in a test and watching `DeadlockDetected` escape.
        if is_conflict(exc):
            # Chained, not discarded. The public payload is the message below
            # and carries nothing internal; the original stays on `__cause__`
            # for logs and tracebacks. `from None` hid the SQLSTATE and the
            # backend context from operators as well as from callers, which
            # solved the wrong half of the problem — the two are separate
            # channels, and sanitising one must not blind the other.
            raise ProposalConflict(
                f"proposal {proposal_id} could not be applied because another "
                "request was changing the same records. Nothing from this "
                "attempt was kept; read the proposal again to see the outcome "
                "that stands") from exc
        raise


def _apply(store, *, proposal_id, owner, worksheet_id, proposal, at, actor,
           run_candidate, runs, derived, access=None) -> ApplyResult:
    with store.transaction():
        # Lock first, then read everything that authorizes the acceptance.
        #
        # These checks used to sit outside the transaction, and two sessions
        # could both pass them before either wrote — one review, two
        # acceptances, two ApplyResults claiming success. A check made before
        # the lock describes state another session is still free to change, so
        # the proposal row is taken first and the status and revision are read
        # after, inside the same transaction that will act on them.
        record = store.lock_worksheet_proposal(proposal_id, owner)
        if record is None:
            # Scoped to this owner, so a proposal belonging to someone else is
            # simply absent — the requester learns nothing about whether that
            # id exists elsewhere. Permitting a missing record used to let an
            # acceptance proceed with nothing to lock, nothing to check for a
            # prior outcome and nothing to transition, which is every guard in
            # this function disabled at once.
            raise ApplyRefused(
                f"no proposal {proposal_id!r} to accept")
        if record["status"] != ProposalStatus.PROPOSED.value:
            raise ApplyRefused(
                f"proposal {proposal_id} is already {record['status']}. "
                "Accepting it twice would produce a second revision for one "
                "review")

        latest = store.get_worksheet(worksheet_id, owner)
        if latest is None:
            raise ApplyRefused(f"no worksheet {worksheet_id!r}")
        worksheet = from_json(latest["payload"])

        if worksheet.revision != proposal.source_revision:
            raise StaleProposal(
                f"proposal {proposal_id} was reviewed against revision "
                f"{proposal.source_revision} and the worksheet is now at "
                f"{worksheet.revision}. Re-plan against the current revision "
                "rather than applying a diff to state nobody reviewed")

        # Artifacts first, always. The revision cites them, and a revision that
        # cites a run which does not exist is the dangling reference this whole
        # ordering exists to prevent.
        if proposal.proposed_scenario_patch is not None:
            if run_candidate is None:
                raise ApplyRefused(
                    "a scenario change requires a run, and no runner was "
                    "supplied. No run means no revision")
            # One delivery, recorded once, cited by every candidate. Inside
            # the transaction with the runs, so a fan-out is atomic: there is
            # no committed state where some candidates cite evidence and the
            # rest do not.
            #
            # The event's own `run_id` is left unset for a fan-out — it was
            # resolved for the acceptance, not for any one candidate — and
            # `record_run` accepts that while still refusing an event recorded
            # *for a different* run. Naming one candidate would make the other
            # citations look like the substitution this is built to detect.
            event = getattr(access, "access_event", None)
            event_id = None
            if event is not None:
                event_id = store.record_access_event(event, owner=owner)

            for index, change in enumerate(proposal.changes):
                candidate = change.value
                result = run_candidate(candidate)
                run_id = f"{proposal_id}-run-{index}"
                store.record_run(run_id=run_id, plan_id=worksheet.scenario_ref,
                                 ran_at=at, result=dict(result),
                                 comparison={"candidate": list(candidate)},
                                 access_event_id=event_id)
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
        moved = store.resolve_worksheet_proposal(
            proposal_id, owner, status=ProposalStatus.ACCEPTED.value,
            resolved_at=at, actor=actor, result_revision=updated.revision,
            result_runs=runs)
        # The transition write must have affected exactly one proposal row.
        #
        # This is an invariant guard, not a branch on expected control flow. The
        # three protections before it answer different questions — the row lock
        # says no competing transaction can decide concurrently, the post-lock
        # re-read says the proposal still authorizes acceptance, and the
        # `status = 'PROPOSED'` predicate says the stored state still permits
        # the transition. This one says *this transaction actually performed
        # it*, which none of the others establishes.
        #
        # Rolling back here is what makes the guarantee whole: raising inside
        # the transaction discards the revision and every candidate run, so a
        # caller that cannot claim success also leaves nothing behind.
        if moved != 1:
            if moved == 0:
                raise ProposalConflict(
                    f"proposal {proposal_id} was resolved by another request "
                    "while this one was applying it. Nothing from this attempt "
                    "was kept; read the proposal again to see the outcome that "
                    "stands")
            # Impossible under the primary key, which is exactly why it is
            # worth saying so rather than silently accepting the first row.
            raise TransitionIntegrityError(
                f"resolving proposal {proposal_id} changed {moved} rows where "
                "one was required. A proposal identifies one review, so more "
                "than one row moving means the stored state no longer matches "
                "what this code can reason about")

    return ApplyResult(proposal_id=proposal_id, revision=updated.revision,
                       runs=tuple(runs), derived=tuple(derived))


def reject(store, *, proposal_id: str, owner: str, at: str,
           actor: str = "pilot") -> None:
    """Record that a proposal was declined. It stays readable."""
    store.resolve_worksheet_proposal(proposal_id, owner,
                           status=ProposalStatus.REJECTED.value,
                           resolved_at=at, actor=actor)
