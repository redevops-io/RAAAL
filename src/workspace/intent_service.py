"""Planning an instruction against the worksheet's real history.

One entry point, `plan_and_record`, which owns both halves:

    load the chain -> plan against it -> persist -> propose

Splitting them is what let the live route plan against nothing. The planner
still accepts a `history` argument — it is a pure function and its tests need
one — but nothing in the application passes a list it built itself.

This never applies anything. A planner that applied its own proposal decides on
the user's behalf exactly where the user's judgement is the point, so the return
value is a proposal awaiting confirmation and acceptance is a separate call to
the already-proven transaction.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

from . import intent as planner
from .intent_history import PLANNER_VERSION, ChainStatus, IntentHistory
from .proposal import WorksheetProposal, propose
from .worksheet import from_json


class IntentRefused(RuntimeError):
    """The instruction was not planned, and why."""


class UntrustworthyHistory(IntentRefused):
    """The stored chain does not verify.

    Refused rather than planned against a partial history. Continuing would
    produce a classification derived from a chain somebody edited, and it would
    look exactly like one that was not.
    """


class StaleInstruction(IntentRefused):
    """The instruction was written against a revision the worksheet has left."""


def instruction_hash(instruction: str) -> str:
    """The durable identity of what was asked.

    Kept when the sentence itself is not: retention can drop the raw text and
    the chain still shows that *this* instruction, not some other, produced the
    classification.
    """
    return hashlib.sha256(instruction.strip().lower().encode()).hexdigest()


@dataclass(frozen=True)
class PlannedIntent:
    """What one instruction produced, before anything is changed."""

    intent: planner.WorksheetIntent
    proposal: WorksheetProposal
    proposal_id: str
    sequence: int
    history: IntentHistory

    @property
    def trial_total_after(self) -> int:
        return self.history.trial_total + self.intent.trial_effect

    def to_json(self) -> Dict[str, Any]:
        return {"intent": self.intent.to_json(),
                "proposal": self.proposal.to_json(),
                "proposal_id": self.proposal_id,
                "sequence": self.sequence,
                "trial_total_before": self.history.trial_total,
                "trial_total_after": self.trial_total_after,
                "chain": self.history.verdict.to_json()}


def plan_and_record(store, *, worksheet_id: str, owner: str, instruction: str,
                    intent_id: str, proposal_id: str, at: str,
                    source_revision: Optional[int] = None,
                    target_run: str = "",
                    store_instruction: bool = False) -> PlannedIntent:
    """Plan one instruction against the worksheet's persisted chain.

    `source_revision` is read from the stored worksheet by default rather than
    taken from the request. A caller-supplied revision is a caller-supplied
    answer to "was this written against what is there now?", which is the
    question a stale check exists to ask.
    """
    record = store.get_worksheet(worksheet_id, owner)
    if record is None:
        raise IntentRefused(f"no worksheet {worksheet_id!r}")
    worksheet = from_json(record["payload"])

    if source_revision is None:
        source_revision = worksheet.revision
    elif source_revision != worksheet.revision:
        raise StaleInstruction(
            f"this instruction was written against revision {source_revision} "
            f"and the worksheet is now at {worksheet.revision}. Re-issue it "
            "against the current revision rather than planning against state "
            "nobody has seen")

    history = IntentHistory.from_store(store, worksheet_id, owner)
    if not history.verdict.trustworthy:
        raise UntrustworthyHistory(
            f"the intent history for {worksheet_id} does not verify "
            f"({history.verdict.status.value}): {history.verdict.detail}. "
            "Trial accounting derived from it would be a number with no "
            "meaning, so nothing is planned against it")

    intent = planner.plan(
        instruction, intent_id=intent_id, source_revision=source_revision,
        history=history.intents, target_run=target_run,
        # Persisted, never inferred later. Whether the user had seen the
        # figures when they chose is the difference between an analytical
        # request and a result-aware selection, and it cannot be reconstructed
        # after the fact.
        results_visible=bool(worksheet.benchmark_run_refs or history.intents))

    sequence = store.append_worksheet_intent(
        worksheet_id=worksheet_id, owner=owner, intent=intent, created_at=at,
        planner_version=PLANNER_VERSION,
        instruction_hash=instruction_hash(instruction),
        store_instruction=store_instruction)

    proposal = propose(intent, worksheet)
    store.save_worksheet_proposal(proposal_id=proposal_id, owner=owner,
                                  worksheet_id=worksheet_id, proposal=proposal,
                                  created_at=at)
    store.link_intent_proposal(intent.intent_id, owner, proposal_id=proposal_id)

    return PlannedIntent(intent=intent, proposal=proposal,
                         proposal_id=proposal_id, sequence=sequence,
                         history=history)
