"""The intent chain for one worksheet, loaded from the store and verified.

    load prior intents for this worksheet and owner
        -> validate the chain
        -> plan the current instruction against it
        -> persist

The planner takes history. Until now the caller supplied it, which meant the
live application supplied nothing: every request arrived with an empty list and
looked like the first. A user could try 21-, 63- and 126-day windows across
three requests, then keep one, and each request would classify as
`ANALYTICAL_ONLY` — the trial accounting was implemented and could not run.

**History is not a parameter here, it is a query.** `from_store` is the only
constructor that reaches production, and it scopes by worksheet and owner in
SQL rather than filtering afterwards. A history filtered after loading is a
history that was loaded, and the tests that built chains by hand were exactly
how a fake sequence got assembled from unrelated `plan()` results.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .intent import (
    EditEffect,
    RepetitionSignature,
    SelectionBasis,
    WorksheetIntent,
)
from .intent_chain import chain_link

#: Bumped when a change to the planner would classify a stored instruction
#: differently. Recorded per intent so a chain can say which planner decided it.
PLANNER_VERSION = "1"

#: Bases that mean values were being evaluated against one another. A family
#: containing any of these was a search, however it was assembled.
_SEARCH = frozenset({SelectionBasis.VARIANT_EXPLORATION,
                     SelectionBasis.AFTER_RESULTS})


class ChainStatus(str, Enum):
    VALID = "VALID"
    BROKEN_LINK = "BROKEN_LINK"
    """A stored intent's classification no longer hashes to its successor's
    link. Something edited the history."""

    MISSING_LINK = "MISSING_LINK"
    """A gap in the sequence. An intent was removed."""

    OUT_OF_ORDER = "OUT_OF_ORDER"
    """Revisions move backwards. The chain cannot be a history of one
    worksheet advancing."""


@dataclass(frozen=True)
class ChainVerdict:
    status: ChainStatus
    detail: str = ""

    @property
    def trustworthy(self) -> bool:
        return self.status is ChainStatus.VALID

    def to_json(self) -> Dict[str, str]:
        return {"status": self.status.value, "detail": self.detail,
                "trustworthy": self.trustworthy}


def rehydrate(payload: Mapping[str, Any]) -> WorksheetIntent:
    """A stored intent, back as the planner's own type.

    Rebuilt from the structured record rather than re-planned from the
    instruction. Re-planning would let a later planner version silently restate
    what an earlier one decided, and the stored classification is the thing the
    user was shown.
    """
    signature = payload.get("repetition_signature") or {}
    return WorksheetIntent(
        intent_id=payload["intent_id"],
        source_revision=int(payload["source_revision"]),
        instruction=payload.get("instruction") or "",
        edit_effect=EditEffect(payload["edit_effect"]),
        selection_basis=SelectionBasis(payload["selection_basis"]),
        repetition_signature=RepetitionSignature(
            target_run=signature.get("target_run", ""),
            block_type=signature.get("block_type", ""),
            metric=signature.get("metric", ""),
            parameter_family=signature.get("parameter_family", ""),
            scenario_dimension=signature.get("scenario_dimension", "")),
        target_blocks=tuple(payload.get("target_blocks") or ()),
        requested_parameters=tuple(payload.get("requested_parameters") or ()),
        alternatives_generated=int(payload.get("alternatives_generated", 0)),
        results_visible=bool(payload.get("results_visible", False)),
        related_prior_intents=tuple(payload.get("related_prior_intents") or ()),
        rerun_required=bool(payload.get("rerun_required", False)),
        trial_effect=(None if payload.get("trial_effect") is None
                      else int(payload["trial_effect"])),
        comparability_impact=payload.get("comparability_impact", ""),
        presentation_only=bool(payload.get("presentation_only", False)),
        requires_user_confirmation=bool(
            payload.get("requires_user_confirmation", False)))


@dataclass(frozen=True)
class IntentHistory:
    """One worksheet's intents, in order, with the verdict on their integrity."""

    worksheet_id: str
    owner: str
    intents: Sequence[WorksheetIntent] = ()
    rows: Sequence[Mapping[str, Any]] = ()
    verdict: ChainVerdict = ChainVerdict(ChainStatus.VALID)

    @classmethod
    def from_store(cls, store, worksheet_id: str, owner: str) -> "IntentHistory":
        rows = store.worksheet_intents(worksheet_id, owner)
        return cls(worksheet_id=worksheet_id, owner=owner,
                   intents=tuple(rehydrate(r["structured_request"]) for r in rows),
                   rows=tuple(rows), verdict=verify(rows))

    @property
    def trial_total(self) -> int:
        """Trials accumulated across the whole chain, across sessions.

        The number this record exists to make possible. Read it beside
        `verdict`: a total from an unverified chain is not a smaller total, it
        is a total with no meaning.

        Counted **per repetition family, not by summing each intent's own
        effect.** A search assembled one request at a time undercounts if
        summed: the first window is `ANALYTICAL_ONLY` when it arrives and
        contributes nothing, and the sequence only becomes a search once the
        second arrives — but the first value was evaluated all the same.
        Twenty-one, sixty-three and one-hundred-twenty-six day windows asked
        separately are three trials, not two.

        Retroactive rather than rewritten. Editing the stored classification of
        an earlier intent would break its chain link, and would also restate
        something the user was already shown.
        """
        return sum(self._family_trials(members)
                   for members in self._families().values())

    def _families(self) -> Dict[str, List[WorksheetIntent]]:
        families: Dict[str, List[WorksheetIntent]] = {}
        for one in self.intents:
            families.setdefault(one.repetition_signature.key(), []).append(one)
        return families

    @property
    def unclassified_count(self) -> int:
        """Instructions the planner could not read.

        Reported beside the total rather than folded into it. None of them
        applied — `propose` refuses an unclassified intent — so none added a
        trial; but each was a request whose trial cost is genuinely unknown, and
        a total that absorbed them would look complete."""
        return sum(1 for one in self.intents if not one.classified)

    @property
    def total_is_complete(self) -> bool:
        return self.unclassified_count == 0

    @staticmethod
    def _family_trials(members: Sequence[WorksheetIntent]) -> int:
        # `None` means unknown, and unknown contributes nothing to an *executed*
        # trial count: an unclassified instruction is refused before it runs.
        # It is surfaced separately by `unclassified_count`, so the uncertainty
        # is visible rather than absorbed into a number that looks whole.
        declared = sum(one.trial_effect or 0 for one in members)
        searching = any(one.selection_basis in _SEARCH for one in members)
        if not searching:
            # Nothing in this family was ever a search, so each intent's own
            # count stands. A single substitution is one trial however many
            # instruments it names.
            return declared

        evaluated = {value for one in members
                     for value in one.requested_parameters}
        # The larger of the two. A family that evaluated four distinct values
        # counts four even if the per-intent arithmetic reached three; a family
        # whose declared count is higher keeps it, because a substitution names
        # two instruments and is still one decision.
        return max(len(evaluated), declared)

    def repetitions_of(self, signature_key: str) -> int:
        return sum(1 for i in self.intents
                   if i.repetition_signature.key() == signature_key)

    def to_json(self) -> Dict[str, Any]:
        return {"worksheet_id": self.worksheet_id,
                "intents": [i.to_json() for i in self.intents],
                "trial_total": self.trial_total,
                "unclassified_count": self.unclassified_count,
                "total_is_complete": self.total_is_complete,
                "verdict": self.verdict.to_json()}


def verify(rows: Sequence[Mapping[str, Any]]) -> ChainVerdict:
    """Recompute every link and compare it with what was stored."""
    previous_hash, expected_sequence = "", 1
    for row in rows:
        if int(row["sequence"]) != expected_sequence:
            return ChainVerdict(
                ChainStatus.MISSING_LINK,
                f"expected position {expected_sequence} and found "
                f"{row['sequence']}; an intent was removed from the chain")
        expected_sequence += 1

        recomputed = chain_link(previous_hash, rehydrate(row["structured_request"]))
        if recomputed != row["chain_hash"]:
            return ChainVerdict(
                ChainStatus.BROKEN_LINK,
                f"intent {row['intent_id']} no longer hashes to its stored "
                "link, so its classification was edited after it was recorded")
        previous_hash = row["chain_hash"]

    revisions = [int(row["source_revision"]) for row in rows]
    if any(later < earlier for earlier, later in zip(revisions, revisions[1:])):
        return ChainVerdict(
            ChainStatus.OUT_OF_ORDER,
            "source revisions move backwards, so these intents are not one "
            "worksheet advancing")

    return ChainVerdict(ChainStatus.VALID)
