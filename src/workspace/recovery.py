"""What of a saved plan can be rebuilt, and on whose authority.

A plan saved before this branch may be missing two different kinds of thing,
and they must not be treated alike:

* **a derivation the compiler got wrong or never ran** — the funding policy
  that F11 dropped, the time window that `resolve` was never called for. These
  are functions of the user's own words and the parse they confirmed. Nothing
  was lost; a compile was wrong. Recompiling restores them.

* **a decision the user made that was never written down** — an answer, an
  acknowledgement, a choice between two funds. `provenance@1` serialized four
  of its eight fields, so these exist on disk only inside `stated`, as
  sentences composed for a screen. Recompiling cannot restore them, and
  reading them back out of those sentences would reverse the direction of
  authority: a rendered sentence is produced from a decision and may never be
  turned back into one.

So the outcomes are three, and only the first may be applied without asking:

    RECOVERABLE_FROM_STRUCTURE       already stored, or deterministic from the
                                     description and the pinned parse
    REQUIRES_OWNER_CONFIRMATION      unknown, and a recompile asks for it —
                                     the owner can answer it again
    UNRECOVERABLE_WITHOUT_PROSE      unknown, and nothing asks — the only
                                     trace is display text, so it stays history

**The absent-versus-empty distinction is the whole problem.** In a `@2` body
`"amended": []` means the user answered nothing. In a `@1` body the key is
missing, which means nobody knows. Those look identical to any code that reads
`body.get("amended") or ()`, and they mean opposite things — which is why
`PROVENANCE_SHAPE` is stamped on every write and why this module branches on
it rather than on emptiness.

**This reports; it does not migrate.** `migrate_plan` decides what to do, under
a named authorisation. A field marked recoverable here is a field a migration
*may* replay, not one it has replayed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

RECOVERABLE = "RECOVERABLE_FROM_STRUCTURE"
NEEDS_OWNER = "REQUIRES_OWNER_CONFIRMATION"
HISTORICAL = "UNRECOVERABLE_WITHOUT_PROSE"

#: Bumped when the classification rules change, so a recorded matrix says
#: which rules produced it. A matrix with no version is a matrix that cannot
#: be compared with the next one.
RECOVERY_VERSION = "plan-recovery@1"


def _at(body: Any, path: str) -> Tuple[bool, Any]:
    """Follow a slash path, reporting presence separately from value.

    Returns `(present, value)`. `(True, [])` and `(False, None)` are the two
    readings this module exists to keep apart.
    """
    node = body
    for step in path.split("/"):
        if not isinstance(node, dict) or step not in node:
            return False, None
        node = node[step]
    return True, node


@dataclass(frozen=True)
class Field:
    """One result-changing field, and where it lives in a stored body."""

    name: str
    path: str
    decided_by_user: bool
    """True when the value records something a person chose. Those can never
    be re-derived, only re-asked. False means a function of the description and
    the parse, which a recompile reproduces exactly."""


#: The four keys `Provenance.to_json` did not write before `3eaa5eb`. It wrote
#: `stated`, `inferred`, `contradictions` and `unresolved` and nothing else, so
#: those four *are* trustworthy in a `@1` body and these four carry no
#: information at all — not even when a later writer supplies them.
#:
#: Named as a set rather than treated as "everything under provenance": the
#: first version of this module used the broader rule and reported `inferred`
#: as unknown in a legacy body, which is false. It reached the right outcome
#: by another route, and the reason an operator reads would have been wrong.
DROPPED_BY_PROVENANCE_1 = ("amended", "excluded", "asset_resolutions",
                           "time_window")

#: Every key of `ScenarioSpecification.semantic_form`, which is the set the
#: preview/save equivalence gate compares. Anything that can change a result
#: is in one list or the other; keeping them the same set is what stops this
#: matrix from quietly omitting a field the gate protects.
FIELDS: Sequence[Field] = (
    Field("held_assets", "methodology/allocation_rule/assets", False),
    Field("weighting", "methodology/allocation_rule/weighting", False),
    Field("funding", "flows/funding", False),
    Field("flows", "flows/cadence", False),
    Field("event_program", "methodology/event_program", False),
    Field("holdings_policy", "methodology/holdings_policy", False),
    Field("tax_treatment", "protocol/tax_treatment", False),
    Field("benchmark_set", "protocol/benchmark_set", False),
    Field("spec_version", "spec_version", False),
    Field("inferred", "provenance/inferred", False),
    # The four `provenance@1` dropped. Three are decisions; the fourth is a
    # reading of the user's own sentence and needs nobody's permission.
    Field("time_window", "provenance/time_window", False),
    Field("amendments", "provenance/amended", True),
    Field("exclusions", "provenance/excluded", True),
    Field("asset_resolutions", "provenance/asset_resolutions", True),
)


@dataclass(frozen=True)
class FieldRecovery:
    field: str
    outcome: str
    stored: bool
    rederived: bool
    agrees: Optional[bool]
    """Whether the stored value and a fresh compile agree, when both exist.
    `False` is not a recovery problem — it means the compiler now reads the
    same words differently, which is a separate thing an owner must be told
    about before anything is replayed."""

    why: str

    def to_json(self) -> Dict[str, Any]:
        return {"field": self.field, "outcome": self.outcome,
                "stored": self.stored, "rederived": self.rederived,
                "agrees": self.agrees, "why": self.why}


@dataclass(frozen=True)
class PlanRecovery:
    plan_id: str
    provenance_shape: str
    fields: Sequence[FieldRecovery]
    open_questions: Sequence[str]
    replayed_decisions: int = 0
    """How many of the owner's recorded decisions the comparison replayed.

    Reported rather than inferred. Zero against a plan whose owner answered
    questions means the comparison was made against a reading of the
    description alone, and every `agrees: false` below is explained by that
    rather than by the compiler having changed.
    """

    version: str = RECOVERY_VERSION

    @property
    def automatic(self) -> bool:
        """Whether a migration could replay this plan without asking anyone."""
        return all(one.outcome == RECOVERABLE and one.agrees is not False
                   for one in self.fields)

    def by_outcome(self, outcome: str) -> Sequence[str]:
        return tuple(one.field for one in self.fields if one.outcome == outcome)

    def to_json(self) -> Dict[str, Any]:
        return {"plan_id": self.plan_id, "version": self.version,
                "provenance_shape": self.provenance_shape,
                "automatic": self.automatic,
                "replayed_decisions": self.replayed_decisions,
                "open_questions": list(self.open_questions),
                "fields": [one.to_json() for one in self.fields]}


def _persisted_decisions(body: Dict[str, Any]):
    """The user's own decisions, read from structure only.

    Every value here comes from a key a serializer wrote from a typed object.
    Nothing is taken from `stated`, which holds the same decisions rendered as
    sentences — `"account_type: TAXABLE (answered)"` — and a sentence is
    produced from a decision, never turned back into one.
    """
    from ..mission.spec import ScenarioAmendment, ScenarioExclusion

    provenance = body.get("provenance") or {}
    amendments = tuple(
        ScenarioAmendment(question_id=str(one.get("question_id", "")),
                          answer=str(one.get("answer", "")),
                          recorded_at=str(one.get("recorded_at", "")))
        for one in (provenance.get("amended") or ())
        if isinstance(one, dict) and one.get("question_id"))
    exclusions = tuple(
        ScenarioExclusion(item=str(one.get("item", "")),
                          reason=str(one.get("reason", "")),
                          decision=str(one.get("decision")
                                       or "PROCEED_WITHOUT_MODELLING"),
                          acknowledged_at=str(one.get("acknowledged_at", "")))
        for one in (provenance.get("excluded") or ())
        if isinstance(one, dict) and one.get("item"))
    return amendments, exclusions


def assess(record: Dict[str, Any], *, context: str = "plan recovery") -> PlanRecovery:
    """Classify every result-changing field of one saved plan.

    `record` is a row from `WorkspaceStore.get_plan`. Two of its columns are
    authoritative original input and are used as such: `stated_text` is the
    user's own words, and `parse` is the stage 1 reading they confirmed. The
    recompile below replays exactly those two and nothing else — no
    amendments, no exclusions — so what it produces is what the description
    alone determines, which is precisely the "no decision required" test.

    The rendered `stated` sentences are never read.
    """
    import json

    from ..mission.spec import provenance_shape_of
    from .draft import compile_draft

    body = record.get("scenario") or {}
    if isinstance(body, str):
        body = json.loads(body)
    shape = provenance_shape_of(body.get("provenance"))

    parsed = None
    stored_parse = record.get("parse")
    if stored_parse:
        from ..mission.parse_model import parse_from_stored

        try:
            payload = (json.loads(stored_parse)
                       if isinstance(stored_parse, str) else stored_parse)
            parsed = parse_from_stored(payload, record["stated_text"])
        except (ValueError, KeyError, TypeError):
            parsed = None       # unpinnable; the recompile re-reads the words

    # Replayed, not re-decided. A `@2` body holds the user's answers as
    # structure, and a recompile that ignored them would compare the stored
    # plan against a reading of the description *without* the decisions that
    # shaped it — every answered plan would then look like it had drifted.
    #
    # A `@1` body holds none, so nothing is replayed and the comparison is
    # honest about that: the fields it cannot supply are the fields it will be
    # asked about below. This is the whole difference between the two shapes,
    # expressed as behaviour rather than as a note.
    replayed = _persisted_decisions(body) if shape != "provenance@1" else ((), ())
    fresh = compile_draft(record["stated_text"], name="recovery-probe",
                          parsed=parsed, amendments=replayed[0],
                          exclusions=replayed[1], context=context).scenario
    fresh_body = fresh.to_json()

    questions = tuple(one.field for one in fresh.provenance.unresolved)

    found = []
    for field in FIELDS:
        stored_present, stored_value = _at(body, field.path)
        fresh_present, fresh_value = _at(fresh_body, field.path)

        # A `@1` provenance body did not write these keys at all, so in the
        # ordinary case their absence is what marks them unknown and this
        # branch changes nothing — a mutation removing it survives the legacy
        # fixture, which is how it was found to be doing no work there.
        #
        # It does work in one case: a body carrying `"amended": []` and no
        # `shape`. `@1` never wrote that key and `@2` always stamps the shape,
        # so such a body was not written by this system, and its emptiness is
        # a claim from an unknown source rather than a record of a decision.
        # Unknown provenance is not consent to migrate.
        legacy_gap = (shape == "provenance@1"
                      and field.path.split("/")[-1] in DROPPED_BY_PROVENANCE_1)
        known = stored_present and not legacy_gap
        rederived = fresh_present and fresh_value not in (None, [], (), {})

        agrees = None
        if known and fresh_present:
            agrees = stored_value == fresh_value

        if known:
            outcome, why = RECOVERABLE, "present in the stored structured body"
        elif field.decided_by_user:
            if questions:
                outcome = NEEDS_OWNER
                why = ("a decision that was never written down; a recompile "
                       "asks for it again")
            else:
                outcome = HISTORICAL
                why = ("a decision that was never written down, and nothing "
                       "asks for it; its only trace is display text")
        elif rederived:
            outcome = RECOVERABLE
            why = ("absent, and determined by the description and the pinned "
                   "parse alone")
        else:
            outcome = HISTORICAL
            why = "absent, and no recompile produces it"

        found.append(FieldRecovery(field=field.name, outcome=outcome,
                                   stored=known, rederived=rederived,
                                   agrees=agrees, why=why))

    return PlanRecovery(plan_id=record.get("plan_id", ""),
                        provenance_shape=shape, fields=tuple(found),
                        open_questions=questions,
                        replayed_decisions=len(replayed[0]) + len(replayed[1]))
