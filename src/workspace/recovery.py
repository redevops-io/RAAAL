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

#: The three `_with_decisions` erased whenever the owner confirmed anything.
#: It rebuilt `Provenance` naming five of eight fields, and these were the
#: three it did not name — so a plan can carry a `provenance@2` stamp, meaning
#: the serializer supported them, and still have been stored without them.
#:
#: `amended` is absent from this list on purpose: the old rebuild did name it,
#: so an amendment lost to a `@2` plan is not explicable this way.
DROPPED_BY_CONFIRMATION = ("excluded", "asset_resolutions", "time_window")

#: What an absence can mean. More than one applying is the finding: absence
#: stops being interpretable, and a field nobody can interpret cannot be
#: migrated on the system's own authority.
NO_DECISION = "NO_DECISION_WAS_RECORDED"
PREDATES_FIELD = "THE_PLAN_PREDATES_THE_FIELD"
DISCARDED = "DISCARDED_WHEN_AN_INFERENCE_WAS_CONFIRMED"


def confirmation_rebuilt(body) -> bool:
    """Whether this plan went through the rebuild that dropped three fields.

    `_with_decisions` returned its input untouched when nothing was agreed, so
    a stored inference marked `confirmed` is the marker: it is the only thing
    that could have put it there, and it is written by the same call that did
    the dropping.

    Derived from the stored body rather than from a date or a build number.
    Both of those would need a table mapping builds to behaviour, and the
    table is the thing that goes stale.
    """
    provenance = (body or {}).get("provenance")
    entries = ((provenance or {}).get("inferred")
               if isinstance(provenance, dict) else None)
    if entries is None:
        entries = (body or {}).get("inferred") or ()
    return any(isinstance(one, dict) and one.get("confirmed")
               for one in entries)


def _comparable(name: str, value):
    """A stored value reduced to what a recompile could reproduce.

    `agrees` answers one question: does today's compiler read the same words
    the same way. `confirmed` is not a reading — it records that a person
    agreed to an inference — and the recompile below never replays
    confirmations, because confirming is a route concern rather than a
    compiler one.

    Left in, every plan whose owner confirmed anything reported `inferred` as
    disagreeing, for ever, and the stated reason was that the compiler had
    changed. It had not. A false reason on a true refusal is still a false
    reason, and this one would have been read as evidence of drift that does
    not exist.
    """
    if name != "inferred" or not isinstance(value, list):
        return value
    return [{k: v for k, v in one.items() if k != "confirmed"}
            if isinstance(one, dict) else one for one in value]


def _rebuild_was_lossy(body) -> bool:
    """Whether this plan's confirmation actually cost it anything.

    The old rebuild dropped all three fields together — it constructed a
    `Provenance` without naming any of them — so one surviving with a value
    proves the plan was written by a build that kept all three, and the empty
    ones are empty because the owner decided nothing.

    Without this, every plan saved from now on would be reported as needing
    its owner's confirmation for fields that are genuinely and correctly
    empty, for ever. The first version did exactly that to a plan saved by the
    fixed build ten minutes after it deployed.

    When all three are empty the question stays open, which is the honest
    answer: a plan that recorded no exclusions, no asset resolutions and no
    period is indistinguishable from one that had them deleted.
    """
    if not confirmation_rebuilt(body):
        return False
    provenance = (body or {}).get("provenance")
    if not isinstance(provenance, dict):
        return True
    return not any(provenance.get(key) for key in DROPPED_BY_CONFIRMATION)


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
    shape_supports: bool = True
    """Whether the serializer that wrote this plan had the field at all.

    Separate from `stored`, because they answer different questions and were
    conflated at first. A `provenance@2` stamp says the field *could* have
    been written; it does not say it was.
    """

    absence_explained_by: tuple = ()
    """Every reading an absence admits. One entry is an answer; more than one
    means the absence carries no information and the owner must be asked."""

    agrees: Optional[bool] = None
    """Whether the stored value and a fresh compile agree, when both exist.
    `False` is not a recovery problem — it means the compiler now reads the
    same words differently, which is a separate thing an owner must be told
    about before anything is replayed."""

    why: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"field": self.field, "outcome": self.outcome,
                "stored": self.stored, "rederived": self.rederived,
                "shape_supports": self.shape_supports,
                "absence_explained_by": list(self.absence_explained_by),
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
    rebuilt_by_confirmation = _rebuild_was_lossy(body)

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
        key = field.path.split("/")[-1]
        legacy_gap = (shape == "provenance@1"
                      and key in DROPPED_BY_PROVENANCE_1)

        # A `provenance@2` stamp says the serializer had the field. It does
        # not say the field survived to disk: `_with_decisions` rebuilt the
        # provenance from five of eight names and erased these three whenever
        # the owner confirmed anything. So the stamp and the storage are two
        # separate questions, and reading the first as the second is what made
        # the four modern plans look structurally complete.
        #
        # Only ever an explanation for an *absence*. A field that is present
        # and non-empty in a plan whose owner confirmed something demonstrably
        # survived the rebuild — the old code would have erased it — so that
        # plan was written by a build that already preserved it, and treating
        # its value as suspect would refuse a plan that is in fact intact.
        empty = not stored_value
        confirmation_gap = (rebuilt_by_confirmation and empty
                            and key in DROPPED_BY_CONFIRMATION)
        shape_supports = not legacy_gap
        known = stored_present and not legacy_gap and not confirmation_gap

        # What the absence could mean, all of it. One reading is an answer;
        # two or more and the absence says nothing at all, which is a stronger
        # statement than "the field is missing" and calls for a different act.
        readings = []
        if empty or legacy_gap:
            if legacy_gap:
                readings.append(PREDATES_FIELD)
            if confirmation_gap:
                readings.append(DISCARDED)
            if empty and not legacy_gap and not confirmation_gap:
                readings.append(NO_DECISION)
        rederived = fresh_present and fresh_value not in (None, [], (), {})

        agrees = None
        if known and fresh_present:
            agrees = _comparable(field.name, stored_value) == \
                _comparable(field.name, fresh_value)

        if known:
            outcome, why = RECOVERABLE, "present in the stored structured body"
        elif confirmation_gap:
            # The shape stamp says the serializer had this field, so its
            # absence is not explained by the plan's age. It may have held a
            # decision that the confirmation rebuild erased, and there is no
            # way to tell that apart from the owner having decided nothing.
            #
            # An uninterpretable absence is not an empty value. Reading it as
            # one would silently assert that the owner accepted no limitation,
            # chose no fund and stated no period — three claims about consent
            # made on their behalf, from a field that was deleted.
            outcome = NEEDS_OWNER
            why = ("the confirmation rebuild discarded this field, so its "
                   "absence cannot be told apart from no decision having "
                   "been made")
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
        elif questions:
            # Absent, and a recompile did not produce it — but the recompile
            # is also still asking. A derived field can be blocked on an
            # unanswered question rather than lost: the production plan's
            # funding policy needs an instrument, the instrument needs the
            # owner's choice between two funds, and until that is made no
            # policy can be built from any amount of structure.
            #
            # Reported as historical at first, which is the one reading that
            # is actively harmful — it tells an operator to stop, when four
            # answers would recover the field.
            outcome = NEEDS_OWNER
            why = ("absent, and blocked on a question the recompile is still "
                   "asking rather than on anything lost")
        else:
            outcome = HISTORICAL
            why = "absent, and no recompile produces it"

        found.append(FieldRecovery(field=field.name, outcome=outcome,
                                   shape_supports=shape_supports,
                                   absence_explained_by=tuple(readings),
                                   stored=known, rederived=rederived,
                                   agrees=agrees, why=why))

    return PlanRecovery(plan_id=record.get("plan_id", ""),
                        provenance_shape=shape, fields=tuple(found),
                        open_questions=questions,
                        replayed_decisions=len(replayed[0]) + len(replayed[1]))
