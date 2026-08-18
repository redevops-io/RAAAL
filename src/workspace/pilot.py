"""The seam: a user's sentence, through the runtime that was built for it.

Until this module existed, `compile_intent` was called from tests and nowhere
else. Every workspace route ran `compile_scenario` — the legacy regex compiler
the migration exists to replace — so the Discovery → fusion → Mission pipeline
was correct, tested, and unreachable from any path a user could take.

    text
      → hosted reader                 (the only witness this profile has)
      → fusion                        (decides; syntax absent, not silent)
      → VerifiedIntent, sealed        (the boundary artifact)
      → compile_intent                (a plan, or refusals by name)

**Model-only is a deployment profile.** `WitnessProfile` carries it and the
persisted record says `MODEL_ONLY_ACCEPTED` rather than `AGREE`, because a
pilot reporting agreement while running one reader would be claiming
corroboration it never had.

**Nothing here re-reads the sentence after sealing.** The intent is stored with
the plan, and reopening compiles from the stored intent. That is the property
the whole migration is built on, and this is the first place a person can
exercise it.

**What this module deliberately does not do.** It does not fall back to
`compile_scenario`. A deployment chooses one interpreter and says which; a
runtime that silently degraded to the legacy grammar would hand two users
different products under one deployment, which is the rule `ParserFallback`
already states for the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence

from runtime_contracts import (
    Author,
    IntentField,
    OpenReason,
    Unresolved,
    VerifiedIntent,
)

from ..discovery.canonical import canonicalise
# `Fusion` only, and from the runtime that owns it. `fuse` and
# `Proposal` left with the cutover: this module no longer makes
# fusion decisions, it reads them.
from discovery_runtime.fusion import Fusion
from ..discovery.reader import ReadingSet, Schema
from ..discovery.schema import QUANTIFY_SCHEMA
from ..discovery.witnesses import MODEL_ONLY, SettledField, WitnessProfile, record
from ..mission.capability import Refusal
from ..mission.from_intent import Compiled, NotExecutable, compile_intent

INTERPRETER_VERSION = "quantify-pilot-interpreter@1"

#: Canonicalisation names its authors as strings so `discovery` need not import
#: the contracts package to say who settled a value. Mapped here, once.
_AUTHORS = {"MODEL": Author.MODEL, "READER": Author.READER,
            "USER": Author.USER, "DEFAULT": Author.DEFAULT}


class InterpreterUnavailable(RuntimeError):
    """The only witness this profile has could not be reached.

    Raised rather than degraded. A deployment that quietly parsed with the
    legacy grammar instead would be serving two different products under one
    name, and the user who got the narrower one would never be told.
    """


@dataclass(frozen=True)
class PilotReading:
    """Everything the runtime concluded about one submission."""

    text: str
    intent: Optional[VerifiedIntent]
    compiled: Optional[Compiled]
    settled: Sequence[SettledField] = ()
    open_fields: Sequence[str] = ()
    """Dimensions fusion could not settle — a genuine ambiguity or a
    disagreement. These are questions."""

    absent_fields: Sequence[str] = ()
    """Dimensions the reader looked for, the sentence did not carry, and
    nothing needs to ask about.

    **Not questions.** "I looked and it does not say" is a reading, and Mission
    answers it by applying a declared default *and saying so*, or by refusing
    the dimension by name. Presenting them as questions would ask a person
    eleven things about one sentence, and ten of the answers would be "I do not
    mind" — which is exactly the state a declared default already expresses."""

    refusals: Sequence[Any] = ()
    profile: WitnessProfile = MODEL_ONLY

    rejected_answers: Mapping[str, str] = field(default_factory=dict)
    """Dimensions the person answered which the answer did not settle, and why.

    A clarification round must strictly reduce what is unresolved, change what
    would execute, or end in a refusal. When an answer settles nothing, the
    page would otherwise render the identical question — and the pilot found
    exactly that: `amount` was asked, answered "1000 usd", and asked again,
    because the numeric reader could not read a currency suffix. The person
    sees their own words come back as an unanswered question, forever.

    Naming the rejection is what makes the round progress even when the value
    does not: "I could not read '1000 usd' as an amount" is a different state
    from "how much are you contributing", and the person can act on it."""

    def clarification_state(self) -> tuple:
        """What must change between rounds, or the conversation is stuck.

        Deliberately the *executable* identity rather than the plan's data, so
        two rounds differing only in how a holding was spelled do not count as
        progress."""
        identity = ""
        scenario = getattr(self.compiled, "scenario", None)
        if scenario is not None:
            from hashlib import sha256
            from json import dumps
            identity = sha256(dumps(scenario.execution_form(), sort_keys=True,
                                    default=str).encode()).hexdigest()[:16]
        return (tuple(sorted(self.open_fields)),
                tuple(sorted(getattr(r, "dimension", "")
                             for r in self.refusals)),
                identity)
    interpreter_version: str = INTERPRETER_VERSION
    reader_id: str = ""

    @property
    def executable(self) -> bool:
        return self.compiled is not None and self.compiled.executable

    @property
    def needs_input(self) -> Sequence[str]:
        """Dimensions Mission refuses for want of a value.

        `assets` is the recurring one: "the intent names nothing to hold, so
        there is no plan to compile — this is a missing statement, not missing
        data". That *is* a question, and it comes from the manifest rather than
        from the reader."""
        return tuple(sorted({r.dimension for r in self.refusals
                             if getattr(r, "kind", "") == "UNRESOLVED_INPUT"}))

    @property
    def questions(self) -> Sequence[str]:
        """What the page actually asks. Two sources, one list."""
        return tuple(sorted(set(self.open_fields) | set(self.needs_input)))

    @property
    def needs_answers(self) -> bool:
        return bool(self.questions)

    def to_json(self) -> dict:
        """What the plan stores. The intent is carried whole, not by hash — a
        hash proves it has not changed and cannot reconstruct it, and reopening
        would then have to re-read the sentence."""
        return {
            "interpreter_version": self.interpreter_version,
            "reader_id": self.reader_id,
            "profile": self.profile.to_json(),
            "intent": None if self.intent is None else self.intent.to_json(),
            "settled": [f.to_json() for f in self.settled],
            "open_fields": list(self.open_fields),
            "absent_fields": list(self.absent_fields),
            "refusals": [{"kind": r.kind, "dimension": r.dimension,
                          "detail": r.detail} for r in self.refusals],
            "derivation": ({} if self.compiled is None
                           else dict(self.compiled.derivation)),
        }


def read(text: str, reader, *, schema: Schema = QUANTIFY_SCHEMA,
         profile: WitnessProfile = MODEL_ONLY,
         syntax_reader=None,
         objective: str = "evaluate_investment_strategy",
         utterance_ref: str = "") -> PilotReading:
    """One submission, through the runtime.

    `reader` is injected rather than constructed so the route can pass the
    deployment's configured reader and a test can pass a recorded one — the
    same reason `pipeline.read` takes its witnesses instead of fetching them.

    **`syntax_reader` does not decide meaning.** When present, the utterance
    goes through `pipeline.read`, where syntax enters `fuse` as *evidence*
    beside the model's proposal. Fusion still only proceeds on `AGREE`, and
    "syntax proposed a value the model never mentioned" is `DISAGREE` — so a
    structural witness can withhold a reading but can never mint one. Without
    that rule this would be a second semantic compiler, which is the thing the
    architecture exists to avoid.

    Why it is here at all: two recordings of one model, same prompt version,
    differed on 24 of 36 corpus sentences. One inverted `persistent_condition`
    to `crossing_event`. A single stochastic witness cannot hold an executable
    meaning still between runs, and syntax is what stops it moving unnoticed.
    """
    reading: ReadingSet = reader.read(text, schema)
    if not reading.ok:
        raise InterpreterUnavailable(
            f"{reading.reader_id} did not answer: {reading.failed}")

    if syntax_reader is not None:
        # Both witnesses, through the official runtime.
        #
        # The asymmetry survives in how the evidence is handed over rather than
        # in a fusion of our own: syntax agreeing becomes supporting evidence,
        # syntax contradicting becomes a contradiction, and a dimension the
        # model never read gets no proposal — which `fuse` answers with
        # DISAGREE. Syntax argues; it never authors a field.
        #
        # No fallback. If the runtime cannot read this, that is a failure to
        # see, not a reason to quietly run a second implementation.
        from ..discovery.adapter import (decisions_via_runtime,
                                         deterministic_witness)
        from ..discovery.guards import as_decisions

        parse = syntax_reader.parse(text)
        syntax_evidence, derived_by_field = deterministic_witness(text, parse)
        decisions = decisions_via_runtime(
            reading, syntax_evidence=syntax_evidence, derived=derived_by_field)

        # A material action the sentence states and the reader dropped. Four
        # live draws of five read `sell the loser and buy a similar fund` and
        # Mission refused it by name; the fifth read no sell at all and
        # produced an executable plan. The dimension and the refusal both
        # existed — nothing downstream simply had anything to refuse.
        decisions.extend(as_decisions(parse, decisions))
    else:
        # The derived readers run here too, and until now they did not.
        #
        # `pipeline.read` runs them, and this branch never calls it — so on a
        # deployment with no deterministic parser, which is every deployment
        # this project actually serves, no derived reader had ever run. Three
        # of them: trigger semantics, weight binding, day of month. Built,
        # tested, and unreachable by a single user.
        #
        # `weight_binding` was rewritten to read from the sentence rather than
        # from a parse *because* production has no Stanza. It then sat behind
        # the branch that only runs when Stanza is present. A reader nobody
        # reaches is the same defect as a login route nothing links to, and
        # this is the third of its kind found by somebody using the site.
        #
        # They take no candidates and no parse here, which is what reading
        # from the sentence means. A reader that needs a parse simply returns
        # nothing, exactly as it does when the parse is silent.
        from ..discovery.derived_readers import DERIVED_READERS

        # One reading per SET dimension before anything indexes by name.
        #
        # `{p.dimension: p for p in proposals}` kept the last, so a reader that
        # emitted 'bonds' and 'stocks' for `assets` — which is what it does for
        # "take from bonds in a down year and from stocks otherwise" — produced
        # a plan naming only stocks. Silent, and the sentence names both.
        #
        derived_by_field = {}
        for _reader_id, derive in DERIVED_READERS:
            found = derive((), None, text)
            if found is not None:
                derived_by_field[found.dimension] = found

        # One witness, through the same runtime. Not a special case: a profile
        # with nothing to argue is one where nothing argues, which an empty
        # syntax mapping already says.
        from ..discovery.adapter import decisions_via_runtime

        decisions = decisions_via_runtime(reading, derived=derived_by_field)

    settled = record(decisions, profile)

    # Open and absent are different facts and only one of them is a question.
    #
    # The first version merged them, and "invest $500 monthly" came back with
    # eleven open fields — every dimension the reader had looked for and not
    # found. Ten of those have declared engine defaults or named refusals, so
    # asking about them would put eleven questions in front of a person whose
    # sentence was four words long, and ten of the answers would be "I do not
    # mind" — which is what a declared default already says.
    open_fields = tuple(sorted(d.dimension for d in decisions
                               if not d.proceeds))
    absent_fields = tuple(sorted(reading.unread))

    intent, unreadable = _intent(text, decisions, reading, objective=objective,
                                 utterance_ref=utterance_ref or _ref(text))

    compiled, refusals = None, ()
    if intent is not None and intent.is_verified:
        try:
            compiled = compile_intent(intent)
            refusals = compiled.refusals
        except NotExecutable as refused:
            refusals = refused.refusals

    # A value stated and unreadable is refused by name, not asked about.
    #
    # It reaches the page from here rather than from Mission, which is the
    # move: deciding that "200usd" cannot be read is a question about the
    # words, and Mission no longer reads words. The refusal keeps the reason
    # Discovery gave it — the generic "readers disagreed" that a blocked
    # dimension would otherwise carry says nothing a person can act on.
    refusals = tuple(refusals) + tuple(
        Refusal(kind="UNRESOLVED_INPUT", dimension=name, detail=why)
        for name, why in unreadable
        if not any(r.dimension == name for r in refusals))

    built = PilotReading(
        text=text, intent=intent, compiled=compiled, settled=settled,
        open_fields=open_fields, absent_fields=absent_fields,
        refusals=refusals, profile=profile, reader_id=reading.reader_id)
    # Absent means "and nothing asks about it". A dimension that is absent and
    # required is a question; leaving it in both would tell a reader of the
    # plan that one field was quietly defaulted *and* explicitly requested.
    return replace(built, absent_fields=tuple(
        f for f in built.absent_fields if f not in set(built.questions)))


def _relation_fields(reading: ReadingSet) -> dict:
    """Relation kinds, as declared fields, so the manifest can refuse them.

    `compile_intent` builds what it asks the manifest about from
    `intent.fields`, which is a flat name -> value map of *dimensions*. A
    relation is not a dimension, so `reserve_policy` and `bucket_policy` would
    have been readable by Discovery and invisible to Mission — a refusal that
    exists in the manifest and never fires, which is the same defect as no
    refusal at all.

    The relation itself stays structured on the reading; this adds only a
    marker under the relation's own name, so the refusal names the thing the
    person described rather than some dimension it was flattened into.
    """
    summary = {}
    for relation in getattr(reading, "relations", ()) or ():
        kind = getattr(relation, "kind", "")
        if not kind:
            continue
        members = ", ".join(
            f"{role}={subject}" for role, subject, *_ in
            (m if isinstance(m, (tuple, list)) else (m, "", "")
             for m in getattr(relation, "members", ())))
        summary[kind] = members or kind
    return summary


def _intent(text: str, decisions, reading: ReadingSet, *, objective: str,
            utterance_ref: str) -> Optional[VerifiedIntent]:
    """The boundary artifact, sealed when its meaning is closed.

    Sealing is attempted, never asserted. `seal()` refuses while a
    result-changing dimension is open, and catching that refusal is how the
    page learns it has a question to ask — rather than the page deciding for
    itself that the answer is good enough.
    """
    settled = {d.dimension: d.value for d in decisions if d.proceeds}
    settled.update(_relation_fields(reading))
    unresolved = tuple(
        Unresolved(dimension=d.dimension,
                   reason=(OpenReason.UNRESOLVED_DISAGREEMENT
                           if d.outcome is not Fusion.AMBIGUOUS_BY_LANGUAGE
                           else OpenReason.NOT_ASKED),
                   detail=d.detail, result_changing=d.material)
        for d in decisions if not d.proceeds and d.dimension not in settled)
    unresolved += tuple(
        Unresolved(dimension=name, reason=OpenReason.NOT_ASKED,
                   detail="the reader was asked and did not answer",
                   result_changing=False)
        for name in reading.unread if name not in settled)

    # Canonical before sealed, because a seal over prose is not a seal.
    #
    # Every value here is now in the form a consumer can act on without reading
    # it: a cadence is one of six names, an amount is a plain decimal, holdings
    # are comma-separated, and a negated disposal has become `sells_allowed`.
    # Mission used to do this work itself, six times, after the meaning was
    # supposed to be closed — so the same sealed artifact could compile
    # differently as that code moved.
    canonical = canonicalise(settled)

    # A stated value that cannot be read blocks the seal rather than being
    # dropped. Dropping it would leave the dimension absent, and absent means
    # "the engine may apply its default" — so an unreadable cadence would
    # quietly become a plan that runs once.
    unresolved += tuple(
        Unresolved(dimension=name, reason=OpenReason.UNRESOLVED_DISAGREEMENT,
                   detail=why, result_changing=True)
        for name, why in canonical.refusals)

    # Relations, structurally, not only as the flat markers `_relation_fields`
    # adds for Mission's compiler.
    #
    # `canonical_form` includes them and says why: "a relationship *is* part of
    # what was asked for. Two intents naming the same instruments in different
    # roles are different requests." Dropping them meant this path could not
    # tell `from=traditional IRA, to=Roth` from the reverse — both hashed the
    # same, because the marker records only that an account_transition exists.
    from ..discovery.adapter import as_intent_relation

    draft = VerifiedIntent(
        objective=objective,
        produced_by=f"{reading.reader_id}+{INTERPRETER_VERSION}",
        utterance_ref=utterance_ref,
        fields={name: IntentField(value=value, author=_AUTHORS[author])
                for name, (value, author) in canonical.fields.items()},
        relations=tuple(as_intent_relation(r)
                        for r in getattr(reading, "relations", ()) or ()),
        unresolved=unresolved)
    try:
        return draft.seal(), canonical.refusals
    except Exception:                                          # NotSealable
        # Not an error. A draft is what an unanswered question looks like, and
        # the page is about to ask it.
        return draft, canonical.refusals


def answer(reading: PilotReading, answers: Mapping[str, Any]) -> PilotReading:
    """A human settles what the reader left open.

    The amendment is authored `USER`, which is the point: the artifact says
    which values a person chose and which a model proposed, and Mission's
    defaults are a third thing again. Collapsing them would make "the user
    agreed" indistinguishable from "we stopped asking".
    """
    return settle(reading, answers, author=Author.USER,
                  provenance="USER_ANSWERED", witness="user",
                  detail="answered on the plan page")


def settle(reading: PilotReading, values: Mapping[str, Any], *,
           author: "Author", provenance: str, witness: str,
           detail: Any = "") -> PilotReading:
    """Supply values for what the reader left open, saying who supplied them.

    One body, because a catalogue assumption and a typed answer must travel the
    same path: both amend the intent, both reseal it, both recompile, and both
    have to survive the open/absent split. Two implementations would drift, and
    the one that drifted would be the one nobody typed into.

    What differs is authority, and it is a parameter rather than a branch.
    `Author.USER` dominates and is never overwritten by a re-read;
    `Author.DEFAULT` is "nobody asserted it, a declared default applied — the
    value a consumer is most entitled to question". A catalogue assumption is
    the second of those and must never be recorded as the first, or the product
    ends up offering its own guess back as the user's choice.

    `detail` may be a string or a mapping from dimension to string, because an
    assumption's reason is specific to the dimension it supplies and an answer's
    is not.
    """
    if reading.intent is None:
        return reading

    # Canonicalised on the way in, exactly like a value the reader produced.
    #
    # Without this an answer typed into the page bypassed the layer that reads
    # notation: somebody asked what to contribute, told "200usd", and settled a
    # field Mission would then have to parse. The whole point of canonicalising
    # at the seal is that there is one form downstream, so every door into the
    # intent uses the same one.
    supplied = {name: value for name, value in values.items()
                if value not in (None, "")}
    canonical = canonicalise(supplied)

    fields = dict(reading.intent.fields)
    for name, (value, _author) in canonical.fields.items():
        fields[name] = IntentField(value=value, author=author)
    # An answer that cannot be read settles nothing. Reported as rejected, which
    # is what `rejected_answers` already means, so the page says "that is not a
    # number I can use" rather than accepting it and asking again — the loop
    # that ran forever on `1000 usd`.
    unreadable = {name: why for name, why in canonical.refusals}
    answers = {name: value for name, value in supplied.items()
               if name not in unreadable}

    still_open = tuple(u for u in reading.intent.unresolved
                       if u.dimension not in answers)

    # The open/absent split has to survive an amendment. The first version
    # derived `open_fields` from every remaining `Unresolved`, so answering one
    # question turned the ten absent dimensions into ten new ones — the same
    # conflation as `read()`, one step later, where it would have looked like
    # the page inventing questions in response to being answered.
    absent = set(reading.absent_fields)
    draft = VerifiedIntent(
        objective=reading.intent.objective,
        produced_by=reading.intent.produced_by,
        utterance_ref=reading.intent.utterance_ref,
        fields=fields, unresolved=still_open)

    try:
        intent = draft.seal()
    except Exception:                                          # NotSealable
        intent = draft

    compiled, refusals = None, ()
    if intent.is_verified:
        try:
            compiled = compile_intent(intent)
            refusals = compiled.refusals
        except NotExecutable as refused:
            refusals = refused.refusals

    settled = list(reading.settled) + [
        SettledField(field=name, value=value, provenance=provenance,
                     witnesses=[witness],
                     detail=(detail.get(name, "") if hasattr(detail, "get")
                             else detail))
        for name, value in answers.items() if value not in (None, "")]

    # Which answers settled nothing. Computed from the refusals the recompile
    # produced, not from a guess: a dimension the person just answered that is
    # still refused as unresolved input is an answer the runtime could not use.
    rejected = dict(unreadable)
    for refusal in refusals:
        name = getattr(refusal, "dimension", "")
        if name in answers and answers[name] not in (None, ""):
            rejected[name] = getattr(refusal, "detail", "") or "not accepted"

    return PilotReading(
        text=reading.text, intent=intent, compiled=compiled,
        rejected_answers=rejected,
        settled=tuple(settled),
        open_fields=tuple(sorted(u.dimension for u in still_open
                                 if u.dimension not in absent)),
        absent_fields=tuple(sorted(absent - set(answers))),
        refusals=refusals, profile=reading.profile,
        reader_id=reading.reader_id)


def reopen(stored: Mapping[str, Any]) -> PilotReading:
    """A stored plan, recompiled from its pinned intent.

    **No fresh interpretation.** The reader is not called, the sentence is not
    re-read, and nothing here can reach one — this function takes a dict. That
    is the property the migration was built for: a plan reopened after a model
    upgrade is the plan the user confirmed, not a fresh request wearing an old
    name.
    """
    from runtime_contracts import intent_from_json

    payload = stored.get("intent")
    if payload is None:
        raise KeyError(
            "this plan carries no pinned intent; it was created before the "
            "runtime was wired in, and reopening it would mean re-reading the "
            "sentence — which is a different request with the same name")

    intent = intent_from_json(payload)
    compiled, refusals = None, ()
    if intent.is_verified:
        try:
            compiled = compile_intent(intent)
            refusals = compiled.refusals
        except NotExecutable as refused:
            refusals = refused.refusals

    return PilotReading(
        text=stored.get("text", ""), intent=intent, compiled=compiled,
        settled=tuple(SettledField(**f) for f in stored.get("settled", ())),
        open_fields=tuple(stored.get("open_fields", ())),
        refusals=refusals,
        profile=MODEL_ONLY, reader_id=stored.get("reader_id", ""))


def _ref(text: str) -> str:
    """A stable handle for the utterance, without carrying the utterance.

    `VerifiedIntent` holds a reference and never the sentence, which is what
    makes "nothing below re-reads the prose" a property of the artifact rather
    than a rule to remember.
    """
    from hashlib import sha256

    return f"utt-{sha256(text.encode()).hexdigest()[:16]}"
