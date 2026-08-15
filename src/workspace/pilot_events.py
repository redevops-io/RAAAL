"""What the pilot can count, and deliberately nothing else.

Seven events, each an objective fact about what happened:

    plan_compiled       a sentence became a plan
    plan_saved          a plan was persisted
    plan_reopened       a saved plan was opened again
    plan_result_shown   the plan produced a figure
    plan_refused        it did not, or a capability was refused by name
    plan_resubmitted    they typed again after seeing a reading
    left_for_legacy     they went back to the old workspace

The last two are later additions and both answer a question the first five
could not. `plan_compiled` staying flat looks the same whether a cohort tried
once and left or never arrived; a departure is only visible if something
records it. And a revision chain is the closest thing to a usability
measurement that requires no opinion from anybody — a person who retypes has
been told something they did not accept.

**Structured context only, and never raw user text.** `refusal_code` and
`capability` are named reasons the engine gave; `open_question_count` is a
number. The first version stored the `unavailable` *message* — rendered copy,
which can embed whatever the user typed, and which changes whenever the wording
does. A telemetry field that moves when a sentence is reworded is a field
nobody can count across a cohort.

Every event carries `parser_mode` and `pilot_reader`, so a model-only cohort and
a future dual-witness one can be separated later. Without that, the two would be
one undifferentiated population by the time anyone thought to ask.

**What is not here, and will not be.** Trust, understanding, confusion,
satisfaction. Those are interview questions, and a telemetry field named
`understood_refusal` would be a number derived from nothing — the
classification-without-evidence defect at the point where it would do the most
damage, in the data used to decide whether the runtime was worth building.

A dashboard reporting "82% understood the refusal" is worse than no dashboard,
because somebody will believe it. What this module can honestly say is "of 40
plans, 12 were refused for `assets` and 3 of those were never saved" — and the
question of whether those three people understood why is asked of them.

**Read-only for analysis.** No route exposes these; `summary()` is for whoever
is running the pilot, at a prompt.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

#: Two namespaces, deliberately not one. `owner` is the tenant and decides who
#: may read the row; `participant` is the study pseudonym and says which subject
#: produced it. There is no foreign key from the pseudonym to an authenticated
#: user, and any mapping the study needs lives in its own narrowly-held table —
#: so that association can be destroyed later without re-keying or losing the
#: experimental evidence.
#:
#: `participant` is therefore *not* part of the key and is nullable. An event is
#: a fact about something that happened, not about who caused it, and making the
#: pseudonym part of storage identity would let deduplication, foreign keys,
#: exports and restores all start depending on the very link that has to stay
#: severable.
SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_events (
    owner         TEXT NOT NULL,
    event_id      TEXT NOT NULL,
    at            TEXT NOT NULL,
    kind          TEXT NOT NULL,
    plan_id       TEXT,
    participant   TEXT,
    detail        TEXT NOT NULL,
    PRIMARY KEY (owner, event_id)
)
"""

#: `participant` arrived after the first version of this table. A deployment
#: that had already created the old one would keep it — `CREATE TABLE IF NOT
#: EXISTS` says nothing about columns — and every insert would then fail into
#: `record`'s own exception guard. The pilot would report zero events and look
#: exactly like a pilot nobody used.
ADDED_COLUMNS = (("participant", "TEXT"),)

PLAN_COMPILED = "plan_compiled"
PLAN_SAVED = "plan_saved"
PLAN_REOPENED = "plan_reopened"
PLAN_RESULT_SHOWN = "plan_result_shown"
PLAN_REFUSED = "plan_refused"

#: Someone typed again after seeing a reading. The single most informative
#: thing the pilot can count without asking anyone anything: a person who
#: revises has been told something they did not accept, and the count of
#: revisions before a save is a measure of how far the first reading was off.
PLAN_RESUBMITTED = "plan_resubmitted"

#: A pilot participant went back to the legacy workspace. Recorded because it
#: is the strongest negative signal available and nothing else would show it —
#: `plan_compiled` staying flat looks identical to nobody visiting at all.
LEFT_FOR_LEGACY = "left_for_legacy"

#: Discovery asked for something before it would seal.
#:
#: The plan-level events above record what happened *to a plan*, which turned
#: out to be the wrong half of the story. The harvested corpus found that 16 of
#: 29 attested strategy statements never reach a plan at all — they stop at a
#: question, because real financial intent is routinely incomplete on first
#: utterance. A pilot instrumented only on plans would record those sessions as
#: near-silence and could not say whether the runtime asked well or badly.
#:
#: `detail` carries the dimension names and nothing else. A dimension name is
#: schema vocabulary — `assets`, `amount` — and never the participant's words,
#: so this stays on the countable side of the split that `pilot_session`
#: describes. What they typed lives in `pilot_transcripts`, under a
#: declaration.
DISCOVERY_ASKED = "discovery_asked"

#: The participant answered, and which dimensions closed as a result.
#:
#: Both halves are needed to say anything about burden. Questions asked alone
#: cannot distinguish a runtime that asked once and got what it needed from one
#: that asked three times and was ignored.
DISCOVERY_ANSWERED = "discovery_answered"

#: Discovery closed the meaning. The boundary event: everything before it is
#: interaction, everything after is execution.
INTENT_SEALED = "intent_sealed"

KINDS = (PLAN_COMPILED, PLAN_SAVED, PLAN_REOPENED, PLAN_RESULT_SHOWN,
         PLAN_REFUSED, PLAN_RESUBMITTED, LEFT_FOR_LEGACY,
         DISCOVERY_ASKED, DISCOVERY_ANSWERED, INTENT_SEALED)

#: Coarse, stable reasons an execution produced no figure. Codes rather than
#: the engine's message: a message is rendered copy, it can carry the user's
#: own words, and it changes whenever the wording does.
NO_MARKET_DATA = "NO_MARKET_DATA"
COVERAGE_REFUSED = "COVERAGE_REFUSED"
ENGINE_REFUSED = "ENGINE_REFUSED"
CAPABILITY_REFUSED = "CAPABILITY_REFUSED"


def _execution_refusal_code(run) -> str:
    """Which kind of refusal this was, without reading the message."""
    coverage = run.get("coverage")
    if coverage is not None and not getattr(coverage, "publishable", True):
        return COVERAGE_REFUSED
    if "market data is not available" in str(run.get("unavailable") or ""):
        return NO_MARKET_DATA
    return ENGINE_REFUSED


def _connect():
    from ..deploy.context import current
    from ..db.engine import Database

    from .study_repair import ensure_owner

    connection = Database(current().database.url).connect()
    connection.execute(SCHEMA)
    _widen(connection)
    ensure_owner(connection, "pilot_events")
    return connection


def _widen(connection) -> None:
    """Add columns a previously-created table is missing.

    Not a migration in `db.migrate`'s sense: this table is study
    instrumentation, it holds nothing financial, and losing it costs a count
    rather than a plan. What it must not do is fail silently, which is what a
    stale schema plus `record`'s exception guard would produce together.
    """
    try:
        present = {row[1] for row in
                   connection.execute("PRAGMA table_info(pilot_events)")}
    except Exception:                                          # noqa: BLE001
        return
    if not present:
        return
    for name, kind in ADDED_COLUMNS:
        if name not in present:
            try:
                connection.execute(
                    f"ALTER TABLE pilot_events ADD COLUMN {name} {kind}")
                connection.commit()
            except Exception:                                  # noqa: BLE001
                return


def _profile() -> dict:
    """The deployment, on every event.

    Recorded per event rather than assumed constant for a run: a pilot that
    changed profile halfway would otherwise have one population wearing two
    labels, and nothing in the data would say when it happened.
    """
    from ..deploy.context import current

    resolved = current()
    model = resolved.model
    # The serving revision, so an observation can be joined to the code that
    # produced it rather than inferred from a timestamp. Without it a cohort
    # spanning a deploy is one population wearing two behaviours, and the data
    # cannot say which reading came from which build.
    #
    # `""` when the deployment cannot say — the manifest reports
    # `observable: False` in the same case, so the two agree rather than one
    # of them inventing a value.
    commit = (resolved.build.private() or {}).get("commit") or ""
    return {"parser_mode": model.mode.value,
            "pilot_reader": model.pilot_reader.value,
            "serving_commit": commit}


def record(kind: str, *, plan_id: str = "", participant: str = "",
           **detail: Any) -> None:
    """One event. Never raises into a request.

    Telemetry is the expendable half: a pilot user losing their plan because an
    analytics table was locked would be a worse outcome than losing the count.
    The same rule `deploy.context` already applies to trace retention.

    `participant` is the opaque cookie token and carries nothing about anyone.
    It is here because a ten-person cohort makes every raw count ambiguous
    without it: forty compiles is a busy pilot or one person struggling, and
    those are opposite conclusions drawn from the same number.
    """
    if kind not in KINDS:
        raise ValueError(f"{kind!r} is not one of {KINDS}")

    from datetime import datetime, timezone
    from hashlib import sha256

    at = datetime.now(timezone.utc).isoformat()
    payload = {**detail, **_profile()}
    event_id = sha256(
        f"{at}{kind}{plan_id}{participant}{json.dumps(payload, sort_keys=True)}"
        .encode()).hexdigest()[:16]
    try:
        connection = _connect()
        try:
            from .owner import current as owner_of

            connection.execute(
                "INSERT INTO pilot_events "
                "(owner, event_id, at, kind, plan_id, participant, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (owner_of(), event_id, at, kind, plan_id, participant,
                 json.dumps(payload)))
            connection.commit()
        finally:
            connection.close()
    except Exception:                                          # noqa: BLE001
        return


def observe(reading, *, plan_id: str = "", participant: str = "",
            reopened: bool = False,
            run: Optional[Mapping[str, Any]] = None) -> None:
    """Every event one page view implies, from what actually happened.

    Derived from the reading and the run rather than from the route's belief
    about them: a handler that recorded `SAVED` before the write could report a
    save that failed.
    """
    run = run or {}

    if reopened:
        record(PLAN_REOPENED, plan_id=plan_id, participant=participant)
    else:
        record(PLAN_COMPILED, plan_id=plan_id, participant=participant,
               executable=bool(reading.executable),
               open_question_count=len(reading.questions))

    observe_discovery(reading, plan_id=plan_id, participant=participant)

    for refusal in reading.refusals:
        record(PLAN_REFUSED, plan_id=plan_id, participant=participant,
               refusal_code=CAPABILITY_REFUSED,
               capability=getattr(refusal, "dimension", ""),
               named_reason=getattr(refusal, "kind", ""))

    # Exactly one of these per execution. A run that emitted both would make
    # "results" and "refusals" sum to more than the executions that happened,
    # and every ratio drawn from them would be wrong in a way nothing shows.
    if run:
        if run.get("result") is not None:
            record(PLAN_RESULT_SHOWN, plan_id=plan_id, participant=participant)
        else:
            record(PLAN_REFUSED, plan_id=plan_id, participant=participant,
                   refusal_code=_execution_refusal_code(run))


#: A dimension Mission could not proceed without and Discovery never raised.
#:
#: The third of the three burden questions, and the only one that cannot be
#: answered by counting questions. A runtime that asks nothing looks perfect on
#: "how many follow-ups?" and is failing worse than one that asks twice — the
#: person is simply refused at the end with no chance to supply what was
#: missing.
NOT_ASKED_ABOUT = "not_asked_about"


def _dimensions_seen(participant: str, kind: str) -> set:
    """Dimensions this participant has already had recorded under `kind`.

    Read back out of the events table rather than held in a session. The
    events *are* the record, so a transition cannot be emitted twice however
    many routes call the emitter, and no new state has to be kept correct.
    """
    if not participant:
        return set()
    try:
        connection = _connect()
    except Exception:                                          # noqa: BLE001
        return set()
    try:
        from .owner import current as owner_of

        rows = connection.execute(
            "SELECT detail FROM pilot_events "
            "WHERE owner = ? AND participant = ? AND kind = ?",
            (owner_of(), participant, kind)).fetchall()
    except Exception:                                          # noqa: BLE001
        return set()
    finally:
        connection.close()

    seen: set = set()
    for (detail,) in rows:
        payload = json.loads(detail or "{}")
        if payload.get("dimension"):
            seen.add(payload["dimension"])
        seen.update(payload.get("dimensions", ()))
    return seen


def observe_answers(reading, *, participant: str = "") -> None:
    """One event per dimension that went from unresolved to answered.

    Tied to a state transition, not to a UI action. That is the whole design:

        /pilot/answer resolves X          -> discovery_answered(X)
        /pilot/save   resolves X and Y    -> discovery_answered(X), (Y)
        X already answered                -> nothing

    The previous version emitted from `observe_resubmission`, which made the
    count a property of which route the participant used — `/pilot/save`
    accepts `answer_<dimension>` fields and never called it, so anybody who
    supplied the missing holding and saved in one step was recorded as having
    answered nothing. Emitting from the save route as well would have
    double-counted anyone who answered and then saved. Neither is a defect in
    the emitter; both come from counting form submissions and calling them
    answers.

    Being a transition makes this idempotent, and idempotence is what lets it
    be called from anywhere a reading exists without anybody having to reason
    about overlap.

    A dimension only counts if it was *asked about* first. Something the
    sentence supplied on the first attempt was never a follow-up and must not
    inflate the burden the runtime is being measured on.
    """
    asked = _dimensions_seen(participant, DISCOVERY_ASKED)
    if not asked:
        return
    already = _dimensions_seen(participant, DISCOVERY_ANSWERED)
    settled = {f.field for f in reading.settled if f.value is not None}

    for dimension in sorted((asked & settled) - already):
        record(DISCOVERY_ANSWERED, participant=participant,
               dimension=dimension)


def _execution_identity(reading) -> str:
    """The compiled plan's identity, or "" when nothing compiled.

    Deliberately the *execution* form rather than the canonical one. Two seals
    differing only in how somebody spelled a holding are the same execution and
    must not read as a question having changed the outcome — which is the
    distinction `execution_form()` exists for.
    """
    from hashlib import sha256
    from json import dumps

    scenario = getattr(getattr(reading, "compiled", None), "scenario", None)
    if scenario is None:
        return ""
    try:
        form = dumps(scenario.execution_form(), sort_keys=True, default=str)
    except Exception:                                          # noqa: BLE001
        return ""
    return sha256(form.encode()).hexdigest()[:16]


def observe_discovery(reading, *, plan_id: str = "",
                      participant: str = "") -> None:
    """What Discovery asked, what it sealed, and what it should have asked.

    Structured only. Dimension names are schema vocabulary and never the
    participant's sentence, so this stays on the countable side of the split —
    what somebody typed lives in `pilot_transcripts` under a declaration.

    The `not_asked_about` list is computed rather than declared: a dimension
    Mission refuses as `UNRESOLVED_INPUT` that never appeared in Discovery's
    questions is, by construction, a material fact the interaction failed to
    ask for. It is derived from what the two layers actually did, so it cannot
    drift out of step with either of them the way a hand-maintained list would.
    """
    questions = sorted(reading.questions)
    if questions:
        record(DISCOVERY_ASKED, plan_id=plan_id, participant=participant,
               dimensions=questions, question_count=len(questions))

    unresolved = sorted({
        getattr(r, "dimension", "") for r in reading.refusals
        if getattr(r, "kind", "") == "UNRESOLVED_INPUT"})
    unasked = [d for d in unresolved if d and d not in questions]
    if unasked:
        record(DISCOVERY_ASKED, plan_id=plan_id, participant=participant,
               dimensions=[], question_count=0,
               **{NOT_ASKED_ABOUT: unasked})

    if reading.executable:
        # The execution identity travels with the seal. Without it the ledger
        # can say how many questions were asked and not whether any of them
        # changed what would run — and "which follow-ups actually matter" is
        # the question that decides whether a dimension deserves a
        # deterministic reader or a better default.
        #
        # A digest, so it stays on the countable side of the split: it cannot
        # carry a sentence, and two seals are comparable without either being
        # readable.
        record(INTENT_SEALED, plan_id=plan_id, participant=participant,
               settled_count=len(list(reading.settled)),
               questions_before_sealing=len(questions),
               execution_identity=_execution_identity(reading))

    # After the questions above are on record, so a dimension asked and
    # answered within one request is still seen as a transition rather than
    # falling between the two calls.
    observe_answers(reading, participant=participant)


def answers_already_in_the_prompt(prompt: str,
                                  answers: Mapping[str, str]) -> Sequence[str]:
    """Fields the participant answered by repeating what they had already said.

    The only mechanical evidence available for "the runtime asked an
    unnecessary clarification". If someone wrote *invest $500 monthly into VTI*,
    was asked what to hold, and typed *VTI*, the reading missed something that
    was in the sentence — and that is a fact about the parse, not a judgement
    about the person.

    It is a **proxy and not the finding**. It cannot see the opposite error, a
    question that should have been asked and was not, and it will miss a
    participant who answers a redundant question in different words. Whether a
    clarification was unnecessary is still settled in the interview; this only
    says which transcripts to read first.

    Word-boundary matched and at least two characters, so an answer of `5` does
    not count itself as already present in `$500`.
    """
    import re

    said = prompt.lower()
    repeated = []
    for field, value in answers.items():
        text = str(value).strip().lower()
        if len(text) < 2:
            continue
        if re.search(rf"(?<!\w){re.escape(text)}(?!\w)", said):
            repeated.append(field)
    return tuple(repeated)


def observe_resubmission(*, attempt: int, changed: Optional[bool],
                         participant: str = "",
                         answered: Sequence[str] = (),
                         repeated: Sequence[str] = ()) -> None:
    """A second or later prompt from one participant.

    `changed` is `None` rather than `False` when transcripts are off: a
    deployment that keeps no prose can still count that someone submitted
    again, but genuinely cannot say whether they reworded. Recording `False`
    there would be an assertion nothing checked, and it would read in a summary
    as "they resubmitted the identical sentence" — a different and much more
    interesting claim than "we do not know".
    """
    record(PLAN_RESUBMITTED, participant=participant, attempt=attempt,
           text_changed=changed, answered_field_count=len(answered),
           repeated_from_prompt=list(repeated))

    # No `discovery_answered` here. It used to be emitted from this function,
    # which made it an event about an HTTP route rather than about anything
    # that happened to the intent — and `/pilot/save` accepts answers without
    # passing through here at all, so the count depended on which button
    # somebody pressed. See `observe_answers`.


def observe_departure(path: str, *, participant: str = "") -> None:
    """A participant reached a legacy workspace page.

    The path is the route pattern this deployment serves, not a URL a user
    composed — a plan id or a query string could carry their own words, and
    nothing in this table may.
    """
    record(LEFT_FOR_LEGACY, participant=participant, destination=path)


def observe_save(reading, plan_id: str, participant: str = "") -> None:
    """One event, with the open count as a number.

    The first version emitted a second event carrying the list of open field
    names. A count is what a cohort can be compared on; the names are already
    on the persisted artifact, where they belong to one plan rather than to a
    statistic.
    """
    record(PLAN_SAVED, plan_id=plan_id, participant=participant,
           executable=bool(reading.executable),
           settled_field_count=len(reading.settled),
           open_question_count=len(reading.questions))

    # `/pilot/save` accepts `answer_<dimension>` fields, so a participant can
    # supply the missing holding and save in one step. That used to record no
    # answer at all. It is safe to call here *and* from `observe_discovery`
    # because the event is a state transition: whichever route gets there
    # first emits it, and the other finds it already recorded.
    observe_answers(reading, participant=participant)


def every_event() -> Sequence[Mapping[str, Any]]:
    connection = _connect()
    try:
        from .owner import current as owner_of

        rows = connection.execute(
            "SELECT event_id, at, kind, plan_id, participant, detail "
            "FROM pilot_events WHERE owner = ? ORDER BY at",
            (owner_of(),)).fetchall()
    finally:
        connection.close()
    return [{"event_id": r["event_id"], "at": r["at"], "kind": r["kind"],
             "plan_id": r["plan_id"],
             "participant": r["participant"] or "",
             **json.loads(r["detail"])} for r in rows]


def attempts_by(participant: str) -> int:
    """How many prompts this participant has had read.

    Counted from the events rather than from the transcript store, so a
    deployment that retains no prose can still tell a first attempt from a
    fifth. Tying it to transcripts would have made the single most useful
    usability signal available only to deployments that also kept people's
    words, which is backwards — the count needs no permission and the words do.
    """
    if not participant:
        return 0
    try:
        connection = _connect()
    except Exception:                                          # noqa: BLE001
        return 0
    try:
        from .owner import current as owner_of

        row = connection.execute(
            "SELECT COUNT(*) AS how_many FROM pilot_events "
            "WHERE owner = ? AND participant = ? AND kind = ?",
            (owner_of(), participant, PLAN_COMPILED)).fetchone()
        return int(row["how_many"]) if row else 0
    except Exception:                                          # noqa: BLE001
        return 0
    finally:
        connection.close()


def summary() -> Mapping[str, Any]:
    """What the pilot can honestly report.

    Counts and names. Every number here is a thing that happened, and none of
    them is a claim about what anyone thought of it.
    """
    from collections import Counter

    events = every_event()
    refusals = Counter(e.get("capability", "") for e in events
                       if e["kind"] == PLAN_REFUSED and e.get("capability"))
    return {
        "events": len(events),
        "participants": len({e.get("participant") for e in events
                             if e.get("participant")}),
        "by_kind": dict(Counter(e["kind"] for e in events)),
        "by_profile": dict(Counter(
            f"{e.get('parser_mode')}/{e.get('pilot_reader')}" for e in events)),
        "refused_capabilities": dict(refusals.most_common()),
        "plans_compiled": sum(1 for e in events if e["kind"] == PLAN_COMPILED),
        "plans_saved": sum(1 for e in events if e["kind"] == PLAN_SAVED),
        "plans_saved_with_questions_open": sum(
            1 for e in events if e["kind"] == PLAN_SAVED
            and e.get("open_question_count")),
        "plans_reopened": len({e["plan_id"] for e in events
                               if e["kind"] == PLAN_REOPENED}),
        "resubmissions": sum(1 for e in events
                             if e["kind"] == PLAN_RESUBMITTED),
        "resubmissions_with_reworded_text": sum(
            1 for e in events
            if e["kind"] == PLAN_RESUBMITTED and e.get("text_changed") is True),
        # Which questions people answered with words already in their sentence.
        # A capability appearing here repeatedly is a parser defect with a name
        # attached, found without anybody being asked anything.
        "clarifications_answered_from_the_prompt": dict(Counter(
            field for e in events if e["kind"] == PLAN_RESUBMITTED
            for field in e.get("repeated_from_prompt", ())).most_common()),
        "departures_to_legacy": sum(1 for e in events
                                    if e["kind"] == LEFT_FOR_LEGACY),
        "results": sum(1 for e in events if e["kind"] == PLAN_RESULT_SHOWN),
        "refusals_by_code": dict(Counter(
            e.get("refusal_code", "") for e in events
            if e["kind"] == PLAN_REFUSED)),
        "not_measured_here": [
            "whether a refusal was understood",
            "whether the evidence made the result more trustworthy",
            "where someone gave up and why",
            "what they expected before they started",
            # Named explicitly because a nearby number looks like it answers
            # it. `clarifications_answered_from_the_prompt` says a question was
            # answered with words the sentence already contained, which is
            # evidence of a bad parse, not a finding about the question.
            "whether a clarification was unnecessary — the counter beside this "
            "is a proxy that says which transcripts to read, not a verdict",
            "why anyone went back to the legacy workspace",
        ],
    }
