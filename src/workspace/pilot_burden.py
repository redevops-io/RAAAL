"""Follow-up burden: was the runtime's asking worth what it cost?

    python -m src.workspace.pilot_burden

Three questions, and they are not the same question:

    how many follow-ups were needed
    which were unnecessary
    which missing material facts were never asked about

The third is the one that cannot be reached by counting. A runtime that asks
nothing scores perfectly on the first and may be failing worse than one that
asks twice, because the person is refused at the end having never been given
the chance to supply what was missing. It is computed from what the two layers
did — a dimension Mission refuses as `UNRESOLVED_INPUT` that Discovery never
raised — rather than from anyone's list.

**Why this became the interesting metric.** The harvested corpus found that 16
of 29 attested strategy statements stop at a question about holdings. Real
financial intent is incomplete on first utterance, and Discovery representing
that absence is the architecture working rather than a recognition gap. What is
still open is whether it asks the *minimum useful* follow-up, and that is a
question only the cohort can answer.

**What this cannot answer, said here rather than discovered later.** Whether
two different phrasings of one strategy cost different follow-up burden needs
somebody to decide which utterances *are* one strategy. No event can carry that
judgement, so it is a transcript-and-annotation task under the retention
declaration, not a telemetry gap to close.

**Proxies are labelled as proxies.** "Unnecessary" here means the participant's
answer restated something their original sentence already contained, which is
evidence and not proof — a person may restate something the runtime was right
to be unsure about. The field is named for what it measures.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from .pilot_events import (
    DISCOVERY_ASKED,
    DISCOVERY_ANSWERED,
    INTENT_SEALED,
    NOT_ASKED_ABOUT,
    PLAN_RESUBMITTED,
)


def _rows() -> list:
    from .pilot_events import _connect

    connection = _connect()
    try:
        found = connection.execute(
            "SELECT kind, participant, detail FROM pilot_events "
            "WHERE kind IN (?, ?, ?, ?)",
            (DISCOVERY_ASKED, DISCOVERY_ANSWERED, INTENT_SEALED,
             PLAN_RESUBMITTED)).fetchall()
    finally:
        connection.close()
    return [{"kind": k, "participant": p, "detail": json.loads(d or "{}")}
            for k, p, d in found]


def summarise(rows) -> Mapping[str, Any]:
    """The three questions, plus the denominators they are over.

    Counts and verbatim lists are kept apart. `asked_by_dimension` is a tally
    and `never_asked_by_dimension` is a tally; neither is a rate, because a
    ten-person cohort produces denominators too small for a percentage to mean
    what a percentage looks like it means.
    """
    # Per participant, then tallied. The unit is a *dimension*, counted once
    # per person however many times they were shown the question — otherwise a
    # participant who reloaded the page five times would look like five people
    # who could not answer.
    asked_per: dict = {}
    answered_per: dict = {}
    sealed_by: set = set()
    identities_by: dict = {}
    never_asked: dict = {}
    restated: dict = {}
    sealed = 0
    participants = set()

    for row in rows:
        detail, who = row["detail"], row["participant"]
        if who:
            participants.add(who)

        if row["kind"] == DISCOVERY_ASKED:
            asked_per.setdefault(who, set()).update(
                detail.get("dimensions", ()))
            for dimension in detail.get(NOT_ASKED_ABOUT, ()):
                never_asked[dimension] = never_asked.get(dimension, 0) + 1

        elif row["kind"] == DISCOVERY_ANSWERED:
            # One event per unresolved -> answered transition, emitted by
            # whichever route performed it. `dimension` is singular for that
            # reason; the plural key is read too so recordings made before the
            # semantics were settled are not silently skipped.
            if detail.get("dimension"):
                answered_per.setdefault(who, set()).add(detail["dimension"])
            answered_per.setdefault(who, set()).update(
                detail.get("dimensions", ()))

        elif row["kind"] == INTENT_SEALED:
            sealed += 1
            if who:
                sealed_by.add(who)
            identity = detail.get("execution_identity")
            if who and identity:
                identities_by.setdefault(who, set()).add(identity)

        elif row["kind"] == PLAN_RESUBMITTED:
            for dimension in detail.get("repeated_from_prompt", ()):
                restated[dimension] = restated.get(dimension, 0) + 1

    def ranked(tally):
        return dict(sorted(tally.items(), key=lambda kv: -kv[1]))

    def by_dimension(per):
        tally: dict = {}
        for dimensions in per.values():
            for dimension in dimensions:
                tally[dimension] = tally.get(dimension, 0) + 1
        return tally

    asked = by_dimension(asked_per)
    answered = by_dimension(answered_per)

    # Abandonment is an absence, so it is derived rather than recorded. A
    # participant who was asked something and never sealed anything left
    # mid-conversation — and an event for "did not come back" cannot be
    # emitted by the thing that did not happen.
    abandoned = sorted(set(asked_per) - sealed_by - {""})

    # Whether the follow-ups a person answered changed what would run. Two
    # seals by one participant with different execution identities means the
    # answering moved the outcome; the same identity twice means it did not.
    # This is the question that decides whether a dimension deserves a
    # deterministic reader or simply a better default.
    moved_the_outcome = sorted(w for w, ids in identities_by.items()
                               if len(ids) > 1)

    # Asked and never settled. Not a failure on its own — somebody may have
    # abandoned the session, which is itself worth seeing — but it is the
    # difference between a question that did its job and one that stopped
    # somebody.
    unanswered = {d: asked[d] - answered.get(d, 0) for d in asked
                  if asked[d] > answered.get(d, 0)}

    return {
        "schema": "quantify-follow-up-burden@3",
        "participants": len(participants),
        "sealed_intents": sealed,
        "people_who_were_asked_something": len(asked_per),
        "questions_asked": sum(len(d) for d in asked_per.values()),
        "answers_supplied": sum(len(d) for d in answered_per.values()),
        "asked_by_dimension": ranked(asked),
        "answered_by_dimension": ranked(answered),
        "asked_and_never_settled": ranked(unanswered),
        "answer_was_already_in_the_prompt": ranked(restated),
        "never_asked_by_dimension": ranked(never_asked),
        "asked_then_abandoned": len(abandoned),
        "answering_changed_the_outcome": len(moved_the_outcome),
        "reading_note": (
            "The unit is a dimension per participant, not a form submission. "
            "`asked_by_dimension` is the burden; `answered_by_dimension` is "
            "what the asking bought; `answer_was_already_in_the_prompt` is a "
            "proxy for burden that was avoidable, and it is a proxy — a person "
            "may restate something the runtime was right to be unsure about; "
            "`never_asked_by_dimension` is the opposite failure, where "
            "somebody was refused for a fact nobody asked them for."),
        "denominator_note": (
            "No rates. A ten-person cohort makes a percentage look like a "
            "measurement and behave like one participant's afternoon."),
    }


def report() -> Mapping[str, Any]:
    try:
        return summarise(_rows())
    except Exception as failure:                                # noqa: BLE001
        # Named rather than swallowed. An empty report and an unreachable
        # table look identical, and one of them means the pilot is running
        # fine while its instrumentation is not.
        return {"schema": "quantify-follow-up-burden@3",
                "unavailable": f"{type(failure).__name__}: {failure}"}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
