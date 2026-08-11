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
    asked: dict = {}
    never_asked: dict = {}
    restated: dict = {}
    sealed, sessions_with_questions, questions_total = 0, 0, 0
    participants = set()

    for row in rows:
        detail = row["detail"]
        if row["participant"]:
            participants.add(row["participant"])

        if row["kind"] == DISCOVERY_ASKED:
            for dimension in detail.get("dimensions", ()):
                asked[dimension] = asked.get(dimension, 0) + 1
            count = int(detail.get("question_count") or 0)
            questions_total += count
            if count:
                sessions_with_questions += 1
            for dimension in detail.get(NOT_ASKED_ABOUT, ()):
                never_asked[dimension] = never_asked.get(dimension, 0) + 1

        elif row["kind"] == INTENT_SEALED:
            sealed += 1

        elif row["kind"] == PLAN_RESUBMITTED:
            # The unnecessary-question proxy, already recorded by
            # `observe_resubmission`: dimensions whose answer was sitting in
            # the sentence the person had already typed.
            for dimension in detail.get("repeated_from_prompt", ()):
                restated[dimension] = restated.get(dimension, 0) + 1

    def ranked(tally):
        return dict(sorted(tally.items(), key=lambda kv: -kv[1]))

    return {
        "schema": "quantify-follow-up-burden@1",
        "participants": len(participants),
        "sealed_intents": sealed,
        "sessions_that_were_asked_something": sessions_with_questions,
        "questions_asked": questions_total,
        "asked_by_dimension": ranked(asked),
        "answer_was_already_in_the_prompt": ranked(restated),
        "never_asked_by_dimension": ranked(never_asked),
        "reading_note": (
            "Three separate questions. `asked_by_dimension` is the burden; "
            "`answer_was_already_in_the_prompt` is a proxy for burden that was "
            "avoidable, and it is a proxy — a person may restate something the "
            "runtime was right to be unsure about; `never_asked_by_dimension` "
            "is the opposite failure, where somebody was refused for a fact "
            "nobody asked them for."),
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
        return {"schema": "quantify-follow-up-burden@1",
                "unavailable": f"{type(failure).__name__}: {failure}"}


if __name__ == "__main__":
    print(json.dumps(report(), indent=2))
