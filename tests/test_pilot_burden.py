"""Follow-up burden, and whether the instrumentation can see the third case.

The interesting property is not that questions are counted. It is that a
runtime which asks *nothing* and refuses at the end must not score better than
one which asks twice and seals — and a metric built only on question counts
scores it perfectly.
"""
from __future__ import annotations

from src.workspace.pilot_burden import summarise
from src.workspace.pilot_events import (
    DISCOVERY_ASKED,
    INTENT_SEALED,
    NOT_ASKED_ABOUT,
    PLAN_RESUBMITTED,
)


def asked(dimensions, participant="p1", **extra):
    return {"kind": DISCOVERY_ASKED, "participant": participant,
            "detail": {"dimensions": list(dimensions),
                       "question_count": len(dimensions), **extra}}


class TestTheThreeQuestionsAreSeparate:
    def test_one_person_asked_twice_about_one_thing_counts_once(self):
        """The unit is a dimension per participant, not a form submission.
        Someone who reloads the page five times is one person who has not
        answered, not five people who could not."""
        out = summarise([asked(["assets"]), asked(["assets", "amount"])])
        assert out["asked_by_dimension"] == {"assets": 1, "amount": 1}
        assert out["questions_asked"] == 2

    def test_two_people_asked_about_one_thing_count_twice(self):
        """The discriminating half: the collapse is per participant, not
        global, or the tally would read as one no matter the cohort size."""
        out = summarise([asked(["assets"], participant="p1"),
                         asked(["assets"], participant="p2")])
        assert out["asked_by_dimension"] == {"assets": 2}
        assert out["participants"] == 2

    def test_a_question_whose_answer_was_already_typed_is_separated(self):
        """A proxy, and named as one. It is burden that need not have been
        spent, which is a different claim from burden."""
        out = summarise([
            asked(["assets"]),
            {"kind": PLAN_RESUBMITTED, "participant": "p1",
             "detail": {"repeated_from_prompt": ["assets"]}}])
        assert out["answer_was_already_in_the_prompt"] == {"assets": 1}
        assert out["asked_by_dimension"] == {"assets": 1}

    def test_a_fact_nobody_was_asked_for_is_counted_apart(self):
        """The case a question-count cannot reach."""
        out = summarise([asked([], **{NOT_ASKED_ABOUT: ["amount"]})])
        assert out["never_asked_by_dimension"] == {"amount": 1}
        assert out["questions_asked"] == 0


class TestSilenceDoesNotScoreAsSuccess:
    def test_asking_nothing_and_refusing_is_visible(self):
        """The failure mode this file exists for. Both runtimes below ask zero
        questions; only one of them left somebody unable to proceed."""
        polite = summarise([{"kind": INTENT_SEALED, "participant": "p1",
                             "detail": {"settled_count": 4,
                                        "questions_before_sealing": 0}}])
        silent = summarise([asked([], **{NOT_ASKED_ABOUT: ["assets"]})])

        assert polite["questions_asked"] == silent["questions_asked"] == 0
        assert not polite["never_asked_by_dimension"]
        assert silent["never_asked_by_dimension"] == {"assets": 1}
        assert polite["sealed_intents"] == 1 and silent["sealed_intents"] == 0


class TestItRefusesToProduceARate:
    def test_no_percentage_is_published(self):
        """A ten-person cohort makes a percentage look like a measurement and
        behave like one participant's afternoon."""
        out = summarise([asked(["assets"])])
        for key, value in out.items():
            if isinstance(value, str):
                continue
            assert "rate" not in key and "percent" not in key, key
        assert "No rates" in out["denominator_note"]

    def test_an_unreachable_table_is_named_not_reported_as_empty(self):
        """Zero events and a broken table look identical, and one of them means
        the pilot is running fine while its instrumentation is not."""
        import src.workspace.pilot_burden as burden

        original = burden._rows
        burden._rows = lambda: (_ for _ in ()).throw(RuntimeError("no db"))
        try:
            out = burden.report()
        finally:
            burden._rows = original
        assert "unavailable" in out and "RuntimeError" in out["unavailable"]


class TestTheContextStaysCountable:
    def test_no_event_detail_carries_prose(self):
        """The split `pilot_session` describes: dimension names are schema
        vocabulary, never the participant's sentence."""
        from src.discovery.schema import QUANTIFY_SCHEMA

        names = {d.name for d in QUANTIFY_SCHEMA.dimensions}
        out = summarise([asked(["assets", "amount"])])
        for dimension in out["asked_by_dimension"]:
            assert dimension in names, (
                f"{dimension!r} is not a schema dimension; the burden report "
                "is carrying something that came from a person's words")


class TestAnswersAreStateTransitionsNotFormSubmissions:
    """The semantics settled before the cohort, because a metric whose meaning
    changes once real data exists cannot be compared to anything."""

    def test_only_a_dimension_that_was_asked_about_can_be_answered(self):
        """Something the first sentence already supplied was never a follow-up.
        Counting it would inflate the burden the runtime is measured on with
        work it never asked anyone to do."""
        from src.workspace.pilot_events import observe_answers

        recorded = []
        _run(observe_answers, recorded, settled=["assets", "amount"],
             asked=set(), already=set())
        assert recorded == []

    def test_an_asked_and_settled_dimension_transitions_once(self):
        from src.workspace.pilot_events import observe_answers

        recorded = []
        _run(observe_answers, recorded, settled=["assets", "amount"],
             asked={"assets"}, already=set())
        assert [d["dimension"] for d in recorded] == ["assets"]

    def test_a_dimension_already_answered_does_not_transition_again(self):
        """Idempotence. This is what makes the emitter safe to call from every
        route rather than exactly one."""
        from src.workspace.pilot_events import observe_answers

        recorded = []
        _run(observe_answers, recorded, settled=["assets"],
             asked={"assets"}, already={"assets"})
        assert recorded == []

    def test_a_dimension_asked_and_still_open_does_not_transition(self):
        from src.workspace.pilot_events import observe_answers

        recorded = []
        _run(observe_answers, recorded, settled=["amount"],
             asked={"assets"}, already=set())
        assert recorded == []


def _run(observe_answers, recorded, *, settled, asked, already):
    """Drives `observe_answers` against stated prior state.

    The prior state is injected rather than built by replaying HTTP, because
    the property under test is the transition rule itself. `test_pilot_events`
    covers the same rule through the real routes, which is where a wiring
    mistake would show.
    """
    from dataclasses import dataclass
    from typing import Any

    import src.workspace.pilot_events as events

    @dataclass
    class Field:
        field: str
        value: Any = "x"

    class Reading:
        def __init__(self, names):
            self.settled = [Field(n) for n in names]

    original_seen, original_record = events._dimensions_seen, events.record
    events._dimensions_seen = lambda who, kind: (
        asked if kind == events.DISCOVERY_ASKED else already)
    events.record = lambda kind, **detail: recorded.append(detail)
    try:
        observe_answers(Reading(settled), participant="p1")
    finally:
        events._dimensions_seen, events.record = original_seen, original_record


class TestTheCohortQuestionsAreAnswerable:
    """Audited before the cohort rather than after. Instrumentation added
    afterwards cannot describe what already happened, so a question nobody can
    answer is a question whose data was never collected."""

    def test_whether_answering_changed_what_would_run(self):
        """The question that decides whether a dimension deserves a
        deterministic reader or merely a better default. Two seals by one
        participant with different execution identities means the answering
        moved the outcome."""
        rows = [
            {"kind": INTENT_SEALED, "participant": "p1",
             "detail": {"execution_identity": "aaaa"}},
            {"kind": INTENT_SEALED, "participant": "p1",
             "detail": {"execution_identity": "bbbb"}},
            {"kind": INTENT_SEALED, "participant": "p2",
             "detail": {"execution_identity": "cccc"}},
            {"kind": INTENT_SEALED, "participant": "p2",
             "detail": {"execution_identity": "cccc"}},
        ]
        out = summarise(rows)
        assert out["answering_changed_the_outcome"] == 1, (
            "p1 sealed two different executions and p2 sealed the same one "
            "twice; only p1's follow-ups moved anything")

    def test_abandonment_is_derived_from_an_absence(self):
        """No event can be emitted by the thing that did not happen, so a
        participant asked something who never sealed is found by subtraction."""
        out = summarise([
            asked(["assets"], participant="stayed"),
            {"kind": INTENT_SEALED, "participant": "stayed",
             "detail": {"execution_identity": "aaaa"}},
            asked(["assets"], participant="left"),
        ])
        assert out["asked_then_abandoned"] == 1

    def test_the_identity_is_an_execution_identity_not_a_plan_dump(self):
        """Two seals differing only in how a holding was spelled are the same
        execution, and must not read as a question having changed anything."""
        from src.workspace.pilot_events import _execution_identity

        class Reading:
            def __init__(self, assets):
                from src.mission.scenario import AllocationRule
                self.compiled = type("C", (), {"scenario": type("S", (), {
                    "execution_form": lambda self, a=assets: {
                        "assets": AllocationRule(assets=a).execution_form()}
                })()})()

        assert _execution_identity(Reading(("the index fund",))) == \
            _execution_identity(Reading(("index fund",)))

    def test_a_reading_that_compiled_nothing_carries_no_identity(self):
        from src.workspace.pilot_events import _execution_identity

        assert _execution_identity(type("R", (), {"compiled": None})()) == ""
