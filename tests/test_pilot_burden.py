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
    def test_questions_asked_are_counted_by_dimension(self):
        out = summarise([asked(["assets"]), asked(["assets", "amount"])])
        assert out["asked_by_dimension"] == {"assets": 2, "amount": 1}
        assert out["questions_asked"] == 3

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
