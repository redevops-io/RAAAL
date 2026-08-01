"""Stage 1 with a model, and the quarantine around it.

The compiler's whole claim is that stages 2–10 are deterministic and that a
description compiles the same way a year from now. Putting a model in stage 1
either breaks that claim or is made safe by verification — there is no third
option, so these tests are written adversarially: most of them are a model
behaving badly.

The failure that matters is not a wrong answer. It is a *plausible* wrong
answer: a fabricated quotation, an invented ticker, a value outside the
vocabulary that sounds like one inside it. Those are what a fluent model
produces when it does not know, and each has a test below.
"""
from __future__ import annotations

import json

import pytest

from src.mission.compiler import Origin, compile_scenario, parse
from src.mission.parse_model import (
    VOCABULARY,
    build_system_prompt,
    merge,
    parse_with_model,
    verify_proposals,
)

DESCRIPTION = ("I put $500 into VTI every month, and whenever it trades below "
               "its 200 day moving average I buy extra with additional cash. "
               "I never sell.")


class FakeClient:
    """Returns whatever the test tells it to, and records what it was asked."""

    def __init__(self, payload, *, raises=None):
        self.payload = payload
        self.raises = raises
        self.calls = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.raises is not None:
            raise self.raises
        if isinstance(self.payload, str):
            return self.payload
        return json.dumps(self.payload)


class TestTheModelWidensRecognition:
    """The reason to add a model at all."""

    def test_a_phrasing_the_regexes_miss_is_recognised(self):
        text = ("Each month I add money to VOO. I want to top it up whenever it "
                "dips under its average, using cash I set aside separately.")

        without = parse(text)
        assert without.value_of("trigger_semantics") is None

        client = FakeClient({"recognitions": [
            {"field": "trigger_semantics", "value": "persistent_condition",
             "span": "whenever it dips under its average"},
            {"field": "funding_source", "value": "additional_cash",
             "span": "cash I set aside separately"},
        ], "assets": ["VOO"], "unclear": []})

        result = parse_with_model(text, client=client)
        assert result.parsed.value_of("trigger_semantics").value == \
            "persistent_condition"
        assert result.provenance.model_available
        assert set(result.provenance.accepted_from_model) == {
            "trigger_semantics", "funding_source"}

    def test_the_prompt_is_built_from_the_vocabulary(self):
        """A hand-written prompt goes stale; a derived one cannot."""
        prompt = build_system_prompt()
        for field_name, values in VOCABULARY.items():
            for value in values:
                assert f"{field_name} = {value}" in prompt

    def test_the_span_requirement_is_stated_in_the_prompt(self):
        assert "verbatim" in build_system_prompt()


class TestTheQuarantine:
    """Everything the model returns is a claim about the text, checked."""

    def test_a_fabricated_quotation_is_rejected(self):
        """The check that catches invention.

        A model that has decided on a reading will happily supply a quotation
        supporting it. If the words are not in the description, the user did not
        write them, and the reading is the model's rather than theirs.
        """
        payload = {"recognitions": [
            {"field": "trigger_semantics", "value": "crossing_event",
             "span": "only on the day it first crosses below"},
        ]}
        recognitions, _assets, _unclear, rejections = verify_proposals(
            payload, DESCRIPTION)

        assert recognitions == ()
        assert len(rejections) == 1
        assert "does not appear in the description" in rejections[0].why

    def test_a_value_outside_the_vocabulary_is_rejected(self):
        payload = {"recognitions": [
            {"field": "trigger_semantics", "value": "momentum_breakout",
             "span": "trades below"},
        ]}
        _r, _a, _u, rejections = verify_proposals(payload, DESCRIPTION)
        assert len(rejections) == 1
        assert "is not one of" in rejections[0].why

    def test_an_invented_field_is_rejected(self):
        payload = {"recognitions": [
            {"field": "stop_loss_percent", "value": "10", "span": "I never sell"},
        ]}
        _r, _a, _u, rejections = verify_proposals(payload, DESCRIPTION)
        assert rejections[0].why == "not a field stage 1 recognises"

    def test_a_ticker_not_in_the_text_is_rejected(self):
        """Resolving a company name to a symbol prices the wrong security."""
        payload = {"recognitions": [], "assets": ["VTI", "GOOGL"]}
        _r, assets, _u, rejections = verify_proposals(payload, DESCRIPTION)

        assert assets == ("VTI",)
        assert any("prices the wrong security" in r.why for r in rejections)

    def test_an_amount_not_in_the_text_is_rejected(self):
        """The one proposal that rescales every figure downstream."""
        payload = {"recognitions": [
            {"field": "amount", "value": "5000", "span": "$500"},
        ]}
        _r, _a, _u, rejections = verify_proposals(payload, DESCRIPTION)
        assert [r.why for r in rejections] == [
            "no such figure appears in the description"]

    def test_an_amount_that_agrees_with_the_text_is_not_reported(self):
        """The deterministic extractor already read it.

        Telling a user "we could not use amount 500" on a plan whose amount was
        read correctly would undermine the one screen whose job is trust.
        """
        payload = {"recognitions": [
            {"field": "amount", "value": "500", "span": "$500"},
        ]}
        _r, _a, _u, rejections = verify_proposals(payload, DESCRIPTION)
        assert rejections == ()

    def test_a_repeated_field_keeps_the_first_and_records_the_rest(self):
        payload = {"recognitions": [
            {"field": "dividends", "value": "reinvested", "span": "VTI"},
            {"field": "dividends", "value": "held_as_cash", "span": "VTI"},
        ]}
        recognitions, _a, _u, rejections = verify_proposals(payload, DESCRIPTION)
        assert [r.value for r in recognitions] == ["reinvested"]
        assert rejections[0].why == "the field was already recognised"

    def test_rejections_are_kept_not_dropped(self):
        """"I did not understand this" is information; silence is not."""
        client = FakeClient({"recognitions": [
            {"field": "trigger_semantics", "value": "crossing_event",
             "span": "words nobody wrote"},
        ]})
        result = parse_with_model(DESCRIPTION, client=client)
        assert len(result.provenance.rejected) == 1
        assert result.provenance.rejected[0].to_json()["field"] == \
            "trigger_semantics"

    def test_unclear_phrases_become_unresolved_not_defaults(self):
        """The rule the whole compiler rests on, applied to model output."""
        text = "I want to ladder into Berkshire over the year."
        client = FakeClient({"recognitions": [], "assets": [],
                             "unclear": ["ladder into"]})
        result = parse_with_model(text, client=client)

        assert "ladder into" in result.parsed.unclear
        compiled = compile_scenario(text, name="s", parsed=result.parsed)
        assert any(u.field == "unclear:ladder into" for u in compiled.unresolved)

    def test_unclear_prose_is_kept_apart_from_an_ambiguous_name(self):
        """Two different questions. One can offer options; the other cannot.

        Rendering them the same way asks the user to tell the two apart, and
        conflating them once crashed the compiler on the main product path: a
        free-text phrase was looked up in the ambiguous-ticker table.
        """
        text = "I want to ladder into Berkshire over the year."
        client = FakeClient({"recognitions": [], "assets": [],
                             "unclear": ["ladder into"]})
        result = parse_with_model(text, client=client)

        assert result.parsed.unrecognized == ("berkshire",)
        assert result.parsed.unclear == ("ladder into",)

        compiled = compile_scenario(text, name="s", parsed=result.parsed)
        questions = {u.field: u.question for u in compiled.unresolved}
        assert "BRK.A or BRK.B?" in questions["asset_identity:berkshire"]
        assert questions["unclear:ladder into"] == \
            "What did you mean by 'ladder into'?"


class TestDeterminismIsPreserved:

    def test_the_deterministic_reading_wins_a_contested_field(self):
        """A regex that fired matched a specific, distinguishing phrase."""
        client = FakeClient({"recognitions": [
            {"field": "trigger_semantics", "value": "crossing_event",
             "span": "trades below"},
        ]})
        result = parse_with_model(DESCRIPTION, client=client)

        assert result.parsed.value_of("trigger_semantics").value == \
            "persistent_condition"
        assert [d.to_json() for d in result.provenance.disagreements] == [
            {"field": "trigger_semantics",
             "deterministic": "persistent_condition",
             "model": "crossing_event"}]

    def test_a_disagreement_is_surfaced_rather_than_settled_silently(self):
        client = FakeClient({"recognitions": [
            {"field": "trigger_semantics", "value": "crossing_event",
             "span": "trades below"},
        ]})
        result = parse_with_model(DESCRIPTION, client=client)
        assert result.provenance.disagreements, (
            "two readers differing on one sentence is exactly what a "
            "confirmation screen must show")

    def test_stages_two_onward_do_not_know_which_parser_ran(self):
        """The property that makes a model in stage 1 acceptable at all."""
        client = FakeClient({"recognitions": [], "assets": [], "unclear": []})
        from_model = parse_with_model(DESCRIPTION, client=client).parsed

        a = compile_scenario(DESCRIPTION, name="s", parsed=from_model)
        b = compile_scenario(DESCRIPTION, name="s")
        assert a.to_json() == b.to_json()

    def test_the_same_parse_compiles_the_same_way_twice(self):
        client = FakeClient({"recognitions": [
            {"field": "dividends", "value": "reinvested",
             "span": "I never sell"},
        ], "assets": ["VTI"], "unclear": []})
        parsed = parse_with_model(DESCRIPTION, client=client).parsed

        first = compile_scenario(DESCRIPTION, name="s", parsed=parsed)
        second = compile_scenario(DESCRIPTION, name="s", parsed=parsed)
        assert first.to_json() == second.to_json()

    def test_a_parse_of_different_text_is_refused(self):
        """Otherwise stages 2-10 describe a scenario nobody wrote."""
        parsed = parse("something else entirely")
        with pytest.raises(ValueError, match="different text"):
            compile_scenario(DESCRIPTION, name="s", parsed=parsed)

    def test_the_template_hint_stays_deterministic(self):
        """A life-event template states cited rules; a model choosing one would
        put a paraphrase of them in front of someone who cannot tell."""
        text = "My RSUs vest quarterly and I sell the vested shares."
        client = FakeClient({"recognitions": [], "assets": [], "unclear": []})
        result = parse_with_model(text, client=client)
        assert result.parsed.template_hint == "rsu-vesting"


class TestItNeverBlocksTheFrontDoor:

    def test_no_client_falls_back_to_the_deterministic_parse(self):
        result = parse_with_model(DESCRIPTION)
        assert result.provenance.model_available is False
        assert result.parsed.value_of("trigger_semantics").value == \
            "persistent_condition"

    def test_a_model_error_falls_back_and_records_why(self):
        client = FakeClient(None, raises=TimeoutError("upstream timed out"))
        result = parse_with_model(DESCRIPTION, client=client)

        assert result.provenance.model_available is False
        assert "TimeoutError" in result.provenance.model_error
        assert result.parsed.recognitions == parse(DESCRIPTION).recognitions

    def test_malformed_json_falls_back(self):
        result = parse_with_model(DESCRIPTION, client=FakeClient("not json"))
        assert result.provenance.model_available is False
        assert result.parsed.recognitions == parse(DESCRIPTION).recognitions

    def test_a_refusal_falls_back(self):
        client = FakeClient("I can't help with investment advice.")
        result = parse_with_model(DESCRIPTION, client=client)
        assert result.provenance.model_available is False

    def test_a_fenced_json_block_is_accepted(self):
        """Models fence JSON constantly; failing on it would be self-inflicted."""
        client = FakeClient('```json\n{"recognitions": [], "assets": ["VTI"]}\n```')
        result = parse_with_model(DESCRIPTION, client=client)
        assert result.provenance.model_available is True

    def test_a_json_array_is_refused(self):
        client = FakeClient('[{"field": "dividends"}]')
        result = parse_with_model(DESCRIPTION, client=client)
        assert result.provenance.model_available is False

    def test_degrading_means_more_questions_not_wrong_answers(self):
        """The correct direction to fail in."""
        text = ("Each month I add money to VOO, topping up whenever it dips "
                "under its average.")
        client = FakeClient(None, raises=RuntimeError("no key"))

        degraded = compile_scenario(
            text, name="s", parsed=parse_with_model(text, client=client).parsed)
        assert degraded.can_save is False
        assert degraded.unresolved, (
            "a parse that recognised less must ask more, never assume more")


class TestNoAdviceEscapes:

    def test_the_prompt_forbids_advice_and_addition(self):
        prompt = build_system_prompt()
        assert "do not give advice" in prompt.lower()
        assert "does not say" in prompt.lower()

    def test_free_text_from_the_model_cannot_reach_a_compiled_field(self):
        """Only vocabulary values do. Prose has nowhere to land."""
        payload = {"recognitions": [
            {"field": "dividends",
             "value": "you should reinvest, it is better long term",
             "span": "I never sell"},
        ]}
        recognitions, _a, _u, rejections = verify_proposals(payload, DESCRIPTION)
        assert recognitions == ()
        assert rejections


class TestASavedPlanIsPinned:
    """The property that makes a model in stage 1 safe to save against.

    The workspace recompiles a plan from its stated text on every view. With a
    model in stage 1 and nothing pinned, a plan would be reinterpreted each time
    it was opened, against a model that may have changed since — so a user could
    confirm one thing and find another later, with no record that anything moved.
    """

    def test_the_parse_round_trips_losslessly(self):
        from src.mission.parse_model import parse_from_stored

        text = ("I put $500 into VTI every month and hold dividends as cash. "
                "I never sell.")
        original = parse(text)
        assert parse_from_stored(original.to_json(), text).to_json() == \
            original.to_json()

    def test_a_model_recognition_survives_the_round_trip(self):
        from src.mission.parse_model import parse_from_stored

        text = "Each month I add to VOO, topping up whenever it dips under its average."
        client = FakeClient({"recognitions": [
            {"field": "trigger_semantics", "value": "persistent_condition",
             "span": "whenever it dips under its average"},
        ], "assets": ["VOO"], "unclear": []})

        pinned = parse_with_model(text, client=client).parsed.to_json()
        restored = parse_from_stored(pinned, text)
        assert restored.value_of("trigger_semantics").value == \
            "persistent_condition"

    def test_a_stored_parse_is_re_verified_not_trusted(self):
        """It arrives through a database row or a browser. Neither is trusted."""
        from src.mission.parse_model import parse_from_stored

        text = "I put $500 into VTI every month."
        tampered = {
            **parse(text).to_json(),
            "recognitions": [{"field": "sells_allowed", "value": "false",
                              "span": "I would never sell any of it"}],
        }
        restored = parse_from_stored(tampered, text)
        assert restored.value_of("sells_allowed") is None, (
            "a span the description does not contain must not become a setting, "
            "whatever route it arrived by")

    def test_a_stored_parse_of_different_text_is_refused(self):
        from src.mission.parse_model import parse_from_stored

        stored = parse("I put $500 into VTI every month.").to_json()
        with pytest.raises(ValueError, match="different text"):
            parse_from_stored(stored, "Something else entirely.")

    def test_reopening_a_plan_does_not_consult_the_model(self):
        """Proven by the client counting calls across a save and two reopens."""
        text = "Each month I add to VOO, topping up whenever it dips under its average."
        client = FakeClient({"recognitions": [
            {"field": "trigger_semantics", "value": "persistent_condition",
             "span": "whenever it dips under its average"},
        ], "assets": ["VOO"], "unclear": []})

        from src.mission.parse_model import parse_from_stored

        pinned = parse_with_model(text, client=client).parsed.to_json()
        assert len(client.calls) == 1

        first = compile_scenario(text, name="p",
                                 parsed=parse_from_stored(pinned, text))
        second = compile_scenario(text, name="p",
                                  parsed=parse_from_stored(pinned, text))

        assert len(client.calls) == 1, "reopening a plan must not re-parse it"
        assert first.to_json() == second.to_json()

    def test_a_plan_saved_before_pinning_still_compiles(self):
        """No stored parse means the deterministic rules, as it always did."""
        text = "I put $500 into VTI every month."
        assert compile_scenario(text, name="p", parsed=None).to_json() == \
            compile_scenario(text, name="p").to_json()
