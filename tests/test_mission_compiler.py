"""Prompts that read alike and must compile differently.

This is the compiler's real test. Two descriptions can be a few words apart and
economically unrelated, and a fluent model smooths precisely those differences
away because both readings sound like the same sentence.

The success criterion is not that the conversation feels smooth:

    Two competent humans reading the compiler confirmation should agree on
    exactly what will be simulated, which choices came from the user, which came
    from the system, which remain unresolved, and which output fields each
    statement controls.

Every test here is a check on some part of that sentence.
"""
from __future__ import annotations

import pytest

from src.mission.compiler import Origin, compile_scenario, parse
from src.mission.defaults import DEFAULT_SET
from src.mission.spec import Objective


def rule_of(text: str, **kw):
    return compile_scenario(text, **kw).scenario


class TestPairsThatMustNotCollapse:
    """Seven pairs. Each differs by a few words and by a lot of money."""

    def test_persistent_condition_versus_crossing_event(self):
        persistent = rule_of("Buy AMZN and NVDA whenever SPY is below its "
                             "200-day average.")
        crossing = rule_of("Buy AMZN and NVDA only on the day SPY crosses below "
                           "its 200-day average.")

        assert persistent.rule_hash != crossing.rule_hash, (
            "a persistent condition buys through the whole decline; a crossing "
            "event buys once"
        )
        assert persistent.event_program[1]["semantics"] == "persistent_condition"
        assert crossing.event_program[1]["semantics"] == "crossing_event"

    def test_equal_at_purchase_versus_equal_maintained(self):
        at_purchase = rule_of("Buy AMZN and NVDA equally at purchase.")
        maintained = rule_of("Buy AMZN and NVDA and rebalance them to equal "
                             "weights.")

        assert at_purchase.allocation_rule.weighting == "equal_weight_at_purchase"
        assert maintained.allocation_rule.weighting == "equal_weight_maintained"
        assert at_purchase.rule_hash != maintained.rule_hash

    def test_contribution_versus_additional_cash(self):
        from_contribution = compile_scenario(
            "I put in $2000 every month. Whenever SPY is below its 200-day "
            "average, use the monthly contribution to buy AMZN.")
        extra = compile_scenario(
            "I put in $2000 every month. Whenever SPY is below its 200-day "
            "average, buy AMZN with additional cash.")

        def funding(result):
            found = [s for s in result.stated if "contribution" in s.lower()
                     or "additional" in s.lower()]
            return found

        assert funding(from_contribution) != funding(extra)
        assert any("additional cash" in s.lower() for s in extra.stated)

    def test_earnings_date_versus_first_session_after(self):
        on_date = parse("Buy on the earnings date.")
        after = parse("Buy the first trading day after earnings.")

        assert on_date.value_of("earnings_timing").value == "earnings_date"
        assert after.value_of("earnings_timing").value == "first_session_after_earnings"

    def test_dividends_reinvested_versus_held_as_cash(self):
        reinvested = parse("Reinvest the dividends.")
        cash = parse("Keep the dividends as cash.")

        assert reinvested.value_of("dividends").value == "reinvested"
        assert cash.value_of("dividends").value == "held_as_cash"

    def test_selling_vested_shares_versus_exercising_options(self):
        sell = parse("Sell the vested shares each quarter.")
        exercise = parse("Exercise the options and sell.")

        assert sell.value_of("vesting_action").value == "sell_vested_shares"
        assert exercise.value_of("vesting_action").value == "exercise_and_sell"

    def test_first_calendar_day_versus_first_trading_session(self):
        calendar = rule_of("Invest $2000 on the first calendar day of each month.")
        session = rule_of("Invest $2000 on the first trading day of each month.")

        assert calendar.flow_schedule.day_rule == "calendar_first_rolled_forward"
        assert session.flow_schedule.day_rule == "first_session_of_period"
        assert calendar.flow_schedule.schedule_hash != session.flow_schedule.schedule_hash


class TestUnrecognizedBecomesAQuestionNotADefault:
    """A compiler that guesses when it does not know is indistinguishable from
    one that knows, right up until it is wrong."""

    def test_an_ambiguous_share_class_is_asked_about(self):
        result = compile_scenario("Buy Google whenever SPY is below its 200-day average.")
        questions = [u for u in result.unresolved
                     if u.field.startswith("asset_identity")]

        assert questions
        assert "GOOGL" in questions[0].question and "GOOG" in questions[0].question
        assert not result.can_save

    def test_a_missing_contribution_is_asked_about(self):
        result = compile_scenario("Buy AMZN whenever SPY is below its 200-day average.")
        assert any(u.field == "amount" for u in result.unresolved)

    def test_a_missing_benchmark_is_asked_about(self):
        """A result with nothing to compare it to cannot be interpreted."""
        result = compile_scenario("Invest $2000 every month in VTI.")
        assert any(u.field == "benchmark_set" for u in result.unresolved)

    def test_supplying_a_benchmark_rule_settles_it(self):
        result = compile_scenario(
            "Invest $2000 every month in VTI.",
            benchmark_rule="benchmark-policy/public-default@1")

        assert not any(u.field == "benchmark_set" for u in result.unresolved)
        assert result.scenario.benchmark_set.generated_by_rule == \
            "benchmark-policy/public-default@1"

    def test_every_unresolved_item_states_its_consequence(self):
        """A question without a consequence gets answered at random."""
        result = compile_scenario("Buy Google whenever SPY is below its 200DMA.")
        for u in result.unresolved:
            assert u.why_it_matters, f"{u.field} asks without saying why"
            assert len(u.why_it_matters) > 30


class TestDefaultsAreVersionedAndVisible:
    def test_the_compilation_names_the_default_set_it_used(self):
        result = compile_scenario("Invest $2000 every month in VTI.")
        assert result.defaults_ref == "compiler-defaults/us-equity-scenario@1"

    def test_every_default_carries_its_consequence(self):
        for entry in DEFAULT_SET.defaults.values():
            assert entry.why and len(entry.why) > 30
            assert entry.changes_result

    def test_inferences_reach_the_user_unconfirmed(self):
        result = compile_scenario("Invest $2000 every month in VTI.")

        assert result.inferred
        assert all(not i.confirmed for i in result.inferred)
        assert not result.can_save

    def test_a_stated_choice_beats_the_default(self):
        stated = compile_scenario("Buy AMZN when SPY crosses below its "
                                  "exponential 200-day average.")
        inferred_fields = {i.field for i in stated.inferred}

        assert "moving_average_kind" not in inferred_fields
        assert any("exponential" in s.lower() for s in stated.stated)

    def test_the_default_set_is_content_hashed(self):
        assert len(DEFAULT_SET.content_hash) == 64


class TestSimulatableIsNotSaveable:
    def test_an_underspecified_plan_may_still_be_run(self):
        """Running a provisional interpretation is how a user sees what it means."""
        result = compile_scenario("Invest $2000 every month in VTI.")

        assert result.can_simulate
        assert not result.can_save
        assert result.status == "NEEDS_INPUT"

    def test_a_structurally_impossible_plan_may_not_be_run(self):
        """There is no shape to show for a plan that cannot execute as written."""
        result = compile_scenario(
            "Invest $2000 every month in AMZN and NVDA, rebalance them to equal "
            "weights, and never sell.")

        assert result.contradictions
        assert not result.can_simulate
        assert result.status == "BLOCKED"

    def test_the_contradiction_names_both_sides(self):
        result = compile_scenario(
            "Invest $2000 every month in AMZN and NVDA, rebalance them to equal "
            "weights, and never sell.")
        [conflict] = result.contradictions

        assert "holdings_policy" in conflict.between
        assert "allocation_rule" in conflict.between
        assert "selling what rose" in conflict.detail

    def test_never_selling_alone_is_fine(self):
        result = compile_scenario(
            "Invest $2000 every month in AMZN and NVDA equally at purchase, and "
            "never sell.",
            benchmark_rule="benchmark-policy/public-default@1")

        assert not result.contradictions
        assert result.can_simulate

    def test_contradictions_are_detected_in_the_compiled_form(self):
        """Not in the prose. A check reading only the input trusts the parser."""
        result = compile_scenario(
            "Rebalance AMZN and NVDA to equal weights and never sell.")
        assert result.scenario.self_conflicts()


class TestTheConfirmationIsCheckableByTwoPeople:
    def test_it_groups_statements_the_way_a_reader_checks_them(self):
        confirmation = compile_scenario(
            "Invest $2000 every month in AMZN and NVDA, rebalance to equal "
            "weights, never sell.").confirmation()

        for group in ("you_stated", "we_inferred", "these_conflict", "we_still_need"):
            assert group in confirmation

    def test_every_inference_names_the_field_it_controls(self):
        confirmation = compile_scenario("Invest $2000 every month in VTI.").confirmation()

        for statement in confirmation["we_inferred"]:
            assert statement["controls"], "a statement controlling nothing is copy"
            assert statement["why"]

    def test_every_open_question_names_the_field_it_controls(self):
        confirmation = compile_scenario("Buy Google when SPY is below its 200DMA.").confirmation()

        for statement in confirmation["we_still_need"]:
            assert statement["controls"]

    def test_stated_spans_are_quoted_not_paraphrased(self):
        """A paraphrase is the compiler's account of what was said."""
        text = "Buy AMZN whenever SPY is below its 200-day average."
        result = compile_scenario(text)

        for span in result.stated:
            assert span.lower() in text.lower()

    def test_the_confirmation_carries_the_gate_state(self):
        confirmation = compile_scenario("Invest $2000 every month in VTI.").confirmation()

        assert confirmation["can_simulate"] is True
        assert confirmation["can_save"] is False
        assert confirmation["defaults_ref"].startswith("compiler-defaults/")


class TestTheParseStageIsQuarantined:
    def test_the_parse_output_is_data_not_decisions(self):
        """Everything downstream must be reproducible from this alone."""
        parsed = parse("Invest $2000 every month in VTI.")

        assert parsed.text
        assert all(isinstance(r.value, str) for r in parsed.recognitions)
        assert all(r.span for r in parsed.recognitions)

    def test_compiling_the_same_text_twice_is_identical(self):
        text = "Invest $2000 every month in AMZN and NVDA equally at purchase."
        first, second = compile_scenario(text), compile_scenario(text)

        assert first.scenario.content_hash == second.scenario.content_hash
