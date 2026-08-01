"""Round-trip fidelity: Mission -> words -> the same Mission.

    stability      many texts   -> one Mission
    round-trip     one Mission  -> text -> the same Mission

Different directions, different failures. Stability catches a compiler that
reads wording as meaning. Round-trip catches one that cannot *say* what it
understood — a field that survives compilation with no way back into language is
a field no user can ever correct.

Only `SPECIFICATION` claims losslessness. A summary that round-tripped by
accident would be worse than one that does not: someone would eventually paste a
plan card back in and expect identical behaviour.
"""
from __future__ import annotations

import pytest

from src.loadtest.catalog import load_strategies
from src.loadtest.paraphrase import Klass, corpus
from src.loadtest.roundtrip import cycles, run, trip
from src.mission.compiler import compile_scenario
from src.mission.render import Purpose, render

BR = "benchmark-policy/public-default@1"

#: The wordings that exercise every field which has ever gone missing.
PROBES = {
    "dividends reinvested": "I put $500 into VTI every month, reinvesting the dividends, and I never sell.",
    "dividends as cash": "I put $500 into VTI every month, holding the dividends as cash, and I never sell.",
    "simple average": "Whenever SPY is below its simple 200 day moving average I buy $500 of VTI with additional cash, every month.",
    "exponential average": "Whenever SPY is below its exponential 200 day moving average I buy $500 of VTI with additional cash, every month.",
    "funding from contribution": "My monthly contribution is $500. Whenever SPY is below its 200 day moving average I buy more out of that contribution.",
    "funding from extra cash": "My monthly contribution is $500. Whenever SPY is below its 200 day moving average I buy more with additional cash.",
    "crossing event": "On the day SPY crosses below its 200 day moving average I buy $500 of VTI with additional cash, every month.",
    "equal dollars": "I put $500 into VTI and BND every month, buying equal dollars at each purchase, and I never sell.",
    "rebalanced": "I put $500 into VTI and BND every month, rebalancing them back to equal weights.",
    "calendar day rule": "I put $500 into VTI every month, on the first calendar day of the month, and I never sell.",
    "session day rule": "I put $500 into VTI every month, on the first trading day of the period, and I never sell.",
    "roth account": "I put $500 into VTI every month in my Roth IRA, and I never sell.",
    "annual cadence": "I put $500 into VTI every year, reinvesting the dividends, and I never sell.",
}


@pytest.fixture(scope="module")
def report():
    texts = [p.text for p in corpus(load_strategies(), 16)
             if p.klass in {Klass.COMPLETE, Klass.PERSISTENT_VS_EVENT,
                            Klass.EQUAL_WEIGHT, Klass.FUNDING_SOURCE,
                            Klass.CALENDAR_VS_SESSION}]
    return run(texts)


class TestSpecificationIsLossless:

    @pytest.mark.parametrize("name", sorted(PROBES))
    def test_each_field_that_has_ever_gone_missing(self, name):
        result = trip(PROBES[name], Purpose.SPECIFICATION)
        assert result is not None and result.exact, (
            f"{name}\n  from: {PROBES[name]}\n  back: {result.regenerated}\n  "
            + "\n  ".join(str(c) for c in (result.diff.changes if result.diff else [])))

    def test_the_whole_corpus_round_trips(self, report):
        summary = report.summarize()[Purpose.SPECIFICATION.value]
        assert summary["exact_rate"] == 100.0, (
            f"{summary['n'] - summary['exact']} of {summary['n']} lost something")

    def test_all_three_identities_are_preserved(self, report):
        summary = report.summarize()[Purpose.SPECIFICATION.value]
        assert summary["rule_hash_kept"] == summary["n"]
        assert summary["schedule_hash_kept"] == summary["n"]

    def test_a_specification_omits_nothing_by_choice(self):
        _p, result = None, compile_scenario(PROBES["dividends as cash"],
                                            name="p", version=1,
                                            benchmark_rule=BR)
        assert render(result.scenario, Purpose.SPECIFICATION).omitted == ()


class TestProvenanceSurvivesTheTrip:
    """The subtle half. Values can round-trip while provenance does not."""

    def test_an_inferred_value_is_not_restated_as_a_decision(self):
        """Writing an inference out returns it as something the user stated,
        and the confirmation screen then asks them to confirm nothing."""
        text = "I put $500 into VTI every month and I never sell."
        result = compile_scenario(text, name="p", version=1, benchmark_rule=BR)
        inferred = {i.field for i in result.scenario.provenance.inferred}
        assert "dividends" in inferred

        regenerated = render(result.scenario, Purpose.SPECIFICATION).text
        assert "dividend" not in regenerated.lower()
        assert trip(text, Purpose.SPECIFICATION).exact

    def test_an_open_question_is_not_answered_by_omission(self):
        """A mention of a condition with no semantics must be asked again.

        Dropping the mention made the regenerated text stop asking — the one
        outcome a specification must never produce.
        """
        text = ("My monthly contribution is $500. When VTI drops I buy more "
                "with additional cash.")
        first = compile_scenario(text, name="p", version=1, benchmark_rule=BR)
        assert any(u.field == "trigger_semantics" for u in first.unresolved)

        result = trip(text, Purpose.SPECIFICATION)
        assert result.exact
        again = compile_scenario(result.regenerated, name="p", version=1,
                                 benchmark_rule=BR)
        assert any(u.field == "trigger_semantics" for u in again.unresolved)

    def test_a_stated_weighting_survives_on_a_single_holding(self):
        """Guarding the clause on holding count dropped a stated rule, which
        then recompiled to the default — the strategy changed on a round trip."""
        assert trip("I keep VTI at equal weights, rebalancing once a year "
                    "with $750.", Purpose.SPECIFICATION).exact


class TestIdentityDoesNotDrift:

    def test_three_cycles_produce_one_identity(self):
        """A trip that is lossless once and shifts on the third pass is worse
        than one that fails immediately: it looks correct in every test."""
        for text in list(PROBES.values()):
            seen = cycles(text, 3)
            assert len(set(seen)) == 1, f"{text} drifted: {seen}"

    def test_the_corpus_does_not_drift(self):
        texts = [p.text for p in corpus(load_strategies(), 4)
                 if p.klass is Klass.COMPLETE][:120]
        drifted = [t for t in texts if len(set(cycles(t, 3))) != 1]
        assert not drifted, drifted[:3]


class TestPurposeIsDeclared:

    def test_only_specification_claims_losslessness(self):
        assert Purpose.SPECIFICATION.claims_lossless
        assert not Purpose.SUMMARY.claims_lossless
        assert not Purpose.EXPLANATION.claims_lossless

    def test_a_summary_says_what_it_dropped(self):
        result = compile_scenario(PROBES["exponential average"], name="p",
                                  version=1, benchmark_rule=BR)
        rendered = render(result.scenario, Purpose.SUMMARY)
        assert "dividend_policy" in rendered.omitted
        assert "funding_source" in rendered.omitted

    def test_a_summary_is_not_expected_to_round_trip(self, report):
        """And demonstrably does not. A summary that round-tripped by accident
        would be worse: someone would paste a plan card back in expecting
        identical behaviour."""
        summary = report.summarize()[Purpose.SUMMARY.value]
        assert summary["exact_rate"] < 50.0

    def test_an_explanation_disclaims_itself(self):
        result = compile_scenario(PROBES["exponential average"], name="p",
                                  version=1, benchmark_rule=BR)
        rendered = render(result.scenario, Purpose.EXPLANATION)
        assert not rendered.purpose.claims_lossless
        assert any("not a specification" in note for note in rendered.omitted)
