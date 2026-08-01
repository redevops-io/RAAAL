"""The strategy corpus, as a regression test.

The full run is 14,400 prompts and takes under a second, so there is no reason
to sample it. What it protects is the set of defects it found, each of which was
invisible to code inspection and to the existing suite:

    a stated weighting rule was discarded when the parser recognised fewer than
    two tickers, so 94% of deliberately contradictory descriptions compiled
    without reporting the contradiction

    four of the nine cadences the corpus uses had no recognizer, so a user
    writing "annually" — the most common cadence in the catalog — was asked how
    often their contribution arrives immediately after saying so

    "out of that contribution" was not read as naming a funding source, so a
    stated choice became a question

    a bare `annual` pattern, added while fixing the above, matched "annual
    rebalance" in a *benchmark* clause and read the user's contribution cadence
    out of a sentence about what to compare against

    the life-event template hand-off was appended to `unresolved` after the
    scenario provenance was built, so the confirmation screen listed it under
    "we still need" while the Save button stayed enabled

Paired prompts that must compile differently are permanent regression tests, per
the load-test plan. That is what the paraphrase classes are.
"""
from __future__ import annotations

import pytest

from src.loadtest.catalog import load_scenarios, load_strategies
from src.loadtest.harness import Report, run_corpus, run_prompt
from src.loadtest.paraphrase import Expect, Klass, corpus, paraphrases

STRATEGIES = load_strategies()


@pytest.fixture(scope="module")
def report() -> Report:
    return Report(run_corpus(corpus(STRATEGIES, 100)))


class TestTheCorpusItself:

    def test_the_catalog_is_committed_and_complete(self):
        assert len(STRATEGIES) == 144
        assert len({s.family for s in STRATEGIES}) == 18
        assert len(load_scenarios()) == 8

    def test_generation_is_deterministic(self):
        """A defect found in a 14,400-prompt run must be reproducible from two
        integers, on any machine."""
        first = paraphrases(STRATEGIES[7], 20)
        second = paraphrases(STRATEGIES[7], 20)
        assert [p.text for p in first] == [p.text for p in second]
        assert first[3].prompt_id == "WM-0008#003"

    def test_every_class_is_exercised(self):
        prompts = corpus(STRATEGIES, 100)
        assert len(prompts) == 14_400
        assert {p.klass for p in prompts} == set(Klass)

    def test_every_prompt_declares_what_it_owes(self):
        """A run reporting "14,400 compiled" has measured a loop."""
        assert all(p.expect in set(Expect) for p in corpus(STRATEGIES, 8))


class TestNothingBreaks:

    def test_no_prompt_crashes_the_compiler(self, report):
        crashes = report.crashes
        assert not crashes, "\n".join(
            f"{o.prompt_id}: {o.error}" for o in crashes[:10])

    def test_every_prompt_gets_what_it_is_owed(self, report):
        problems = report.distinct_problems()
        assert not problems, "\n".join(
            f"{len(ids):,} x {message}" for message, ids in problems.items())

    def test_a_fully_specified_description_reaches_a_saveable_plan(self, report):
        """Measured against the prompts that *claim* to be complete.

        The COMPLETE class also contains wordings downgraded to
        ASKS_A_QUESTION — a row whose account the compiler cannot place, or
        whose cadence is an event. Counting those in the denominator measures
        the corpus rather than the compiler.
        """
        claims = [o for o in report.outcomes
                  if o.klass == Klass.COMPLETE.value
                  and o.expect == Expect.COMPILES_SAVEABLE.value]
        # Nothing left to ask. `can_save` is additionally False while an
        # inference is unconfirmed, which is correct and is what the
        # confirmation screen exists for — asserting on it would report the
        # confirmation step itself as a failure.
        ready = [o for o in claims if not o.unresolved]
        assert len(ready) == len(claims), (
            f"{len(claims) - len(ready)} of {len(claims)} fully specified "
            "descriptions still have an open question; the pilot journey ends "
            "at 'save the plan'")


class TestTheDefectsItFound:
    """Each of these failed before the corpus existed."""

    def test_a_stated_weighting_rule_survives_a_single_ticker(self):
        """The guard existed to avoid inventing a question, not to discard an
        answer."""
        from src.mission.compiler import compile_scenario

        text = ("I buy $1,500 of VTI every month and rebalance them back to "
                "equal weights every quarter, but I never sell anything.")
        assert compile_scenario(text, name="t").contradictions

    def test_a_single_holding_is_still_not_asked_about_weighting(self):
        """The other direction. Asking about something the user did not raise
        is noise that trains people to click through confirmations."""
        from src.mission.compiler import compile_scenario

        result = compile_scenario("I buy $500 of VTI every month and never sell.",
                                  name="t")
        assert not any(u.field == "weighting" for u in result.unresolved)

    @pytest.mark.parametrize("phrase,expected", [
        ("annually", "annual"), ("every year", "annual"), ("yearly", "annual"),
        ("quarterly", "quarterly"), ("every quarter", "quarterly"),
        ("every payday", "payroll"), ("out of each paycheck", "payroll"),
        ("daily", "daily"), ("every day", "daily"),
        ("every other week", "biweekly"), ("monthly", "monthly"),
    ])
    def test_the_cadences_the_catalog_uses_are_recognised(self, phrase, expected):
        from src.mission.compiler import parse

        found = parse(f"I invest $500 {phrase}.").value_of("cadence")
        assert found is not None and found.value == expected

    def test_a_cadence_is_not_read_out_of_a_benchmark_clause(self):
        """`annual` is an adjective. A bare pattern for it matched "annual
        rebalance" in a sentence about what to compare against."""
        from src.mission.compiler import parse

        assert parse("Compare it against buy and hold and annual rebalance."
                     ).value_of("cadence") is None

    @pytest.mark.parametrize("phrase", [
        "out of that contribution", "out of the contribution",
        "from my contribution", "out of the usual transfer",
    ])
    def test_ordinary_wordings_of_funding_source_are_read(self, phrase):
        from src.mission.compiler import parse

        found = parse(f"I buy more {phrase}.").value_of("funding_source")
        assert found is not None and found.value == "contribution"

    def test_a_non_blocking_offer_is_not_in_the_blocking_list(self):
        """The confirmation screen and the Save button must agree."""
        from src.mission.compiler import compile_scenario

        result = compile_scenario(
            "My RSUs vest quarterly and I sell the vested shares. I put $2,000 "
            "in every month and never sell.", name="t",
            benchmark_rule="benchmark-policy/public-default@1")

        assert result.template_offer is not None
        assert all(not u.field.startswith("template:") for u in result.unresolved)
        assert result.confirmation()["a_better_route"]


class TestWhatTheHarnessMeasures:

    def test_a_recommendation_request_never_compiles_to_a_saveable_plan(self, report):
        """The platform does not answer "which is best"."""
        bait = [o for o in report.outcomes
                if o.klass == Klass.RECOMMENDATION_BAIT.value]
        assert bait and not any(o.can_save for o in bait)

    def test_an_underspecified_description_always_asks(self, report):
        under = [o for o in report.outcomes
                 if o.klass == Klass.UNDERSPECIFIED.value]
        assert under and all(o.unresolved or o.inferred for o in under)

    def test_latency_is_recorded_per_stage(self, report):
        """"compile took 40 ms" cannot be acted on; "stage 1 took 38 of it" can."""
        latency = report.latency()
        assert latency["parse_us"]["n"] == 14_400
        assert latency["total_us"]["p95"] > 0
