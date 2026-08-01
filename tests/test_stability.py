"""Semantic stability, and the defect class it exposed.

Recognition accuracy asks whether the compiler understood. Stability asks
whether it understood *the same thing every time*. A parser can be perfectly
accurate on a benchmark and still be unusable if a synonym changes the answer.

The benchmark found something stronger than instability on its first run: a
choice that was recognised, confirmed to the user, and never represented at all.
"""
from __future__ import annotations

import pytest

from src.loadtest.catalog import load_strategies
from src.loadtest.stability import facts_for, run, summarize, wordings
from src.mission.compiler import compile_scenario

BENCHMARK_RULE = "benchmark-policy/public-default@1"
FAMILIES = [s for s in load_strategies() if facts_for(s)]


def rule_hash(text: str) -> str:
    result = compile_scenario(text, name="s", version=1,
                              benchmark_rule=BENCHMARK_RULE)
    return result.scenario.rule_hash


@pytest.fixture(scope="module")
def families():
    return run(FAMILIES, 40)


class TestTheGeneratorVariesWhatItClaims:

    def test_wordings_are_sampled_across_the_whole_product(self):
        """`itertools.product` varies its rightmost axis fastest.

        Taking the first N held the verb and the asset ordering fixed — the two
        axes most likely to break stability — and reported 100% while never
        testing them.
        """
        facts = facts_for(FAMILIES[2])
        texts = wordings(facts, 40)
        assert len({t.split(",")[0] for t in texts}) > 1, "the verb never varied"

    def test_every_wording_means_the_same_thing(self):
        """The generator's contract. A near-synonym that shifts meaning would
        make a real instability look like a generator bug, and the reverse."""
        facts = facts_for(FAMILIES[0])
        texts = wordings(facts, 12)
        assert len(set(texts)) == len(texts)
        for text in texts:
            assert f"{facts.amount:,}" in text
            for asset in facts.assets:
                assert asset in text


class TestStability:

    def test_paraphrases_of_one_plan_compile_to_one_plan(self, families):
        summary = summarize(families)
        unstable = [f for f in families if not f.stable]
        assert not unstable, "\n".join(
            f"{f.strategy_id}: {len(f.distinct)} distinct rule hashes\n"
            f"{f.divergence()}" for f in unstable[:5])
        assert summary["stability_rate"] == 100.0

    def test_the_check_can_fail(self):
        """A stability test that cannot detect a difference is decoration."""
        base = ("I put $1,000 into VTI and BND, yearly, in my taxable account, "
                "on the first trading day of the period, reinvesting the "
                "dividends, and I never sell.")
        assert rule_hash(base) != rule_hash(base.replace("VTI and BND",
                                                         "VTI and SCHD"))

    def test_asset_order_is_not_meaning(self, families):
        base = ("I put $1,000 into VTI and BND, yearly, in my taxable account, "
                "on the first trading day of the period, reinvesting the "
                "dividends, and I never sell.")
        assert rule_hash(base) == rule_hash(base.replace("VTI and BND",
                                                         "BND and VTI"))

    def test_a_synonym_is_not_meaning(self):
        base = ("I put $1,000 into VTI and BND, yearly, in my taxable account, "
                "on the first trading day of the period, reinvesting the "
                "dividends, and I never sell.")
        for swap, with_ in (("I put $1,000 into", "I invest $1,000 in"),
                            ("yearly", "annually"),
                            ("in my taxable account", "in my brokerage account")):
            assert rule_hash(base) == rule_hash(base.replace(swap, with_)), swap


class TestRecognizedIsNotRepresented:
    """The defect the benchmark found, and the shape of it.

        recognised -> confirmed to the user -> never represented

    The compiler read "hold the dividends as cash", the confirmation screen
    quoted it back under "you stated", and the compiled scenario contained no
    trace of it. Two materially different strategies shared one `content_hash`.
    This is declaration-without-behaviour, the failure the whole project exists
    to close, inside the compiler.
    """

    BASE = ("I put $1,000 into VTI and BND, yearly, in my taxable account, on "
            "the first trading day of the period, {d}, and I never sell.")

    def test_dividend_treatment_reaches_the_compiled_scenario(self):
        reinvest = compile_scenario(self.BASE.format(d="reinvesting the dividends"),
                                    name="s", version=1,
                                    benchmark_rule=BENCHMARK_RULE)
        cash = compile_scenario(self.BASE.format(d="holding the dividends as cash"),
                                name="s", version=1, benchmark_rule=BENCHMARK_RULE)

        assert reinvest.scenario.holdings_policy.dividend_policy == "reinvested"
        assert cash.scenario.holdings_policy.dividend_policy == "held_as_cash"

    def test_two_different_strategies_do_not_share_an_identity(self):
        """Reinvesting compounds the position; holding as cash does not. Over a
        long horizon these are materially different plans."""
        reinvest = compile_scenario(self.BASE.format(d="reinvesting the dividends"),
                                    name="s", version=1,
                                    benchmark_rule=BENCHMARK_RULE)
        cash = compile_scenario(self.BASE.format(d="holding the dividends as cash"),
                                name="s", version=1, benchmark_rule=BENCHMARK_RULE)

        assert reinvest.scenario.rule_hash != cash.scenario.rule_hash
        assert reinvest.scenario.content_hash != cash.scenario.content_hash

    def test_the_limitation_is_declared_rather_than_left_looking_enforced(self):
        """Representing it without saying it is not simulated moves the defect
        one layer up: the scenario looks enforced and the figure ignores it."""
        from src.mission.scenario import UNSIMULATED
        from src.workspace.routes import declare_unsimulated

        compiled = compile_scenario(
            self.BASE.format(d="holding the dividends as cash"), name="s",
            version=1, benchmark_rule=BENCHMARK_RULE)
        scope = declare_unsimulated(compiled.scenario, None)

        entry = scope["declared_but_not_simulated"]["dividend_policy"]
        assert entry["declared"] == "held_as_cash"
        assert "price series only" in entry["why"]
        assert "dividend_policy" in UNSIMULATED

    def test_every_unsimulated_declaration_is_disclosed(self):
        """Derived from the scenario, not hardcoded, so a behaviour that becomes
        simulatable stops being disclosed by deleting one entry."""
        from src.mission.scenario import UNSIMULATED
        from src.workspace.routes import declare_unsimulated

        compiled = compile_scenario(self.BASE.format(d="reinvesting the dividends"),
                                    name="s", version=1,
                                    benchmark_rule=BENCHMARK_RULE)
        disclosed = declare_unsimulated(
            compiled.scenario, None)["declared_but_not_simulated"]
        assert set(disclosed) == set(UNSIMULATED)
