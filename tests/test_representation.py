"""Recognition is not representation.

The compiler can be correct at two layers and wrong between them:

    Recognized    the parser read the phrase             correct
    Represented   it reached the compiled scenario       MISSING
    Validated     checks ran over what was represented   vacuous
    Compiled      a scenario was produced                correct
    Confirmed     the screen quoted the phrase back      correct

Three defects had exactly that shape, and none was a parser bug — the parser was
right every time. The gap was between reading a thing and carrying it, and
nothing checked that edge because both ends looked fine.

These tests make the edge mechanical, so the next one is a diff rather than a
discovery.
"""
from __future__ import annotations

import pytest

from src.mission.compiler import _RULES, compile_scenario, parse
from src.mission.representation import (
    CARRIES,
    UNCARRIED,
    representation_gaps,
    undeclared_fields,
)

BR = "benchmark-policy/public-default@1"


def compiled(text: str):
    parsed = parse(text)
    return parsed, compile_scenario(text, name="s", version=1,
                                    benchmark_rule=BR, parsed=parsed)


class TestEveryRecogniserNamesADestination:

    def test_no_field_is_read_without_somewhere_to_go(self):
        """The check that stops the map going stale.

        A new recogniser with no entry fails here rather than quietly reading
        something nothing carries — which is how all three defects arrived.
        """
        known = {field for field, _value, _pattern in _RULES} | {"cadence", "amount"}
        assert undeclared_fields(sorted(known)) == []

    def test_a_field_declared_as_uncarried_says_why(self):
        """An entry there is a decision someone wrote down. A bare exemption
        list is a place for defects to hide."""
        for field, exemption in UNCARRIED.items():
            assert len(exemption.reason) > 60, (
                f"{field} is exempted without an explanation")

    def test_every_exemption_names_an_owner_or_a_blocker(self):
        """What makes an exemption expire.

        Without it, "deliberately uncarried" is permanent: nothing can find the
        entries that stopped being true when the responsible artifact shipped.
        """
        for field, exemption in UNCARRIED.items():
            assert exemption.status in {"delegated", "unsupported"}, field
            if exemption.status == "delegated":
                assert exemption.owner, f"{field} delegates to nobody"
            else:
                assert exemption.blocked_by, f"{field} is blocked by nothing"

    def test_an_exemption_expires_when_its_blocker_arrives(self):
        from src.mission.representation import obsolete_exemptions

        assert obsolete_exemptions(["EarningsCalendarRuntime"]) == \
            ["earnings_timing"]
        assert obsolete_exemptions(["template/rsu-vesting@1"]) == \
            ["vesting_action"]
        assert obsolete_exemptions([]) == []

    def test_an_exemption_cannot_be_written_without_one(self):
        from src.mission.representation import Exemption

        with pytest.raises(ValueError, match="must name the artifact"):
            Exemption(status="delegated", reason="x" * 70)
        with pytest.raises(ValueError, match="must name what would unblock"):
            Exemption(status="unsupported", reason="x" * 70)

    def test_the_two_lists_do_not_overlap(self):
        assert not (set(CARRIES) & set(UNCARRIED))


class TestTheThreeDefectsItFound:
    """Each produced an identical content hash for two different strategies."""

    def test_dividend_treatment_is_represented(self):
        base = ("I put $500 into VTI monthly, {d}, and never sell.")
        _p, a = compiled(base.format(d="reinvesting the dividends"))
        _p, b = compiled(base.format(d="holding the dividends as cash"))
        assert a.scenario.content_hash != b.scenario.content_hash

    def test_the_moving_average_estimator_is_represented(self):
        """Simple and exponential averages cross at different times, so they
        are different rules and must not share an identity."""
        base = ("Whenever SPY is below its {k} 200 day moving average I buy "
                "$500 of VTI with additional cash, monthly, and never sell.")
        _p, a = compiled(base.format(k="simple"))
        _p, b = compiled(base.format(k="exponential"))
        assert a.scenario.rule_hash != b.scenario.rule_hash
        assert a.scenario.event_program[0]["estimator"] == "simple"
        assert b.scenario.event_program[0]["estimator"] == "exponential"

    def test_the_funding_source_is_represented(self):
        """Out of the contribution the plan invests the same total; as extra
        cash it invests more, and more money in a rising market always looks
        like a better rule. The compiler asked the question for years and never
        carried the answer."""
        base = ("My monthly contribution is $500. When VTI drops I buy more "
                "{f}, monthly.")
        _p, a = compiled(base.format(f="out of that contribution"))
        _p, b = compiled(base.format(f="with additional cash on top of it"))
        assert a.scenario.flow_schedule.funding_source == "contribution"
        assert b.scenario.flow_schedule.funding_source == "additional_cash"
        assert a.scenario.content_hash != b.scenario.content_hash
        assert (a.scenario.flow_schedule.schedule_hash
                != b.scenario.flow_schedule.schedule_hash)


class TestTheAccountVocabularyGap:
    """The largest source of friction the model evaluation found.

    `tax_treatment` had always been on the scenario and in the content hash, and
    nothing ever set it — so every plan compiled from prose was NONE_APPLIED and
    a Roth compared as identical to a taxable account, which is this project's
    own founding example of a defect. 80.5% of model-assisted cases raised an
    extra question because account type mapped to nothing.
    """

    @pytest.mark.parametrize("phrase,expected", [
        ("in my taxable brokerage account", "TAXABLE"),
        ("in my brokerage account", "TAXABLE"),
        ("in my Roth IRA", "ROTH"),
        ("in my Roth 401(k)", "ROTH_401K"),
        ("in my 401(k)", "TRADITIONAL_401K"),
        ("in my 401k", "TRADITIONAL_401K"),
        ("in my traditional IRA", "TRADITIONAL_IRA"),
    ])
    def test_the_account_reaches_the_compiled_scenario(self, phrase, expected):
        _p, result = compiled(f"I put $500 into VTI monthly {phrase} and never sell.")
        assert result.scenario.tax_treatment == expected

    def test_roth_and_taxable_do_not_share_an_identity(self):
        """The founding example, now actually enforced for prose."""
        _p, roth = compiled("I put $500 into VTI monthly in my Roth IRA and never sell.")
        _p, taxable = compiled(
            "I put $500 into VTI monthly in my taxable account and never sell.")
        assert roth.scenario.content_hash != taxable.scenario.content_hash

    def test_an_unnamed_account_is_asked_about_not_assumed(self):
        _p, result = compiled("I put $500 into VTI monthly and never sell.")
        question = next(u for u in result.unresolved if u.field == "account_type")
        assert len(question.why_it_matters) > 100, (
            "a question without a consequence gets answered at random")
        assert "Roth" in question.why_it_matters

    def test_an_unplaceable_account_is_still_a_question(self):
        """Donor-advised funds, inherited IRAs and "my retirement accounts" are
        not modelled. Guessing between traditional and Roth is the defect."""
        for phrase in ("in my inherited IRA account", "in my retirement accounts",
                       "in my DAF account"):
            _p, result = compiled(
                f"I put $500 into VTI monthly {phrase} and never sell.")
            assert any(u.field == "account_type" for u in result.unresolved), phrase


class TestSpyIsAHoldingWhenItIsBought:
    """The model read this correctly and the deterministic rules did not.

    `SPY` is reserved because it is usually the *reference* in a trend rule.
    "I buy $500 of SPY every week" names a holding, and the reserved list made
    that compile to no assets at all.
    """

    @pytest.mark.parametrize("text", [
        "I buy $500 of SPY every week and never sell.",
        "$500 goes into SPY, weekly, in my brokerage account.",
        "I invest $500 in SPY, weekly.",
        "I contribute $500 to SPY every month.",
    ])
    def test_spy_is_read_as_a_holding(self, text):
        assert "SPY" in parse(text).assets

    @pytest.mark.parametrize("text", [
        "Whenever SPY is below its 200 day moving average I buy $500 of VTI.",
        "On the day SPY crosses below its average I buy VTI.",
        "I buy VTI when SPY is above its 50 day moving average.",
    ])
    def test_spy_is_read_as_a_signal(self, text):
        assert "SPY" not in parse(text).assets

    def test_the_rule_is_written_as_a_signal_test(self):
        """The first fix enumerated purchase verbs and missed "goes into",
        which the stability benchmark caught within one run."""
        import inspect

        from src.mission import compiler

        source = inspect.getsource(compiler.parse)
        assert "moving average" in source and "below" in source


class TestTheCheckWorks:

    def test_it_detects_a_value_that_does_not_travel(self, monkeypatch):
        """A check that cannot fail is decoration.

        Pointing a field at a destination it does not reach must be reported,
        because that is precisely the state the compiler was in three times.
        """
        monkeypatch.setitem(CARRIES, "dividends", "flows.cadence")
        parsed, result = compiled(
            "I put $500 into VTI monthly, holding the dividends as cash, "
            "and never sell.")
        gaps = representation_gaps(parsed, result.scenario)
        assert any(g.field == "dividends" for g in gaps)

    def test_presence_is_checked_against_the_canonical_form(self):
        """A value carried on the object but excluded from the canonical form
        is still invisible to identity, comparison and replay."""
        parsed, result = compiled(
            "I put $500 into VTI monthly, holding the dividends as cash, "
            "and never sell.")
        import json

        blob = json.dumps(result.scenario.canonical_form())
        assert "held_as_cash" in blob
        assert representation_gaps(parsed, result.scenario) == []

    def test_the_compiler_reports_gaps_in_its_own_verification(self):
        """The check runs on every compile, not only in a test."""
        _p, result = compiled("I put $500 into VTI monthly and never sell.")
        assert not any(v.startswith("unrepresented:") for v in result.verification)


class TestTheWholeCorpusIsRepresented:

    def test_no_recognised_value_is_dropped_across_14400_compiles(self):
        """The sweep that turns a class of defect into a measured zero."""
        from src.loadtest.catalog import load_strategies
        from src.loadtest.paraphrase import corpus

        gaps = []
        for prompt in corpus(load_strategies(), 100):
            parsed = parse(prompt.text)
            result = compile_scenario(prompt.text, name="s", version=1,
                                      benchmark_rule=BR, parsed=parsed)
            gaps.extend(representation_gaps(parsed, result.scenario))
        assert not gaps, "\n".join(str(g) for g in gaps[:10])
