"""Phase 4's first property: a plan compiled from intent, never from prose.

This is the module that lets the legacy reader be deleted. While the only route
to a `ScenarioSpecification` runs through `compile_scenario(text, ...)`, the
regex compiler is load-bearing however the intent was produced, and "Discovery
is authoritative" describes the top of a pipeline whose bottom still parses
sentences.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from runtime_contracts import Author, IntentField, NotSealable, Unresolved, OpenReason, VerifiedIntent
from src.mission.from_intent import (
    COMPILER_VERSION,
    DEFAULTS,
    NotExecutable,
    compile_intent,
)

RULE = "benchmark-policy/public-default@1"


def intent(**fields) -> VerifiedIntent:
    base = {"assets": "SPY", "amount": "1000", "cadence": "monthly"}
    base.update(fields)
    return VerifiedIntent(
        objective="evaluate_investment_strategy",
        produced_by="discovery-runtime@test",
        fields={k: IntentField(value=v, author=Author.USER)
                for k, v in base.items() if v is not None}).seal()


class TestItNeverConsultsTheSentence:
    def test_the_module_does_not_import_the_compiler(self):
        """A convenience import added later would restore the dependency
        silently, and the deletion would then fail in a way nobody predicted."""
        source = Path("src/mission/from_intent.py").read_text()
        # Code only. The module docstring names `compile_scenario` while
        # explaining why this module exists, and a scan that reads its own
        # explanation as a violation is the same false positive the oracle
        # check already produced once.
        code = source.split('"""', 2)[-1]
        for forbidden in ("from .compiler", "import compiler",
                          "compile_scenario(", "parse("):
            assert forbidden not in code, f"{forbidden!r} reintroduces prose"

    def test_the_entry_point_takes_no_text(self):
        import inspect

        signature = inspect.signature(compile_intent)
        assert "text" not in signature.parameters
        assert "utterance" not in signature.parameters

    def test_a_plan_compiles_from_fields_alone(self):
        out = compile_intent(intent(), benchmark_rule=RULE)
        assert out.executable
        assert out.scenario.allocation_rule.assets == ("SPY",)


class TestIdenticalIntentGivesIdenticalPlan:
    """Phase 4's first acceptance property. A compiler that produced two plans
    from one intent would make replay meaningless before replay was reached."""

    def test_the_same_intent_compiles_the_same_way(self):
        one = compile_intent(intent(), benchmark_rule=RULE)
        other = compile_intent(intent(), benchmark_rule=RULE)
        assert one.scenario.content_hash == other.scenario.content_hash

    def test_a_different_intent_does_not(self):
        """The discriminating half."""
        a = compile_intent(intent(cadence="monthly"), benchmark_rule=RULE)
        b = compile_intent(intent(cadence="annual"), benchmark_rule=RULE)
        assert a.scenario.content_hash != b.scenario.content_hash

    def test_the_plan_names_the_intent_it_came_from(self):
        source = intent()
        out = compile_intent(source, benchmark_rule=RULE)
        assert out.derivation["compiled_from"] == source.intent_hash
        assert out.derivation["compiled_by"] == COMPILER_VERSION
        assert out.derivation["intent_produced_by"] == "discovery-runtime@test"


class TestItRefusesRatherThanAdjusting:
    def test_an_unexecutable_value_is_refused_by_name(self):
        out = compile_intent(intent(allocation_method="inverse_volatility"),
                             benchmark_rule=RULE)
        assert not out.executable and out.scenario is None
        assert "allocation_method" in {r.dimension for r in out.refusals}

    def test_no_partial_plan_accompanies_a_refusal(self):
        """A plan beside a refusal is a plan a caller renders anyway, and then
        a figure exists for a request that was refused."""
        out = compile_intent(intent(periodic_rebalancing="quarterly"),
                             benchmark_rule=RULE)
        assert out.refusals and out.scenario is None

    def test_an_intent_holding_nothing_is_refused(self):
        """The first version compiled this happily — an empty allocation and a
        trigger with no subject. Nothing downstream would have priced it, but
        the failure would have read as missing data rather than as an intent
        that never said what to buy."""
        out = compile_intent(intent(assets=None), benchmark_rule=RULE)
        assert "assets" in {r.dimension for r in out.refusals}

    def test_a_draft_is_refused_outright(self):
        draft = VerifiedIntent(objective="o", fields={
            "assets": IntentField("SPY", Author.USER)})
        with pytest.raises(NotExecutable) as raised:
            compile_intent(draft)
        assert "draft" in str(raised.value)

    def test_an_unresolved_disagreement_is_refused(self):
        blocked = replace(intent(), unresolved=(Unresolved(
            "trigger_semantics", OpenReason.UNRESOLVED_DISAGREEMENT),))
        out = compile_intent(blocked, benchmark_rule=RULE)
        assert "trigger_semantics" in {r.dimension for r in out.refusals}


class TestSilenceIsAnAppliedDefaultAndSaysSo:
    def test_defaults_are_reported_not_hidden(self):
        """A plan is only reproducible if a reader can see which values nobody
        asked for. This is the `execution_timing` defect made visible."""
        out = compile_intent(intent(), benchmark_rule=RULE)
        assert "day_rule" in out.applied_defaults
        assert out.scenario.funding.day_rule == DEFAULTS["day_rule"]

    def test_a_stated_value_is_not_reported_as_a_default(self):
        out = compile_intent(intent(day_rule="last_session_of_period"),
                             benchmark_rule=RULE)
        assert "day_rule" not in out.applied_defaults
        assert out.scenario.funding.day_rule == "last_session_of_period"

    def test_every_default_is_for_a_dimension_the_engine_executes(self):
        """Found by this file: the table also carried `dividend_policy` and
        `tax_treatment`, which the manifest refuses and does not model. The
        compiler was supplying a value nothing would act on and reporting it as
        an applied default, which is declared-but-not-executed inside the
        module written to prevent it."""
        from src.mission.capability import EXECUTED, MANIFEST

        for dimension in DEFAULTS:
            entry = MANIFEST.get(dimension)
            if entry is None:
                continue
            assert entry.support == EXECUTED, (
                f"{dimension} has a default and is {entry.support}")

    def test_engine_constants_are_not_reported_as_choices(self):
        """Nobody left them open; they are not choices."""
        from src.mission.from_intent import ENGINE_CONSTANTS

        out = compile_intent(intent(), benchmark_rule=RULE)
        assert not set(ENGINE_CONSTANTS) & set(out.applied_defaults)


class TestTheAssetsAreTheUsersWords:
    def test_a_description_is_not_resolved_to_a_ticker(self):
        """"a core index fund" stays that. The engine refusing to price it is
        the correct failure; choosing VTI is the substitution this boundary
        exists to prevent."""
        out = compile_intent(intent(assets="a core index fund"),
                             benchmark_rule=RULE)
        assert out.scenario.allocation_rule.assets == ("a core index fund",)


class TestFundingAndItsProjectionCannotDisagree:
    def test_a_scheduled_plan_projects_its_own_cadence(self):
        out = compile_intent(intent(cadence="annual"), benchmark_rule=RULE)
        assert out.scenario.funding.cadence == "annual"
        assert out.scenario.flow_schedule.cadence == "annual"

    def test_an_event_plan_declares_no_calendar(self):
        """`funding` is the authority and `flow_schedule` is its projection."""
        out = compile_intent(intent(trigger_semantics="crossing_event",
                                    cadence=None), benchmark_rule=RULE)
        assert out.scenario.flow_schedule.cadence == "event_triggered"
        assert out.scenario.flow_schedule.amount == 0.0

    def test_a_calendar_stated_beside_a_trigger_is_refused_not_dropped(self):
        """The second silent reduction, found by the general stranded-dimension
        check rather than by looking for it.

        This case used to compile: `cadence="monthly"` alongside a crossing
        trigger produced an `EventTriggered` schedule whose `cadence` read
        `"event_triggered"`, and the stated *monthly* went nowhere. Nothing in
        the result said a word of the request had been discarded, so the person
        who asked to contribute monthly *and* on a crossing was shown a plan
        that did one of those things and told it was their plan.

        The event path never consults `cadence`, which is defensible — a
        trigger and a calendar are two different authorities on when money
        moves, and this build has no representation for both at once. What is
        not defensible is deciding that silently. Refusing by name leaves the
        person able to drop one of the two and get what they asked for.
        """
        out = compile_intent(intent(trigger_semantics="crossing_event"),
                             benchmark_rule=RULE)
        assert out.scenario is None
        assert [r.dimension for r in out.refusals] == ["cadence"]
        assert out.refusals[0].kind == "UNSUPPORTED_DIMENSION"


class TestAStatedNumberIsNeverQuietlyDefaulted:
    """`_decimal(...) or <default>` conflated "not stated" with "stated and
    unreadable". Both call sites did it, so this is a property of the module
    rather than a bug in one line."""

    def test_an_unreadable_amount_is_refused_rather_than_zeroed(self):
        """The finding, in the form that makes it serious.

        "invest $1k monthly into VTI" compiled. Asset right, cadence right, day
        rule right — and `amount = 0`. The plan was indistinguishable from the
        one asked for except that it invested nothing, and nothing in the
        result said so. A backtest of it would have reported a portfolio that
        never grew, and the honest-looking explanation would have been that the
        market did badly.
        """
        out = compile_intent(intent(amount="$1k"), benchmark_rule=RULE)
        assert out.scenario is None
        assert [r.dimension for r in out.refusals] == ["amount"]
        assert "$1k" in out.refusals[0].detail

    def test_a_recurring_plan_with_no_amount_is_asked_about(self):
        """This assertion previously ran the other way, and it was wrong.

        It read: nobody stated a figure, so there is nothing to fail to read,
        and a plan with no contributions is a plan. The harvested corpus showed
        what that permits. "putting a portion of my cash savings into I-Bonds
        every year" was the *only* one of 29 attested strategy statements to
        reach a plan, and the plan held I-Bonds on an annual cadence and
        contributed zero — no question, and `amount` not even reported as an
        applied default.

        The incoherence needs no reading of the sentence: the cadence says
        money moves every year and the amount says none does.
        """
        out = compile_intent(intent(amount=None), benchmark_rule=RULE)
        assert out.scenario is None
        assert [r.dimension for r in out.refusals] == ["amount"]
        assert out.refusals[0].kind == "UNRESOLVED_INPUT"

    def test_an_explicit_zero_is_an_instruction_and_is_honoured(self):
        """Zero is a statement, not a gap.

            missing material quantity   -> unresolved
            explicitly zero quantity    -> zero

        The first version of the recurring check refused both, which made the
        runtime reject something the person had actually said in order to
        prevent something they had not. Somebody modelling "what if I stop
        contributing" is asking a real question and this is how they ask it.
        """
        out = compile_intent(intent(amount="0"), benchmark_rule=RULE)
        assert out.scenario is not None, [r.detail for r in out.refusals]
        assert out.scenario.flow_schedule.amount == 0.0
        assert out.scenario.flow_schedule.cadence == "monthly"

    def test_and_so_is_a_currency_formatted_zero(self):
        """Through the layer that reads notation, which is no longer this one.

        Mission used to strip `$`, `,` and currency words itself. That is a
        reading question and `discovery.canonical` answers it; what arrives
        here is a plain decimal. The property is unchanged — a stated zero is a
        stated zero — and only the door it comes through has moved.
        """
        from src.discovery.canonical import canonicalise

        canonical = canonicalise({"amount": "$0"}).fields["amount"][0]
        out = compile_intent(intent(amount=canonical), benchmark_rule=RULE)
        assert out.scenario is not None
        assert out.scenario.flow_schedule.amount == 0.0

    def test_a_one_off_plan_with_no_amount_still_compiles(self):
        """The exception, and a real one: a plan may model opening capital with
        no contributions after it. `once` is not a claim that money moves
        repeatedly, so there is nothing for a zero amount to contradict."""
        out = compile_intent(intent(amount=None, cadence="once"),
                             benchmark_rule=RULE)
        assert out.scenario is not None
        assert out.scenario.flow_schedule.amount == 0.0

    def test_every_numeric_dimension_is_listed(self):
        """Structural, not a source grep for prose.

        Walks the AST for `_decimal(...)` calls and recovers the dimension each
        one reads. A numeric dimension added later and left out of `NUMERIC`
        would default silently, which is the whole class this closes — so the
        omission has to fail here rather than wait to be noticed.
        """
        import ast

        from src.mission.from_intent import NUMERIC

        tree = ast.parse(Path("src/mission/from_intent.py").read_text())
        read = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_decimal" and node.args):
                continue
            inner = node.args[0]
            # `_decimal(value("amount"))` — the dimension is the literal.
            if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                    and inner.func.id == "value" and inner.args
                    and isinstance(inner.args[0], ast.Constant)):
                read.add(inner.args[0].value)

        assert read, "no `_decimal(value(...))` call sites found; this check " \
                     "would pass by finding nothing"
        missing = sorted(read - set(NUMERIC))
        assert not missing, (
            f"{missing} are read as numbers and are not in NUMERIC, so an "
            "unreadable value for them falls through to a default silently")


class TestTheSetMemberRuleLivesInOnePlaceNow:
    """It used to live twice, and only one copy knew about `and`.

    Mission may not import Discovery, so the separator was duplicated on both
    sides and the copies were not the same: "split equally between VTI and BND"
    compiled to one instrument called `"VTI and BND"`, weighted at 100%, while
    fusion had agreed the sentence named two assets. `canonical_form` sorts the
    assets, so the sort ran over a one-element list and reported nothing wrong.

    The rule is now Discovery's alone, and what reaches Mission is a
    comma-separated list. So the thing to protect changed shape: not "do the two
    copies agree" but "is there still only one copy".
    """

    def test_mission_has_no_separator_of_its_own(self):
        import src.mission.from_intent as compiler

        assert not hasattr(compiler, "SET_SEPARATOR"), (
            "Mission has a set-separator again, which means it is splitting "
            "prose — the duplication that produced a portfolio of one")

    def test_canonicalisation_splits_the_way_fusion_compares(self):
        """The two rules that must still agree, both inside Discovery."""
        from src.discovery.adapter import same_value_for
        from src.discovery.canonical import canonicalise

        assert same_value_for("assets", "VTI and BND", "BND, VTI"), (
            "fusion no longer treats `and` as a member separator")
        assert canonicalise({"assets": "VTI and BND"}).fields["assets"][0] \
            == "VTI,BND"

    def test_two_holdings_named_with_and_are_two_holdings(self):
        from src.discovery.canonical import canonicalise

        canonical = canonicalise({"assets": "VTI and BND"}).fields["assets"][0]
        out = compile_intent(intent(assets=canonical), benchmark_rule=RULE)
        assert out.scenario.allocation_rule.assets == ("VTI", "BND")

    def test_and_naming_them_in_either_order_is_the_same_plan(self):
        """The property the replay guarantee rests on: execution identity is
        canonical executable semantics, not the order somebody typed."""
        import json
        from hashlib import sha256

        from src.discovery.canonical import canonicalise

        def digest(assets):
            canonical = canonicalise({"assets": assets}).fields["assets"][0]
            out = compile_intent(intent(assets=canonical), benchmark_rule=RULE)
            return sha256(json.dumps(out.scenario.canonical_form(),
                                     sort_keys=True,
                                     default=str).encode()).hexdigest()

        assert digest("VTI and BND") == digest("BND and VTI")


class TestExecutionIdentityIsNotOrthography:
    """`VerifiedIntent` keeps what was said and `canonical_form()` keeps what
    the plan holds. Plan identity is a third thing, and the live drift lane
    found the last place it was still coupled to spelling."""

    def _digest(self, assets):
        import json
        from hashlib import sha256

        out = compile_intent(intent(assets=assets), benchmark_rule=RULE)
        return sha256(json.dumps(out.scenario.execution_form(), sort_keys=True,
                                 default=str).encode()).hexdigest()

    def test_a_leading_article_does_not_change_the_plan(self):
        """The blocker, measured in CI: a serving reader returned
        `"the index fund"` on four draws of one sentence and `"index fund"` on
        the fifth, and the two compiled to different plan digests. Same money,
        same instrument, two identities — which the gate correctly called
        `UNSTABLE_EXECUTABLE`, because by its definition a draw had changed
        what executes."""
        assert self._digest("the index fund") == self._digest("index fund")

    def test_a_resolved_phrase_collapses_to_the_registry_subject(self):
        """`S&P 500` and `the S&P` are one subject because the resolver says
        so, not because a rule here strips words."""
        assert self._digest("S&P 500") == self._digest("the S&P")

    def test_but_different_holdings_are_still_different_plans(self):
        """The discriminating half. A canonicaliser that made everything equal
        would pass the tests above and destroy the property they exist for."""
        assert self._digest("VTI") != self._digest("BND")

    def test_and_the_plan_still_holds_what_was_written(self):
        """The seam this was first implemented in by mistake.
        `canonical_form()` is consumed as data — `representation.py` and
        `to_json()` read it — so canonicalising there turned `SPY` into `spy`,
        the engine found no prices, and sixteen tests failed on absent
        signals."""
        out = compile_intent(intent(assets="SPY"), benchmark_rule=RULE)
        held = out.scenario.canonical_form()["methodology"]["allocation_rule"]
        assert held["assets"] == ["SPY"]
