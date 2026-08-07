"""Two readers disagreeing produces a question, not a choice.

Stage 1 has two independent readers: deterministic phrase rules, and a model
constrained to a closed vocabulary whose every proposal must quote a span that
is actually in the text. `merge()` compares them and emits a `Disagreement`.

That detector worked. Its output went nowhere.

    deterministic (pre-fix): persistent_condition
    model                  : crossing_event  (span: "crosses below")
    disagreements recorded : [(trigger_semantics, persistent, crossing)]
    value that executed    : persistent_condition

The model read the pilot's sentence correctly, the regex did not, the conflict
was detected, and the regex won because the policy was "the deterministic one
wins every contested field". The plan committed $60,000 instead of $13,000.
The information needed to prevent it was computed and discarded one line
later.

The policy is now:

    deterministic execution requires semantic agreement on all material
    execution fields

Agreement is the authority. Neither reader is. On a material field where they
differ, both readings are dropped, the field falls to `unresolved`, and the
clarification machinery that already exists asks the user — the same path a
phrase neither reader recognised takes.

**Not every field.** A gate that stopped a journey over a dividend-policy
quibble would be widened back out within a month, and then it would catch
nothing. `MATERIAL_EXECUTION_FIELDS` is the scope.
"""
from __future__ import annotations

import pytest

from src.mission.compiler import ParsedUtterance, Recognition
from src.mission.parse_model import MATERIAL_EXECUTION_FIELDS, merge

CONTROL = ("I buy $1,000 of SPY whenever it crosses below its 200-day moving "
           "average, over the past five years.")


def deterministic(field, value, span="a phrase"):
    return ParsedUtterance(text=CONTROL, assets=("SPY",), recognitions=(
        Recognition(field=field, value=value, span=span),))


def proposal(field, value, span="crosses below"):
    return (Recognition(field=field, value=value, span=span),)


def run(det, model):
    combined, _assets, _unrec, _unclear, disagreements, accepted = merge(
        det, model, ("SPY",), ())
    return {one.field: one.value for one in combined}, disagreements, accepted


class TestThePremise:
    def test_the_scope_is_not_everything(self):
        """A gate covering every field would stop journeys over cosmetic
        differences, and a gate that fires on noise gets removed."""
        from src.mission.parse_model import VOCABULARY

        assert MATERIAL_EXECUTION_FIELDS < set(VOCABULARY)
        assert "dividends" not in MATERIAL_EXECUTION_FIELDS

    def test_the_field_that_caused_the_defect_is_in_scope(self):
        assert "trigger_semantics" in MATERIAL_EXECUTION_FIELDS


class TestTheDefectThatMotivatedIt:
    """Replayed with the exact values production produced."""

    def test_the_contested_reading_no_longer_executes(self):
        values, disagreements, _ = run(
            deterministic("trigger_semantics", "persistent_condition",
                          "whenever it crosses below"),
            proposal("trigger_semantics", "crossing_event"))

        assert disagreements, "the conflict was not even detected"
        assert "trigger_semantics" not in values, (
            f"a contested execution field still resolved to "
            f"{values.get('trigger_semantics')!r}; this is the defect")

    def test_neither_reader_is_preferred(self):
        """Not 'the model wins' either. Swapping which reader is right must
        produce the same refusal — the point is disagreement, not authority."""
        values, _, _ = run(
            deterministic("trigger_semantics", "crossing_event"),
            proposal("trigger_semantics", "persistent_condition"))
        assert "trigger_semantics" not in values

    def test_the_disagreement_is_still_recorded(self):
        """Dropping the value must not drop the reason. An operator asking why
        a question appeared needs the two readings."""
        _, disagreements, _ = run(
            deterministic("trigger_semantics", "persistent_condition"),
            proposal("trigger_semantics", "crossing_event"))
        one = disagreements[0]
        assert (one.field, one.deterministic, one.model) == \
            ("trigger_semantics", "persistent_condition", "crossing_event")


class TestAgreementAndSilenceStillWork:
    """The gate must not fire on the ordinary cases, or it stops everything."""

    def test_agreement_passes_through(self):
        values, disagreements, _ = run(
            deterministic("trigger_semantics", "crossing_event"),
            proposal("trigger_semantics", "crossing_event"))
        assert values["trigger_semantics"] == "crossing_event"
        assert not disagreements

    def test_the_model_still_widens_recognition(self):
        """Its whole purpose: a phrasing the regexes miss. The deterministic
        reader proposes nothing, so there is nothing to disagree with."""
        values, disagreements, accepted = run(
            ParsedUtterance(text=CONTROL, assets=("SPY",)),
            proposal("trigger_semantics", "crossing_event"))
        assert values["trigger_semantics"] == "crossing_event"
        assert tuple(accepted) == ("trigger_semantics",)
        assert not disagreements

    def test_a_cosmetic_disagreement_does_not_block(self):
        """`dividends` is outside the scope. The deterministic reading stands
        and the conflict is recorded for anyone who wants it."""
        values, disagreements, _ = run(
            deterministic("dividends", "reinvested"),
            proposal("dividends", "held_as_cash"))
        assert values["dividends"] == "reinvested"
        assert disagreements

    def test_an_unrelated_field_is_untouched(self):
        det = ParsedUtterance(text=CONTROL, assets=("SPY",), recognitions=(
            Recognition("trigger_semantics", "persistent_condition", "x"),
            Recognition("account_type", "TAXABLE", "y")))
        values, _, _ = run(det, proposal("trigger_semantics", "crossing_event"))
        assert values["account_type"] == "TAXABLE"


@pytest.fixture
def deployment(monkeypatch):
    from src.deploy.context import bind, resolve, unbind

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
    try:
        yield
    finally:
        unbind()


class TestItReachesTheUserAsAQuestion:
    """Dropping the recognition is only correct if the compiler then asks.
    A field that silently defaulted instead would be worse than the defect."""

    def test_a_contested_trigger_becomes_an_open_question(self, deployment):
        from src.mission.compiler import compile_scenario
        from src.mission.parse_model import merge as real_merge

        det = deterministic("trigger_semantics", "persistent_condition",
                            "whenever it crosses below")
        combined, assets, unrec, unclear, _, _ = real_merge(
            det, proposal("trigger_semantics", "crossing_event"), ("SPY",), ())
        contested = ParsedUtterance(text=CONTROL, recognitions=combined,
                                    assets=assets, unrecognized=unrec,
                                    unclear=unclear)

        import src.workspace.routes as routes

        plan = compile_scenario(
            CONTROL, name="p", version=1, parsed=contested,
            benchmark_rule=routes.BENCHMARK_RULE,
            priceable=tuple(routes._market_data("gate").frame.columns))

        assert "trigger_semantics" in [one.field for one
                                       in plan.scenario.provenance.unresolved]

    def test_no_figure_is_produced_while_it_is_contested(self, deployment):
        """The honest outcome. A plan whose rule is undecided has no result to
        show, and showing one would be choosing a reading in the only place a
        user actually looks."""
        from src.mission.compiler import compile_scenario
        from src.mission.parse_model import merge as real_merge

        import src.workspace.routes as routes

        det = deterministic("trigger_semantics", "persistent_condition",
                            "whenever it crosses below")
        combined, assets, unrec, unclear, _, _ = real_merge(
            det, proposal("trigger_semantics", "crossing_event"), ("SPY",), ())
        plan = compile_scenario(
            CONTROL, name="p", version=1,
            parsed=ParsedUtterance(text=CONTROL, recognitions=combined,
                                   assets=assets, unrecognized=unrec,
                                   unclear=unclear),
            benchmark_rule=routes.BENCHMARK_RULE,
            priceable=tuple(routes._market_data("gate").frame.columns))
        assert not plan.scenario.event_program
