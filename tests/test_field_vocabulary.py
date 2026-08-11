"""One enumeration, and it is also the validation.

Three things were separate and disagreed. The compiler decided which questions
to ask; `confirmation.CHOICES` decided which options to show, and had already
drifted — `cadence` and `execution_timing` were asked with no options at all;
and nothing decided which answers were *acceptable*.

That third gap is the one that mattered. `cadence=banana` removed the question
and recorded "cadence: banana (answered)" as a stated fact, so a saved plan
could carry a cadence the renderer has no word for.
"""
from __future__ import annotations

import pytest

from src.mission import vocabulary
from src.mission.compiler import compile_scenario
from src.mission.spec import ScenarioAmendment

RAISES_QUESTIONS = "I buy $500 of VOO."


def compiled(field, answer, text=RAISES_QUESTIONS):
    amendments = (ScenarioAmendment(question_id=field, answer=answer,
                                    recorded_at="t"),)
    return compile_scenario(text, amendments=amendments).scenario


def still_asked(scenario, field):
    return field in {one.field for one in scenario.provenance.unresolved}


class TestTheVocabularyIsTheValidation:
    def test_a_value_outside_the_set_does_not_settle_the_question(self):
        scenario = compiled("cadence", "banana")
        assert still_asked(scenario, "cadence"), (
            "a cadence nothing understands settled the question")

    def test_and_is_not_recorded_as_a_stated_fact(self):
        """The worse half. The question disappearing is visible; a nonsense
        value stored as something the user stated is not."""
        scenario = compiled("cadence", "banana")
        assert not [one for one in scenario.provenance.stated
                    if "banana" in str(one)]

    def test_a_value_inside_the_set_settles_it(self):
        """The premise. If nothing settled, the assertions above would hold
        for a compiler that ignored answers entirely."""
        scenario = compiled("cadence", "monthly")
        assert not still_asked(scenario, "cadence")
        assert [one for one in scenario.provenance.stated
                if "cadence: monthly" in str(one)]

    @pytest.mark.parametrize("field,bad", [
        ("account_type", "Mars colony"),
        ("dividends", "set them on fire"),
        ("cadence", "annually"),  # reads naturally, is not the engine's word
    ])
    def test_each_closed_field_refuses_a_stranger(self, field, bad):
        assert not vocabulary.accepts(field, bad)

    def test_an_amount_is_not_enumerated(self):
        """Money has no option list, and validating it against one would
        refuse every number."""
        assert vocabulary.accepts("amount", "500")
        assert vocabulary.accepts("amount", "1234.56")

    def test_an_unknown_field_is_not_refused(self):
        """Failing closed on *unrecognised* is a different decision from
        failing closed on *wrong* — it would make each new question
        unanswerable until somebody remembered this file."""
        assert vocabulary.accepts("some_future_field", "anything")

    def test_a_prefixed_field_has_no_vocabulary(self):
        assert vocabulary.field_for("unclear:x") is None
        assert vocabulary.field_for("asset_identity:SPX") is None


class TestEveryOfferedValueIsAccepted:
    """The page must not offer a value the compiler will throw away.

    Derived from the registry rather than listed here, so a value added to one
    and not the other cannot pass.
    """

    @pytest.mark.parametrize("name", sorted(
        n for n, f in vocabulary.FIELDS.items() if f.options))
    def test_the_field_accepts_all_of_its_own_options(self, name):
        field = vocabulary.FIELDS[name]
        for option in field.options:
            assert field.accepts(option.value), (
                f"{name} offers {option.value} and refuses it")

    def test_the_page_offers_exactly_the_registry(self):
        """`confirmation.CHOICES` is derived, not restated. It was a second
        hand-written copy and had already fallen out of step."""
        from src.workspace.confirmation import CHOICES

        for name, field in vocabulary.FIELDS.items():
            if not field.options:
                assert name not in CHOICES
                continue
            offered = [one["value"] for one in CHOICES[name]]
            assert offered == [one.value for one in field.options]


class TestTheGapsThatStartedThis:
    @pytest.mark.parametrize("name", ["cadence", "execution_timing"])
    def test_the_previously_optionless_fields_have_options(self, name):
        assert vocabulary.FIELDS[name].options, (
            f"{name} is asked and still offers nothing to choose")

    def test_cadence_values_are_the_renderers(self):
        """A dropdown offering a word the renderer cannot use would settle a
        question and then produce a sentence with a raw token in it."""
        from src.mission.render import _CADENCE_WORDS

        for option in vocabulary.FIELDS["cadence"].options:
            assert option.value in _CADENCE_WORDS, (
                f"{option.value} is offered and has no rendering")


class TestNoFieldFallsBackToAGenericBox:
    """Every field the compiler can leave unresolved is accounted for.

    Two outcomes and no third: a registry entry that backs a real control, or
    a named exemption for something a deployment settles. Without this, adding
    a question to the compiler silently produced a free-text box that the
    validation then had nothing to check against — which is how `cadence` and
    `execution_timing` came to be asked with no options at all.
    """

    def compiler_fields(self):
        """Derived from the compiler's own call sites, not restated here."""
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path("src/mission/compiler.py").read_text())
        found = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)):
                continue
            if (node.func.id in ("settle", "answered") and node.args
                    and isinstance(node.args[0], ast.Constant)):
                found.add(node.args[0].value)
            if node.func.id == "Unresolved":
                if node.args and isinstance(node.args[0], ast.Constant):
                    found.add(node.args[0].value)
                for keyword in node.keywords:
                    if (keyword.arg == "field"
                            and isinstance(keyword.value, ast.Constant)):
                        found.add(keyword.value.value)
        return found

    def test_the_derivation_finds_something(self):
        """A silent zero here would make every assertion below vacuous."""
        assert len(self.compiler_fields()) >= 10

    def test_every_field_is_backed_or_exempt(self):
        unaccounted = sorted(
            name for name in self.compiler_fields()
            if name not in vocabulary.FIELDS
            and name not in vocabulary.POLICY_SETTLED)
        assert not unaccounted, (
            f"{unaccounted} would render as a generic text box with nothing "
            f"to validate against")

    def test_a_closed_field_has_options(self):
        """A CHOICE entry with no options is a dropdown with nothing in it."""
        for name, field in vocabulary.FIELDS.items():
            if field.kind is vocabulary.FieldKind.CHOICE:
                assert field.options, f"{name} is a choice with no choices"

    def test_the_exemption_list_is_not_a_dumping_ground(self):
        """An exemption is a claim that a deployment settles the field. If it
        also has a registry entry, one of the two is wrong."""
        for name in vocabulary.POLICY_SETTLED:
            assert name not in vocabulary.FIELDS
