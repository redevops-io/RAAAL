"""The confirmation screen: what it shows, and what it refuses to decide.

The screen answers one question — **"did Quantify understand my plan?"** — and
everything else on it is subordinate to that, including the machinery that makes
the answer trustworthy.

Its shape follows what the corpus measured rather than what the architecture
finds interesting:

    69.3% of complete descriptions have nothing left to ask   -> one screen
    30.7% need one or two focused questions
     0.0% need three or more

So the majority must not be walked through a wizard built for the minority.
"""
from __future__ import annotations

import re

import pytest

from src.mission.compiler import compile_scenario
from src.workspace.confirmation import (
    ACCOUNT_CONTEXT,
    CHOICES,
    UNSUPPORTED_ACCOUNTS,
    build,
)

BR = "benchmark-policy/public-default@1"

COMPLETE = ("I put $2,000 into SPY every month in my Roth IRA, on the first "
            "trading day of the period, reinvesting the dividends, and I never "
            "sell.")
NEEDS_ACCOUNT = ("I put $2,000 into SPY every month, on the first trading day "
                 "of the period, reinvesting the dividends, and I never sell.")
CONTRADICTORY = ("I buy $2,000 of VTI and BND every month in my taxable "
                 "account, rebalancing them to equal weights, but I never sell.")


def view(text: str):
    return build(compile_scenario(text, name="p", version=1,
                                  benchmark_rule=BR), text=text)


class TestTheTwoPaths:
    """69.3% one screen, 30.7% one or two questions. Named, not inferred."""

    def test_a_complete_description_is_on_the_fast_path(self):
        v = view(COMPLETE)
        assert v.path == "FAST"
        assert v.question_count == 0

    def test_a_missing_account_asks_and_does_not_assume(self):
        v = view(NEEDS_ACCOUNT)
        assert v.path == "CLARIFY"
        assert [q.field for q in v.questions] == ["account_type"]

    def test_a_contradiction_blocks(self):
        v = view(CONTRADICTORY)
        assert v.path == "BLOCKED"
        assert v.conflicts
        assert v.headline == "These instructions conflict"

    def test_the_path_is_decided_here_not_in_the_template(self):
        """Otherwise the layout can disagree with the decision."""
        import inspect

        from src.workspace import confirmation

        assert "def path" in inspect.getsource(confirmation.ConfirmationView)


class TestTheSummaryLeads:
    """The first thing on the page answers the only question that matters."""

    def test_it_states_the_plan_in_plain_language(self):
        rows = {r["key"]: r["value"] for r in view(COMPLETE).summary}
        assert rows["holdings"] == "SPY"
        assert rows["amount"] == "$2,000"
        assert rows["cadence"] == "every month"
        assert rows["account"] == "Roth IRA"
        assert rows["dividends"] == "reinvested"

    def test_it_is_ordered_the_way_a_reader_checks_it(self):
        keys = [r["key"] for r in view(COMPLETE).summary]
        assert keys.index("holdings") < keys.index("amount") < keys.index("cadence")
        assert keys.index("cadence") < keys.index("account")

    def test_stated_details_are_a_count_not_a_checklist(self):
        """A long list of things the user already said makes confirming feel
        like filing a form."""
        v = view(COMPLETE)
        assert v.stated_count >= 4
        assert len(v.stated_detail) == v.stated_count


class TestInferencesAreThePrimaryTarget:

    def test_an_inference_carries_its_reason_and_its_options(self):
        v = view("I put $2,000 into SPY every month in my Roth IRA and never sell.")
        inference = next(i for i in v.inferences if i.field == "dividends")
        assert inference.why
        assert {c["value"] for c in inference.choices} == {"reinvested",
                                                           "held_as_cash"}

    def test_the_versioned_default_set_is_named(self):
        """The difference between "we guessed" and "a published, versioned
        default decided this, and here is its id"."""
        assert view(COMPLETE).defaults_ref.startswith("compiler-defaults/")


class TestQuestionsOfferTheirAnswers:

    def test_a_closed_vocabulary_becomes_options(self):
        """A free-text field where the answers are finite invites a phrasing
        the compiler then has to re-read, and re-reading is where meaning is
        lost."""
        question = next(q for q in view(NEEDS_ACCOUNT).questions
                        if q.field == "account_type")
        assert {c["value"] for c in question.choices} == set(ACCOUNT_CONTEXT)

    def test_every_offered_choice_is_a_value_the_compiler_reads(self):
        """An option the compiler cannot read back is a dead end."""
        from src.mission.compiler import _RULES

        vocabulary = {}
        for field, value, _pattern in _RULES:
            vocabulary.setdefault(field, set()).add(value)
        for field, choices in CHOICES.items():
            if field not in vocabulary:
                continue
            offered = {c["value"] for c in choices}
            assert offered <= vocabulary[field], f"{field}: {offered}"

    def test_a_question_states_its_consequence(self):
        for question in view(NEEDS_ACCOUNT).questions:
            assert len(question.why_it_matters) > 30, question.field

    def test_an_unmodellable_account_is_routed_not_refused(self):
        """A donor-advised fund is a real thing a user has, and "not supported"
        with no next step is a dead end."""
        v = view("I put $2,000 into SPY every month in my inherited IRA account.")
        question = next(q for q in v.questions if q.field == "account_type")
        assert question.routing
        assert "distribution schedule" in question.routing

    @pytest.mark.parametrize("phrase", sorted(UNSUPPORTED_ACCOUNTS))
    def test_every_unsupported_account_has_a_next_step(self, phrase):
        assert len(UNSUPPORTED_ACCOUNTS[phrase]) > 30


class TestTheAccountCard:
    """Account type decides tax treatment, contribution limits, withdrawal
    constraints and comparability, so it gets more than a one-line card."""

    def test_it_says_what_is_modelled_and_what_is_not(self):
        card = view(COMPLETE).account
        assert card["label"] == "Roth IRA"
        assert any("not taxed" in item for item in card["modelled"])
        assert any("state-specific" in item for item in card["not_modelled"])

    @pytest.mark.parametrize("account", sorted(ACCOUNT_CONTEXT))
    def test_every_account_declares_both_sides(self, account):
        entry = ACCOUNT_CONTEXT[account]
        assert entry["modelled"] and entry["not_modelled"], account

    def test_an_unnamed_account_has_no_card(self):
        assert view(NEEDS_ACCOUNT).account is None


class TestUnsimulatedChoicesAppearBeforeTheRun:

    def test_a_declared_but_unsimulated_choice_is_shown_up_front(self):
        """Not buried in the modelling scope afterwards. A choice the engine
        cannot honour is something to know while there is still a decision to
        make about running it."""
        v = view("I put $2,000 into SPY every month in my Roth IRA, holding the "
                 "dividends as cash, and I never sell.")
        entry = next(n for n in v.not_simulated if n.field == "dividend_policy")
        assert entry.declared == "held_as_cash"
        assert "price series only" in entry.why


class TestTheTemplateDecidesNothing:
    """The rule that caught a real defect on the library pages: a screen that
    recalculates what the compiler already decided is a second implementation,
    and the copy in the template is the one that drifts."""

    TEMPLATE = "src/workspace/templates/_confirmation.html"

    def test_it_contains_no_comparisons_or_arithmetic(self):
        body = open(self.TEMPLATE).read()
        # Jinja conditionals over prepared booleans and loops are fine; tests
        # of *values* are not.
        for pattern in (r"\{\%[^%]*[<>]=?\s*\d", r"\{\%[^%]*\bnot in\b",
                        r"\{\{[^}]*[-+*/]\s*\d"):
            assert not re.search(pattern, body), pattern

    def test_it_reads_only_prepared_fields(self):
        body = open(self.TEMPLATE).read()
        referenced = set(re.findall(r"view\.([a-z_]+)", body))
        prepared = set(view(COMPLETE).__dict__) | {"path", "question_count"}
        assert referenced <= prepared, referenced - prepared

    def test_the_headline_is_chosen_in_python(self):
        body = open(self.TEMPLATE).read()
        assert "{{ view.headline }}" in body
        assert "conflict" not in body.split("<h2>")[1].split("</h2>")[0]
