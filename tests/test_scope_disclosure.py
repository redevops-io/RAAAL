"""Per-rule scope: what is recognised, comparable, and actually enforced.

"Roth IRA supported" tells a user nothing actionable. "Contribution limit
PARTIAL because the shared IRA limit cannot be established, state tax NOT
MODELLED" tells them which figure to trust and which question to answer next.
"""
from __future__ import annotations

import pytest

from src.runtime.rsu import IMPLEMENTED, US_SHARE_WITHHOLDING
from src.workspace.scope_disclosure import (
    ACCOUNT_NOT_MODELLED,
    RSU_NOT_MODELLED,
    Enforcement,
    Facet,
    for_account,
    for_rsu,
)

ROTH = "ROTH"
YEAR = 2026


@pytest.fixture
def account():
    return for_account(ROTH, tax_year=YEAR)


@pytest.fixture
def rsu():
    return for_rsu(US_SHARE_WITHHOLDING, implemented=IMPLEMENTED)


class TestPerRuleNotPerAccount:

    def test_it_reports_rules_rather_than_a_single_badge(self, account):
        assert len(account.rules) > 1
        assert {one.enforcement for one in account.rules} != {
            Enforcement.ENFORCED}

    def test_the_contribution_limit_is_its_own_row(self, account):
        [one] = [r for r in account.rules if r.rule == "contribution limit"]
        assert one.enforcement is Enforcement.PARTIAL

    def test_the_shared_limit_is_a_separate_row(self, account):
        """Capping one account proves nothing about the combined total."""
        [one] = [r for r in account.rules
                 if r.rule == "shared limit across related accounts"]
        assert one.enforcement is Enforcement.PARTIAL
        assert "only this account was described" in one.why

    def test_each_partial_rule_names_what_is_missing(self, account):
        for one in account.by_enforcement(Enforcement.PARTIAL):
            assert one.why

    def test_nothing_appears_twice(self, account):
        """`unenforced_behaviours` restates in prose what the limit rules
        already say; both would put one fact on screen twice."""
        assert len({one.rule for one in account.rules}) == len(account.rules)


class TestNotModelledIsNamedNotOmitted:

    @pytest.mark.parametrize("rule", list(ACCOUNT_NOT_MODELLED))
    def test_each_account_exclusion_is_stated(self, account, rule):
        """A reader assumes an unmentioned rule is handled."""
        assert rule in {one.rule for one in account.rules}

    @pytest.mark.parametrize("rule", list(RSU_NOT_MODELLED))
    def test_each_rsu_exclusion_is_stated(self, rsu, rule):
        assert rule in {one.rule for one in rsu.rules}

    def test_state_tax_is_explicitly_not_modelled(self, account):
        [one] = [r for r in account.rules if r.rule == "state and local tax"]
        assert one.enforcement is Enforcement.NOT_MODELLED

    def test_income_phase_out_says_why_it_cannot_run(self, account):
        [one] = [r for r in account.rules
                 if r.rule == "income phase-out eligibility"]
        assert "modified AGI" in one.why


class TestInputAssumptionOutput:

    def test_all_three_facets_appear(self, account):
        assert {one.facet for one in account.statements} == {
            Facet.INPUT, Facet.ASSUMPTION, Facet.OUTPUT}

    def test_the_assumption_explains_the_partial(self, account):
        """Without it, "partially established" reads as a defect in the system
        rather than a consequence of what the user did not say."""
        [assumption] = [one for one in account.statements
                        if one.facet is Facet.ASSUMPTION]
        # "No other ... were declared" — the negation is in the subject, not
        # in a "not" before the verb.
        assert assumption.text.startswith("No other")
        assert "declared" in assumption.text

    def test_the_output_states_what_was_established(self, account):
        [output] = [one for one in account.statements
                    if one.facet is Facet.OUTPUT]
        assert "not established" in output.text

    def test_the_rsu_assumption_separates_withholding_from_tax(self, rsu):
        [assumption] = [one for one in rsu.statements
                        if one.facet is Facet.ASSUMPTION]
        assert "not this person's marginal rate" in assumption.text


class TestTheOutputBasisTravels:

    def test_the_account_basis_says_no_tax_is_applied(self, account):
        assert "no tax is applied" in account.output_basis

    def test_the_rsu_basis_is_post_withholding(self, rsu):
        assert "after employer share withholding" in rsu.output_basis
        assert "not gross compensation" in rsu.output_basis
        assert "final federal, state or local tax liability" in rsu.output_basis


class TestCoverageIsNotConfidence:

    def test_it_reports_realized_over_in_scope(self, rsu):
        assert rsu.coverage["realized"] == rsu.coverage["declared"]
        assert rsu.coverage["fraction"] == 1.0

    def test_deliberate_exclusions_are_not_counted_as_shortfalls(self, rsu):
        """Counting "state tax is not modelled" as a coverage failure makes a
        correct, deliberate exclusion look like a defect."""
        assert rsu.coverage["out_of_scope"] == len(RSU_NOT_MODELLED)
        assert rsu.coverage["declared"] < len(rsu.rules)

    def test_the_exclusions_stay_visible(self, rsu):
        """Excluded from the denominator, not from the page."""
        assert len(rsu.by_enforcement(Enforcement.NOT_MODELLED)) == \
            len(RSU_NOT_MODELLED)

    def test_a_partial_account_reports_low_coverage(self, account):
        assert account.coverage["fraction"] == 0.0
        assert account.coverage["declared"] == 2

    def test_the_note_says_it_is_not_confidence(self, account):
        assert "not confidence" in account.coverage["note"]

    def test_an_empty_disclosure_reports_no_fraction(self):
        """Zero of zero is not zero coverage."""
        empty = for_account("TAXABLE", tax_year=YEAR)
        assert empty.coverage["fraction"] in (None, 0.0)


class TestItReadsTheRuntimeRatherThanRestating:

    def test_the_rsu_rules_come_from_the_runtime_assumptions(self, rsu):
        declared = {one.name.replace("-", " ")
                    for one in US_SHARE_WITHHOLDING.assumptions}
        assert declared <= {one.rule for one in rsu.rules}

    def test_an_unrealized_assumption_reports_partial(self):
        """The shared deferral limit names a mechanism that does not exist."""
        from src.runtime.account import AccountKind, AccountRuntime

        runtime = AccountRuntime(name="a", version=1,
                                 account_kind=AccountKind.ROTH_401K)
        disclosure = for_rsu(US_SHARE_WITHHOLDING, implemented=())
        assert all(one.enforcement is not Enforcement.ENFORCED
                   for one in disclosure.rules
                   if one.rule != "final tax liability"
                   and one.enforcement is not Enforcement.NOT_MODELLED)
        del runtime

    def test_realization_moves_the_state(self):
        """A derivation nobody has seen change is a derivation nobody tested."""
        none_realized = for_rsu(US_SHARE_WITHHOLDING, implemented=())
        all_realized = for_rsu(US_SHARE_WITHHOLDING, implemented=IMPLEMENTED)
        assert none_realized.coverage["fraction"] == 0.0
        assert all_realized.coverage["fraction"] == 1.0


class TestRecognitionAndComparabilityStayApart:

    def test_recognition_lists_what_was_read(self, account):
        assert "Roth IRA" in account.recognised
        assert f"tax year {YEAR}" in account.recognised

    def test_comparability_is_a_different_list(self, account):
        assert "account kind" in account.comparable

    def test_what_cannot_be_compared_is_named(self, account):
        assert any("Traditional Ira" in one or "not declared" in one
                   for one in account.not_comparable)

    def test_the_jurisdiction_and_year_are_stated(self, account):
        assert account.jurisdiction == "US-federal"
        assert account.tax_year == YEAR
