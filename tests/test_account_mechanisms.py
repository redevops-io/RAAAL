"""The account mechanisms themselves: `cap_contribution` and `apply_match`.

The rulesets they read are covered in tests/test_account_rules.py. This file is
about the registry that claims these mechanisms exist, and about what they do
once they run.
"""
from __future__ import annotations

import pytest

from src.runtime import ACCOUNT_IMPLEMENTED, AccountKind, AccountRuntime
from src.runtime.account_limits import load


def runtime(kind, **kw):
    return AccountRuntime(name="a", version=1, account_kind=kind, **kw)


class TestTheRegistryCannotClaimAMechanismIntoExistence:
    """`ACCOUNT_IMPLEMENTED` is a hand-written tuple. Adding a string to it
    moves every account display toward ENFORCED, so the strings must resolve to
    real code."""

    def test_every_implemented_name_is_a_real_callable(self):
        for name in ACCOUNT_IMPLEMENTED:
            attribute = getattr(AccountRuntime, name, None)
            assert callable(attribute), (
                f"ACCOUNT_IMPLEMENTED names {name!r}, which is not a method on "
                "AccountRuntime. A realization registry that names absent "
                "mechanisms reports unenforced behaviour as enforced")

    def test_a_declared_mechanism_that_is_absent_is_reported_unrealized(self):
        account = runtime(AccountKind.ROTH_401K, annual_contribution_limit=7000.0,
                          employer_match_rate=0.5)
        for assumption in account.assumptions:
            if assumption.realized_by not in ACCOUNT_IMPLEMENTED:
                assert assumption.name in account.unrealized(ACCOUNT_IMPLEMENTED)

    def test_the_shared_deferral_rule_does_not_claim_cap_contribution(self):
        """`cap_contribution` caps one account and never sees the plan's other
        half, so it cannot enforce a limit shared between them. Naming it would
        let a scenario splitting the deferral across both halves pass twice."""
        account = runtime(AccountKind.ROTH_401K)
        [shared] = [a for a in account.assumptions
                    if a.name == "shared-deferral-limit"]
        assert shared.realized_by != "cap_contribution"
        assert shared.name in account.unrealized(ACCOUNT_IMPLEMENTED)


class TestCapContribution:

    def test_it_refuses_the_excess_and_names_its_size(self):
        ruleset = load(2026)
        decision = runtime(AccountKind.ROTH_IRA).cap_contribution(
            24_000.0, tax_year=2026, ruleset=ruleset)

        assert decision.exceeds_on_this_account_alone
        assert decision.permitted == 7_500.0
        assert decision.refused == 16_500.0

    def test_a_contribution_within_the_limit_is_not_refused(self):
        decision = runtime(AccountKind.ROTH_IRA).cap_contribution(
            3_000.0, tax_year=2026)
        assert not decision.exceeds_on_this_account_alone
        assert decision.permitted == 3_000.0

    def test_an_absent_limit_permits_rather_than_invents_a_restriction(self):
        """Refusing on a figure nobody entered is the same defect pointed the
        other way."""
        decision = runtime(AccountKind.TAXABLE).cap_contribution(
            500_000.0, tax_year=2026)
        assert not decision.exceeds_on_this_account_alone
        assert decision.permitted == 500_000.0

    def test_the_decision_carries_the_ruleset_that_decided_it(self):
        decision = runtime(AccountKind.ROTH_IRA).cap_contribution(
            3_000.0, tax_year=2026)
        assert decision.limit.ruleset_ref == "account-rules/us-federal-2026@1"
        assert decision.tax_year == 2026


class TestApplyMatch:

    def test_a_roth_401k_match_lands_pre_tax_and_is_kept_apart(self):
        """The match is pre-tax even in a Roth plan, so the account holds two
        differently-taxed balances. One reported total would have lost it."""
        result = runtime(AccountKind.ROTH_401K,
                         employer_match_rate=0.5).apply_match(10_000.0)

        assert result["match"] == 5_000.0
        assert result["after_tax_balance"] == 10_000.0
        assert result["pre_tax_balance"] == 5_000.0

    def test_a_traditional_401k_holds_one_pre_tax_balance(self):
        result = runtime(AccountKind.TRADITIONAL_401K,
                         employer_match_rate=0.5).apply_match(10_000.0)
        assert result["after_tax_balance"] == 0.0
        assert result["pre_tax_balance"] == 15_000.0

    def test_the_cap_binds(self):
        result = runtime(AccountKind.ROTH_401K, employer_match_rate=1.0,
                         employer_match_cap=2_000.0).apply_match(10_000.0)
        assert result["match"] == 2_000.0

    def test_no_match_rate_means_no_match(self):
        result = runtime(AccountKind.ROTH_IRA).apply_match(5_000.0)
        assert result["matched"] is False
        assert result["match"] == 0.0
