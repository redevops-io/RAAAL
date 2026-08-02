"""Contribution limits: the figure, the mechanism, and the distance between them.

Three things can go wrong and they are not the same thing:

    the mechanism is absent      nothing caps anything
    the figure is absent         nothing to cap against
    the figure is unchecked      capping happens, against a number nobody read

The third is the dangerous one, because it is the only one that produces a
confident answer with no visible defect.
"""
from __future__ import annotations

import datetime as dt

import pytest
import yaml

from src.runtime import ACCOUNT_IMPLEMENTED, AccountKind, AccountRuntime
from src.runtime.account_limits import LIMITS_DIR, LimitState, load
from src.workspace.account_support import Support, support_for


@pytest.fixture
def table():
    return load()


def runtime(kind, **kw):
    return AccountRuntime(name="a", version=1, account_kind=kind, **kw)


class TestTheRegistryCannotClaimAMechanismIntoExistence:
    """`IMPLEMENTED` is a hand-written tuple. Adding a string to it moves every
    account display to ENFORCED, so the strings must resolve to real code."""

    def test_every_implemented_name_is_a_real_callable(self):
        for name in ACCOUNT_IMPLEMENTED:
            attribute = getattr(AccountRuntime, name, None)
            assert callable(attribute), (
                f"ACCOUNT_IMPLEMENTED names {name!r}, which is not a method on "
                "AccountRuntime. A realization registry that names absent "
                "mechanisms reports unenforced behaviour as enforced")

    def test_every_declared_mechanism_is_implemented_or_reported_unrealized(self):
        """The other direction: a declared assumption whose mechanism is absent
        must show up in `unrealized`, not vanish."""
        account = runtime(AccountKind.ROTH_401K, annual_contribution_limit=7000.0,
                          employer_match_rate=0.5)
        for assumption in account.assumptions:
            if assumption.realized_by not in ACCOUNT_IMPLEMENTED:
                assert assumption.name in account.unrealized(ACCOUNT_IMPLEMENTED)


class TestTheFigureComesFromTheTable:

    def test_a_roth_ira_limit_is_read_not_remembered(self, table):
        limit = table.limit_for("ROTH_IRA", 2026)
        assert limit.amount is not None
        assert limit.rule == "ira_contribution"

    def test_an_ira_limit_is_shared_with_the_other_ira(self, table):
        """A Roth IRA and a traditional IRA do not each get the full amount."""
        assert "TRADITIONAL_IRA" in table.limit_for("ROTH_IRA", 2026).shared_with

    def test_a_401k_limit_is_shared_across_both_halves(self, table):
        assert "TRADITIONAL_401K" in table.limit_for("ROTH_401K", 2026).shared_with

    def test_an_uncovered_year_is_absent_not_unlimited(self, table):
        """Silence must not read as permission."""
        limit = table.limit_for("ROTH_IRA", 1998)
        assert limit.state is LimitState.ABSENT
        assert limit.amount is None

    def test_an_account_kind_not_yet_entered_is_absent(self, table):
        assert table.limit_for("HSA", 2026).state is LimitState.ABSENT

    def test_a_taxable_account_is_unlimited_deliberately(self, table):
        limit = table.limit_for("TAXABLE", 2026)
        assert limit.rule == "unlimited"


class TestCapContribution:

    def test_it_refuses_the_excess_and_names_its_size(self, table):
        """$24,000/year into an account permitting far less is not a rounding
        error, and the size is what tells a user they mis-stated the account."""
        decision = runtime(AccountKind.ROTH_IRA).cap_contribution(
            24_000.0, year=2026, table=table)

        assert not decision.within_limit
        assert decision.refused == pytest.approx(
            24_000.0 - table.limit_for("ROTH_IRA", 2026).amount)
        assert decision.permitted == table.limit_for("ROTH_IRA", 2026).amount

    def test_a_contribution_within_the_limit_passes_whole(self, table):
        decision = runtime(AccountKind.ROTH_IRA).cap_contribution(
            3_000.0, year=2026, table=table)
        assert decision.within_limit
        assert decision.permitted == 3_000.0
        assert decision.refused == 0.0

    def test_an_absent_limit_permits_rather_than_invents_a_restriction(self, table):
        """Refusing on a figure nobody entered would be the same defect pointed
        the other way."""
        decision = runtime(AccountKind.TAXABLE).cap_contribution(
            500_000.0, year=2026, table=table)
        assert decision.within_limit
        assert decision.permitted == 500_000.0

    def test_catch_up_raises_the_allowance_for_an_older_saver(self, table):
        limit = table.limit_for("ROTH_IRA", 2026)
        if not limit.catch_up_50:
            pytest.skip("no catch-up figure entered for this year")
        younger = runtime(AccountKind.ROTH_IRA).cap_contribution(
            50_000.0, year=2026, table=table, age=40)
        older = runtime(AccountKind.ROTH_IRA).cap_contribution(
            50_000.0, year=2026, table=table, age=55)
        assert older.permitted == younger.permitted + limit.catch_up_50


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


class TestAnUncheckedFigureIsNotEnforcement:
    """The failure this whole file exists for: the mechanism runs, the number is
    wrong, and every display reads ENFORCED."""

    def test_an_unverified_limit_reports_partial_not_enforced(self):
        support = support_for("ROTH", year=2026)
        if support.declared == "" or not support.declared_behaviours:
            pytest.skip("no limit governs this declared kind")
        assert support.enforced is not Support.YES

    def test_it_says_the_figure_may_be_wrong(self):
        support = support_for("ROTH_401K", year=2026)
        stated = " ".join(support.unenforced_behaviours).lower()
        if "has not been checked" in stated:
            assert "may be wrong" in stated

    def test_verifying_the_figure_moves_the_state(self, tmp_path, monkeypatch):
        """A derivation nobody has seen change is a derivation nobody tested."""
        payload = yaml.safe_load((LIMITS_DIR / "us-federal@1.yaml").read_text())
        for rule in payload["limits"].values():
            for entry in rule["by_year"].values():
                entry["verified_against_source"] = True
        (tmp_path / "us-federal@1.yaml").write_text(yaml.safe_dump(payload))

        verified = load(directory=tmp_path)
        assert verified.limit_for("ROTH_IRA", 2026).state is LimitState.VERIFIED
        assert load().limit_for("ROTH_IRA", 2026).state is LimitState.UNVERIFIED

    def test_the_shipped_table_is_still_unverified(self):
        """Guards the flag itself. If someone flips the file without reading the
        IRS notice, this is the test that should make them stop and think."""
        payload = yaml.safe_load((LIMITS_DIR / "us-federal@1.yaml").read_text())
        assert payload["verified_against_source"] is False, (
            "the shipped limits are marked verified — if they genuinely were "
            "checked against the published figures, update this test and say so "
            "in the commit")
