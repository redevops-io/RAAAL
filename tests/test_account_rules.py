"""Tax-year account rules, and the difference between a right figure and an
enforceable one.

Two distinct claims, repeatedly confused:

    the figure is correct         $7,500 is the 2026 IRA limit
    the figure can be applied     this plan's total IRA contributions are known

The second does not follow from the first. The IRA limit is combined across
every IRA a person holds, so checking one account can prove a plan is *over* and
can never prove it is *under*. Reporting the second as established is how a
doubled contribution passes every check.
"""
from __future__ import annotations

import pytest
import yaml

from src.runtime import AccountKind, AccountRuntime
from src.runtime.account_limits import (
    RULES_DIR,
    Enforcement,
    RulesetNotFound,
    load,
    published_years,
)
from src.workspace.account_support import Support, support_for


def account(kind):
    return AccountRuntime(name="a", version=1, account_kind=kind)


class TestRulesetsArePinnedToATaxYear:

    def test_both_verified_years_are_published(self):
        assert set(published_years()) >= {2025, 2026}

    def test_the_ref_names_the_year(self):
        assert load(2026).ref == "account-rules/us-federal-2026@1"

    def test_an_unpublished_year_refuses_rather_than_substituting(self):
        """Simulating 2019 under 2026 figures would produce a confident, wrong
        refusal, and nothing would indicate which year's rules applied."""
        with pytest.raises(RulesetNotFound, match="2019"):
            load(2019)

    def test_the_years_hold_different_figures(self):
        """The reason the split exists. One 'current' table would silently
        re-judge a stored 2025 scenario the moment 2026 landed."""
        assert (load(2025).limit_for("ROTH_IRA").amount
                != load(2026).limit_for("ROTH_IRA").amount)

    def test_updating_one_year_cannot_reach_another(self, tmp_path):
        payload = yaml.safe_load((RULES_DIR / "us-federal-2026@1.yaml").read_text())
        payload["rules"]["ira_contribution"]["value"] = 999_999
        (tmp_path / "us-federal-2026@1.yaml").write_text(yaml.safe_dump(payload))
        (tmp_path / "us-federal-2025@1.yaml").write_text(
            (RULES_DIR / "us-federal-2025@1.yaml").read_text())

        assert load(2026, directory=tmp_path).limit_for("ROTH_IRA").amount == 999_999
        assert load(2025, directory=tmp_path).limit_for("ROTH_IRA").amount == 7_000


class TestTheVerifiedFigures:
    """Transcribed from the IRS sources named in each ruleset and checked field
    by field. A wrong figure here refuses a legal plan or permits an illegal
    one, so they are asserted rather than trusted."""

    @pytest.mark.parametrize("year,kind,expected", [
        (2025, "ROTH_IRA", 7_000), (2026, "ROTH_IRA", 7_500),
        (2025, "TRADITIONAL_IRA", 7_000), (2026, "TRADITIONAL_IRA", 7_500),
        (2025, "ROTH_401K", 23_500), (2026, "ROTH_401K", 24_500),
        (2025, "TRADITIONAL_401K", 23_500), (2026, "TRADITIONAL_401K", 24_500),
    ])
    def test_contribution_limits(self, year, kind, expected):
        assert load(year).limit_for(kind).amount == expected

    @pytest.mark.parametrize("year,fifty,sixty", [
        (2025, 7_500, 11_250), (2026, 8_000, 11_250)])
    def test_deferral_catch_ups(self, year, fifty, sixty):
        limit = load(year).limit_for("ROTH_401K")
        assert limit.catch_up_50 == fifty
        assert limit.catch_up_60_63 == sixty

    @pytest.mark.parametrize("year,expected", [(2025, 1_000), (2026, 1_100)])
    def test_ira_catch_up(self, year, expected):
        assert load(year).limit_for("ROTH_IRA").catch_up_50 == expected

    @pytest.mark.parametrize("year,expected", [(2025, 70_000), (2026, 72_000)])
    def test_annual_additions(self, year, expected):
        assert load(year).limit_for(
            "ROTH_401K", rule="annual_additions").amount == expected

    @pytest.mark.parametrize("year,self_only,family", [
        (2025, 4_300, 8_550), (2026, 4_400, 8_750)])
    def test_hsa(self, year, self_only, family):
        rules = load(year).rules
        assert rules["hsa_self_only"]["value"] == self_only
        assert rules["hsa_family"]["value"] == family

    def test_every_shipped_figure_is_marked_verified(self):
        for year in (2025, 2026):
            for name, rule in load(year).rules.items():
                assert rule.get("verified_against_source") is True, (year, name)


class TestTheAnnualAdditionsLimitIsADifferentRule:
    """Merging it with the elective-deferral limit would refuse a legal plan
    that defers the employee maximum and receives a large employer match."""

    def test_it_is_not_returned_as_the_contribution_limit(self):
        assert load(2026).limit_for("ROTH_401K").rule == "employee_deferral"

    def test_it_is_skipped_by_rule_not_by_luck_of_ordering(self):
        """The plain assertion above passes because `employee_deferral` happens
        to precede `annual_additions` in the file. Reordered, a $72,000 ceiling
        would answer "may I contribute this?" and a plan deferring $30,000 would
        be cleared. This puts it first and checks the guard, not the order."""
        from dataclasses import replace

        ruleset = load(2026)
        reordered = replace(ruleset, rules={
            "annual_additions": ruleset.rules["annual_additions"],
            **{k: v for k, v in ruleset.rules.items() if k != "annual_additions"},
        })
        assert list(reordered.rules)[0] == "annual_additions"
        assert reordered.limit_for("ROTH_401K").rule == "employee_deferral"

    def test_it_is_reachable_when_asked_for_by_name(self):
        limit = load(2026).limit_for("ROTH_401K", rule="annual_additions")
        assert limit.amount == 72_000

    def test_it_counts_employer_money_and_says_so(self):
        limit = load(2026).limit_for("ROTH_401K", rule="annual_additions")
        assert "employer_contributions_for_tax_year" in limit.requires


class TestASharedLimitCannotBeEnforcedFromOneAccount:

    def test_the_ira_limit_is_combined_across_both_ira_kinds(self):
        assert load(2026).limit_for("ROTH_IRA").combined_across == (
            "TRADITIONAL_IRA",)

    def test_the_deferral_limit_is_combined_across_both_plan_halves(self):
        assert load(2026).limit_for("ROTH_401K").combined_across == (
            "TRADITIONAL_401K",)

    def test_two_maxed_iras_each_pass_alone(self):
        """The failure the combined limit exists to catch: $7,500 into a Roth
        IRA and $7,500 into a traditional IRA is double the legal amount, and
        every single-account check clears it."""
        ruleset = load(2026)
        roth = account(AccountKind.ROTH_IRA).cap_contribution(
            7_500, tax_year=2026, ruleset=ruleset)
        trad = account(AccountKind.TRADITIONAL_IRA).cap_contribution(
            7_500, tax_year=2026, ruleset=ruleset)

        assert not roth.exceeds_on_this_account_alone
        assert not trad.exceeds_on_this_account_alone
        # ...and neither claims to have established compliance.
        assert not roth.compliance_established
        assert not trad.compliance_established

    def test_being_under_alone_is_not_compliance(self):
        decision = account(AccountKind.ROTH_IRA).cap_contribution(
            3_000, tax_year=2026)
        assert not decision.exceeds_on_this_account_alone
        assert not decision.compliance_established
        assert "all_ira_contributions_for_tax_year" in decision.missing_inputs

    def test_being_over_alone_is_still_certain(self):
        """One account already over a shared limit is over it however the rest
        of the picture looks, so this refuses without the missing inputs."""
        decision = account(AccountKind.ROTH_IRA).cap_contribution(
            24_000, tax_year=2026)
        assert decision.exceeds_on_this_account_alone
        assert decision.refused == 16_500

    def test_supplying_the_totals_establishes_compliance(self):
        """A derivation nobody has seen change is a derivation nobody tested."""
        decision = account(AccountKind.ROTH_IRA).cap_contribution(
            3_000, tax_year=2026,
            known={"all_ira_contributions_for_tax_year": 3_000,
                   "participant_age": 41})
        assert decision.compliance_established
        assert decision.enforcement is Enforcement.ENFORCED

    def test_the_reason_names_the_other_account(self):
        support = support_for("ROTH", year=2026)
        stated = " ".join(support.unenforced_behaviours)
        assert "shared" in stated.lower()
        assert "Traditional Ira" in stated or "TRADITIONAL_IRA" in stated


class TestCatchUpTiers:

    def test_ages_60_to_63_get_the_higher_deferral_catch_up(self):
        decision = account(AccountKind.ROTH_401K).cap_contribution(
            60_000, tax_year=2026, known={"participant_age": 61})
        assert decision.permitted == 24_500 + 11_250

    def test_age_50_gets_the_standard_catch_up(self):
        decision = account(AccountKind.ROTH_401K).cap_contribution(
            60_000, tax_year=2026, known={"participant_age": 52})
        assert decision.permitted == 24_500 + 8_000

    def test_age_64_returns_to_the_standard_catch_up(self):
        """The higher tier is a band, not a floor. Treating it as 60-and-over
        would permit an extra $3,250 to everyone past 63."""
        decision = account(AccountKind.ROTH_401K).cap_contribution(
            60_000, tax_year=2026, known={"participant_age": 64})
        assert decision.permitted == 24_500 + 8_000

    def test_an_unstated_age_gets_no_catch_up(self):
        decision = account(AccountKind.ROTH_401K).cap_contribution(
            60_000, tax_year=2026)
        assert decision.permitted == 24_500


class TestIncomeEligibilityIsNotAContributionCeiling:
    """A Roth contribution can be reduced or disallowed entirely by income —
    a different refusal from exceeding a limit, and one nothing here can yet
    evaluate."""

    def test_the_phase_out_is_recorded(self):
        phase_out = load(2026).phase_out_for("ROTH_IRA")
        assert phase_out is not None
        assert phase_out.verified

    def test_it_stays_unenforceable_until_filing_status_and_magi_exist(self):
        phase_out = load(2026).phase_out_for("ROTH_IRA")
        assert phase_out.enforcement(known={}) is not Enforcement.ENFORCED
        assert set(phase_out.requires) == {"filing_status", "modified_agi"}

    def test_the_2025_thresholds_are_recorded(self):
        bands = load(2025).income_phase_outs["roth_ira"]["by_filing_status"]
        assert bands["single_or_head_of_household"] == {"begins": 150_000,
                                                        "ends": 165_000}
        assert bands["married_filing_jointly"] == {"begins": 236_000,
                                                   "ends": 246_000}
        assert bands["married_filing_separately_living_with_spouse"] == {
            "begins": 0, "ends": 10_000}

    def test_the_2026_thresholds_are_recorded(self):
        bands = load(2026).income_phase_outs["roth_ira"]["by_filing_status"]
        assert bands["single_or_head_of_household"] == {"begins": 153_000,
                                                        "ends": 168_000}
        assert bands["married_filing_jointly"] == {"begins": 242_000,
                                                   "ends": 252_000}

    def test_an_unsupplied_2026_band_is_absent_not_inherited(self):
        """Carrying 2025's figure forward would give an unstated number the
        authority of a published one."""
        bands = load(2026).income_phase_outs["roth_ira"]["by_filing_status"]
        assert bands["married_filing_separately_living_with_spouse"] is None


class TestTheDisplayStaysHonest:

    def test_a_shared_limit_reports_partial_not_enforced(self):
        assert support_for("ROTH", year=2026).enforced is Support.PARTIAL
        assert support_for("ROTH_401K", year=2026).enforced is Support.PARTIAL

    def test_a_taxable_account_has_nothing_to_enforce(self):
        assert support_for("TAXABLE", year=2026).enforced is Support.NO
