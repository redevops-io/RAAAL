"""Account rules, which are not tax rules.

Contribution limits, withdrawal penalties, required distributions, employer
matching and rollover eligibility are properties of an *account*, and they bind
whether or not any tax is modelled. Folding them into a tax runtime would make
"Roth versus Traditional" a question about tax rates, when most of what differs
is when money may move.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Sequence

from .base import RuntimeArtifact, RuntimeAssumption, RuntimeLimitation


class AccountKind(str, Enum):
    TAXABLE = "TAXABLE"
    TRADITIONAL_401K = "TRADITIONAL_401K"
    ROTH_401K = "ROTH_401K"
    """First-class, and deliberately not an alias.

    Aliased to `TRADITIONAL_401K` it would report tax-deferred contributions and
    taxable withdrawals, which is the opposite of what it does. Aliased to
    `ROTH_IRA` it would report an IRA's contribution limit, no employer match,
    and IRA withdrawal mechanics — none of which apply to an employer plan.

    Either substitution records a tax treatment the user did not describe, which
    is this project's founding defect. Until this existed, a Roth 401(k) plan
    was left unpinned and could not claim isolated attribution — correct, and
    now unnecessary."""

    ROTH_IRA = "ROTH_IRA"
    TRADITIONAL_IRA = "TRADITIONAL_IRA"
    HSA = "HSA"
    PLAN_529 = "PLAN_529"
    TRUST = "TRUST"


#: Employer plans share one employee elective-deferral limit across their
#: traditional and Roth halves. Declared here because the constraint belongs to
#: the pair, not to either account alone — a limit stated twice is a limit that
#: gets applied twice.
SHARES_DEFERRAL_LIMIT = frozenset(
    {AccountKind.TRADITIONAL_401K, AccountKind.ROTH_401K})

#: Contributions made after tax, so qualified growth and withdrawals are not
#: taxed again.
AFTER_TAX_CONTRIBUTIONS = frozenset(
    {AccountKind.ROTH_IRA, AccountKind.ROTH_401K, AccountKind.TAXABLE})


@dataclass(frozen=True)
class AccountRuntime(RuntimeArtifact):
    kind: ClassVar[str] = "account"

    name: str
    version: int
    account_kind: AccountKind
    annual_contribution_limit: Optional[float] = None
    employer_match_rate: Optional[float] = None
    employer_match_cap: Optional[float] = None
    early_withdrawal_penalty: Optional[float] = None
    penalty_free_age: Optional[int] = None
    required_distribution_age: Optional[int] = None
    title: str = ""

    @property
    def accepts_employer_shares(self) -> bool:
        """Whether employer stock can be delivered into this account.

        A 529 or HSA cannot receive a vest. Asked of the account because the
        account is the thing that knows, rather than the flow runtime carrying a
        table of account types."""
        return self.account_kind not in {AccountKind.PLAN_529, AccountKind.HSA}

    @property
    def tax_deferred(self) -> bool:
        """Whether gains go untaxed inside the account during accumulation.

        The predicate a tax runtime's `no-capital-gains` limitation points at.
        Asked of the account rather than inferred by the environment, because the
        account is the thing that knows."""
        return self.account_kind in {
            AccountKind.TRADITIONAL_401K, AccountKind.ROTH_401K,
            AccountKind.ROTH_IRA, AccountKind.TRADITIONAL_IRA, AccountKind.HSA,
            AccountKind.PLAN_529,
        }

    @property
    def after_tax_contributions(self) -> bool:
        """Whether money enters having already been taxed.

        The difference between a Roth 401(k) and a traditional one, and the
        reason they cannot share an identity: the same contribution produces a
        different balance and a different withdrawal.
        """
        return self.account_kind in AFTER_TAX_CONTRIBUTIONS

    @property
    def shares_employee_deferral_limit(self) -> bool:
        """Whether the employee limit is shared with the plan's other half.

        A Roth 401(k) and a traditional 401(k) do not each get the annual
        limit; they share one. Modelling them as independent would let a
        scenario contribute twice what the law permits.
        """
        return self.account_kind in SHARES_DEFERRAL_LIMIT

    @property
    def employer_contributions_are_pre_tax(self) -> bool:
        """Employer money into a Roth 401(k) is still pre-tax.

        It lands in a traditional sub-account and is taxed on withdrawal, so a
        Roth 401(k) with a match holds two differently-taxed balances. Declared
        because a plan that reports one balance has already lost the
        distinction.
        """
        return self.account_kind in SHARES_DEFERRAL_LIMIT

    # ---- mechanisms -------------------------------------------------------
    #
    # These are what `realized_by` names. Until they existed, every account
    # assumption pointed at a mechanism that did not, and `IMPLEMENTED` was
    # empty because that was the only honest thing it could be.

    def cap_contribution(self, annual_amount: float, *, tax_year: int,
                         ruleset=None, known: Optional[Dict[str, Any]] = None):
        """What this account permits of a stated annual contribution.

        Refuses rather than silently capping. The assumption text promises
        contributions above the limit are "refused rather than silently
        accepted", and quietly reducing $24,000 to $7,500 would produce a
        balance the user never described while every displayed figure looked
        deliberate.

        A limit this ruleset does not carry permits the full amount and says so.
        Refusing on an absent figure would invent a restriction, which is the
        same defect pointed the other way.

        What it deliberately does **not** do is establish compliance with a
        shared limit. The IRA ceiling is combined across every IRA a person
        holds, so this method can prove a plan is over it and can never prove a
        plan is under it. `ContributionDecision.compliance_established` carries
        that difference; `missing_inputs` names what would settle it.
        """
        from .account_limits import ContributionDecision, load

        ruleset = load(tax_year) if ruleset is None else ruleset
        limit = ruleset.limit_for(self.account_kind.value)
        known = dict(known or {})

        allowance = limit.amount
        age = known.get("participant_age")
        if allowance is not None and age is not None:
            if 60 <= age <= 63 and limit.catch_up_60_63:
                allowance += limit.catch_up_60_63
            elif age >= 50 and limit.catch_up_50:
                allowance += limit.catch_up_50

        if allowance is None:
            permitted, refused = annual_amount, 0.0
        else:
            permitted = min(annual_amount, allowance)
            refused = max(0.0, annual_amount - allowance)

        return ContributionDecision(
            requested=annual_amount, permitted=permitted, refused=refused,
            limit=limit, tax_year=ruleset.tax_year,
            missing_inputs=limit.missing_inputs(known))

    def apply_match(self, employee_contribution: float):
        """Employer money on top of an employee contribution.

        Returns the match and where it lands. In a Roth 401(k) the match is
        pre-tax and sits in a traditional sub-account, so this returns two
        balances rather than one total — a plan reporting a single Roth balance
        has already lost the distinction that makes the withdrawal different.
        """
        if not self.employer_match_rate:
            return {"match": 0.0, "after_tax_balance": employee_contribution,
                    "pre_tax_balance": 0.0, "matched": False}

        match = employee_contribution * self.employer_match_rate
        if self.employer_match_cap is not None:
            match = min(match, self.employer_match_cap)

        employee_after_tax = (employee_contribution
                              if self.after_tax_contributions else 0.0)
        employee_pre_tax = (0.0 if self.after_tax_contributions
                            else employee_contribution)
        # The match itself is pre-tax whenever the plan is an employer plan,
        # including the Roth half.
        match_pre_tax = match if self.employer_contributions_are_pre_tax else 0.0
        match_after_tax = match - match_pre_tax

        return {
            "match": match,
            "after_tax_balance": employee_after_tax + match_after_tax,
            "pre_tax_balance": employee_pre_tax + match_pre_tax,
            "matched": True,
        }

    def declared_form(self) -> Dict[str, Any]:
        return {
            "kind": self.kind, "name": self.name, "version": self.version,
            "account_kind": self.account_kind.value,
            "annual_contribution_limit": self.annual_contribution_limit,
            "employer_match_rate": self.employer_match_rate,
            "employer_match_cap": self.employer_match_cap,
            "early_withdrawal_penalty": self.early_withdrawal_penalty,
            "penalty_free_age": self.penalty_free_age,
            "required_distribution_age": self.required_distribution_age,
            "title": self.title,
        }

    def comparable_form(self) -> Dict[str, Any]:
        declared = self.declared_form()
        for prose in ("title", "name", "version"):
            declared.pop(prose, None)
        return declared

    @property
    def assumptions(self) -> Sequence[RuntimeAssumption]:
        out = []
        if self.shares_employee_deferral_limit:
            out.append(RuntimeAssumption(
                name="shared-deferral-limit",
                statement=("The employee deferral limit is shared with the "
                           "plan's other half; contributions to both together "
                           "are capped once, not twice."),
                # Deliberately NOT `cap_contribution`. That method caps one
                # account against the limit; it never sees the plan's other
                # half, so it cannot enforce a limit shared between them.
                # Naming it here would report a rule as enforced on the strength
                # of a mechanism that cannot perform it — a scenario splitting
                # $24,500 across both halves would pass twice.
                realized_by="cap_shared_deferral",
            ))
        if self.employer_contributions_are_pre_tax and self.employer_match_rate:
            out.append(RuntimeAssumption(
                name="employer-match-tax-treatment",
                statement=("Employer contributions are pre-tax and are taxed on "
                           "withdrawal even in a Roth plan, so the account holds "
                           "two differently-taxed balances."),
                realized_by="apply_match",
            ))
        if self.annual_contribution_limit is not None:
            out.append(RuntimeAssumption(
                name="contribution-limit",
                statement=(f"Contributions above "
                           f"${self.annual_contribution_limit:,.0f} per year are "
                           f"refused rather than silently accepted."),
                realized_by="cap_contribution",
            ))
        if self.employer_match_rate:
            out.append(RuntimeAssumption(
                name="employer-match",
                statement=(f"The employer adds "
                           f"{self.employer_match_rate:.0%} of contributions"
                           + (f", capped at ${self.employer_match_cap:,.0f}"
                              if self.employer_match_cap else "") + "."),
                realized_by="apply_match",
            ))
        return tuple(out)

    @property
    def limitations(self) -> Sequence[RuntimeLimitation]:
        out = []
        if self.required_distribution_age is None:
            out.append(RuntimeLimitation(
                name="no-required-distributions",
                statement=("Required minimum distributions are not modelled, so "
                           "a long horizon will not show forced withdrawals."),
            ))
        if self.early_withdrawal_penalty is None:
            out.append(RuntimeLimitation(
                name="no-early-withdrawal-penalty",
                statement="Early-withdrawal penalties are not applied.",
            ))
        return tuple(out)


TAXABLE_BROKERAGE = AccountRuntime(
    name="taxable-brokerage", version=1, account_kind=AccountKind.TAXABLE,
    title="Ordinary taxable brokerage account",
)

#: Both name methods on `AccountRuntime` above. `tests/test_account_limits.py`
#: resolves every entry to a real callable, so this tuple cannot claim a
#: mechanism into existence — the failure it would otherwise invite is a display
#: that reads ENFORCED because someone added a string.
IMPLEMENTED = ("cap_contribution", "apply_match")
