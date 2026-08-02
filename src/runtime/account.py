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
                realized_by="cap_contribution",
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

IMPLEMENTED = ()
