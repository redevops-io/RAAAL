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
    ROTH_IRA = "ROTH_IRA"
    TRADITIONAL_IRA = "TRADITIONAL_IRA"
    HSA = "HSA"
    PLAN_529 = "PLAN_529"
    TRUST = "TRUST"


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
            AccountKind.TRADITIONAL_401K, AccountKind.ROTH_IRA,
            AccountKind.TRADITIONAL_IRA, AccountKind.HSA, AccountKind.PLAN_529,
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
