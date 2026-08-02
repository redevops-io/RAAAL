"""Contribution limits, read from a versioned table rather than remembered.

The figures live in `accounts/<name>@<version>.yaml`. Code reads them; it does
not carry them. A limit embedded in a function is a limit nobody revisits in
November, and one that silently governs every plan compiled the following year.

Two states are kept apart, because collapsing them is how a wrong number becomes
an enforced one:

    ABSENT       no limit is entered for this account kind and year
    UNVERIFIED   a figure is entered but has not been read off the IRS notice

`ABSENT` cannot enforce. `UNVERIFIED` enforces but refuses to claim `ENFORCED`,
because a mechanism that runs correctly against the wrong number is worse than
one that does not run — it produces a confident answer with no visible defect.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

LIMITS_DIR = Path(__file__).resolve().parents[2] / "accounts"


class LimitState(str, Enum):
    ABSENT = "ABSENT"
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"


@dataclass(frozen=True)
class Limit:
    """One governing limit, with the provenance that decides how far to trust it."""

    amount: Optional[float]
    state: LimitState
    rule: str
    """Which entry governed, e.g. `employee_deferral`. Named because a 401(k) and
    an IRA are refused by different rules and a user asking "why" needs the
    right one."""

    year: int
    shared_with: Sequence[str] = ()
    catch_up_50: Optional[float] = None
    source: str = ""

    @property
    def enforceable(self) -> bool:
        return self.amount is not None and self.state is not LimitState.ABSENT

    @property
    def why_not_enforced(self) -> Optional[str]:
        if self.state is LimitState.ABSENT:
            return (f"No contribution limit is entered for this account in "
                    f"{self.year}, so none is applied.")
        if self.state is LimitState.UNVERIFIED:
            return (f"The {self.year} limit of ${self.amount:,.0f} has not been "
                    "checked against the published IRS figure. It is applied, "
                    "and it may be wrong.")
        return None

    def to_json(self) -> Dict[str, Any]:
        return {"amount": self.amount, "state": self.state.value,
                "rule": self.rule, "year": self.year,
                "shared_with": list(self.shared_with),
                "catch_up_50": self.catch_up_50}


@dataclass(frozen=True)
class LimitTable:
    name: str
    version: int
    source: str
    file_verified: bool
    rules: Mapping[str, Any]
    unlimited: Sequence[str]
    not_yet_entered: Sequence[str]

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def limit_for(self, account_kind: str, year: int) -> Limit:
        """The governing limit for one account kind in one year.

        Returns an `ABSENT` limit rather than raising, and rather than
        defaulting to unlimited. An account this table does not cover must not
        acquire permission from the table's silence.
        """
        if account_kind in self.unlimited:
            return Limit(amount=None, state=LimitState.ABSENT, year=year,
                         rule="unlimited", source=self.source)

        for rule_name, rule in self.rules.items():
            if account_kind not in rule.get("applies_to", ()):
                continue
            entry = (rule.get("by_year") or {}).get(year)
            if entry is None:
                return Limit(amount=None, state=LimitState.ABSENT, year=year,
                             rule=rule_name, source=self.source)

            verified = bool(entry.get("verified_against_source",
                                      self.file_verified))
            shared = ()
            if rule.get("shared_across_applies_to"):
                shared = tuple(k for k in rule["applies_to"] if k != account_kind)
            return Limit(
                amount=float(entry["amount"]),
                state=LimitState.VERIFIED if verified else LimitState.UNVERIFIED,
                rule=rule_name, year=year, shared_with=shared,
                catch_up_50=(float(entry["catch_up_50"])
                             if entry.get("catch_up_50") is not None else None),
                source=self.source)

        return Limit(amount=None, state=LimitState.ABSENT, year=year,
                     rule="not-entered", source=self.source)


def load(ref: str = "us-federal@1", *, directory: Path = LIMITS_DIR) -> LimitTable:
    name, _, version = ref.partition("@")
    payload = yaml.safe_load((directory / f"{ref}.yaml").read_text())
    return LimitTable(
        name=name, version=int(version), source=payload.get("source", ""),
        file_verified=bool(payload.get("verified_against_source", False)),
        rules=payload.get("limits") or {},
        unlimited=tuple(payload.get("unlimited") or ()),
        not_yet_entered=tuple(payload.get("not_yet_entered") or ()))


@dataclass(frozen=True)
class ContributionDecision:
    """What the account permits, and what it refused.

    `refused` is an amount, not a boolean: a plan that contributes $24,000 into
    an account permitting $7,500 is wrong by $16,500, and the size is the thing
    that tells a user whether they mis-stated the cadence or the account.
    """

    requested: float
    permitted: float
    refused: float
    limit: Limit
    year: int

    @property
    def within_limit(self) -> bool:
        return self.refused <= 0

    def to_json(self) -> Dict[str, Any]:
        return {"requested": self.requested, "permitted": self.permitted,
                "refused": self.refused, "within_limit": self.within_limit,
                "year": self.year, "limit": self.limit.to_json()}
