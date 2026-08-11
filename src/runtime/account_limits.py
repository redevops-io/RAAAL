"""Account rules, pinned to a tax year and read from a versioned table.

The figures live in `account-rules/<name>-<year>@<version>.yaml`. Code reads
them; it does not carry them. A limit embedded in a function is a limit nobody
revisits in November, and one that silently governs every plan compiled the
following year.

**Pinned to a year, not to "current".** A ruleset meaning whatever the latest
figures are would re-judge a stored scenario the moment the annual limits were
updated: the same plan, replayed, would refuse a contribution it previously
permitted, with no version change to point at. A run pins the ruleset for the
year it simulates.

Three states, kept apart, because collapsing them is how a wrong answer acquires
confidence:

    NOT_ENFORCED   no figure is entered for this account and year
    PARTIAL        the figure is right and the inputs to apply it are absent
    ENFORCED       the figure is verified and everything it needs is present

`PARTIAL` is the state this module exists for. The IRA limit is *combined*
across every IRA a person holds, so capping one account against it proves
nothing about the total — a plan contributing the maximum to a Roth IRA and the
maximum again to a traditional IRA passes every single-account check and is
double the legal amount. Reporting that as enforced is the failure mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

RULES_DIR = Path(__file__).resolve().parents[2] / "account-rules"
JURISDICTION = "us-federal"


class Enforcement(str, Enum):
    NOT_ENFORCED = "NOT_ENFORCED"
    PARTIAL = "PARTIAL"
    ENFORCED = "ENFORCED"


class RulesetNotFound(LookupError):
    """No ruleset is published for the requested tax year.

    Raised rather than falling back to the nearest year. Simulating 2019 under
    2026 figures would produce a confident, wrong refusal and nothing would
    indicate which year's rules were applied.
    """


@dataclass(frozen=True)
class Limit:
    """One governing figure, with everything needed to judge how far to trust it."""

    amount: Optional[float]
    rule: str
    """Which entry governed, e.g. `ira_contribution`. Named because a 401(k) and
    an IRA are refused by different rules and a user asking "why" needs the
    right one."""

    tax_year: int
    ruleset_ref: str
    verified: bool = False
    combined_across: Sequence[str] = ()
    """Other account kinds sharing this one limit. Non-empty means a
    single-account check cannot establish compliance."""

    requires: Sequence[str] = ()
    catch_up_50: Optional[float] = None
    catch_up_60_63: Optional[float] = None

    def missing_inputs(self, known: Mapping[str, Any] = ()) -> Sequence[str]:
        """Required facts this scenario does not carry."""
        available = {k for k, v in dict(known).items() if v is not None}
        return tuple(name for name in self.requires if name not in available)

    def enforcement(self, known: Mapping[str, Any] = ()) -> Enforcement:
        if self.amount is None:
            return Enforcement.NOT_ENFORCED
        if not self.verified:
            return Enforcement.PARTIAL
        return (Enforcement.ENFORCED if not self.missing_inputs(known)
                else Enforcement.PARTIAL)

    def why_not_enforced(self, known: Mapping[str, Any] = ()) -> Optional[str]:
        """The sentence a user reads, naming what is missing rather than the
        fact that something is."""
        if self.amount is None:
            return (f"No {self.rule.replace('_', ' ')} figure is entered for "
                    f"{self.tax_year}, so none is applied.")
        if not self.verified:
            return (f"The {self.tax_year} figure of ${self.amount:,.0f} has not "
                    "been checked against the published IRS source. It is "
                    "applied, and it may be wrong.")
        missing = self.missing_inputs(known)
        if not missing:
            return None
        readable = ", ".join(m.replace("_", " ") for m in missing)
        if self.combined_across:
            shared = ", ".join(k.replace("_", " ").title()
                               for k in self.combined_across)
            return (f"This limit is shared with {shared}. Only this account was "
                    f"described, so the combined total cannot be checked — "
                    f"missing: {readable}.")
        return f"Not checked, because this plan does not state: {readable}."

    def to_json(self) -> Dict[str, Any]:
        return {"amount": self.amount, "rule": self.rule,
                "tax_year": self.tax_year, "ruleset_ref": self.ruleset_ref,
                "verified": self.verified,
                "combined_across": list(self.combined_across),
                "requires": list(self.requires),
                "catch_up_50": self.catch_up_50}


@dataclass(frozen=True)
class Ruleset:
    """One tax year's published account rules."""

    name: str
    version: int
    tax_year: int
    source: Mapping[str, Any]
    rules: Mapping[str, Any]
    income_phase_outs: Mapping[str, Any]
    unlimited: Sequence[str]
    not_yet_entered: Sequence[str]

    @property
    def ref(self) -> str:
        """What a run pins. Carried into run conditions so a replay can prove
        which year's figures decided the outcome."""
        return f"account-rules/{self.name}@{self.version}"

    def limit_for(self, account_kind: str, *, rule: Optional[str] = None) -> Limit:
        """The governing limit for one account kind.

        Returns a `NOT_ENFORCED` limit rather than raising, and never defaults to
        unlimited. An account this ruleset does not cover must not acquire
        permission from the ruleset's silence.
        """
        if account_kind in self.unlimited:
            return Limit(amount=None, rule="unlimited", tax_year=self.tax_year,
                         ruleset_ref=self.ref)

        for rule_name, entry in self.rules.items():
            if account_kind not in entry.get("applies_to", ()):
                continue
            if rule is not None and rule_name != rule:
                continue
            # The elective-deferral limit governs what a participant may defer.
            # The annual-additions limit counts employer money too and is a
            # different question, so it never answers "may I contribute this?".
            if rule is None and rule_name == "annual_additions":
                continue

            combined = ()
            if entry.get("combined_across_applies_to"):
                combined = tuple(k for k in entry["applies_to"]
                                 if k != account_kind)
            return Limit(
                amount=float(entry["value"]),
                rule=rule_name, tax_year=self.tax_year, ruleset_ref=self.ref,
                verified=bool(entry.get("verified_against_source", False)),
                combined_across=combined,
                requires=tuple(entry.get("requires") or ()),
                catch_up_50=_optional(entry.get("catch_up_50")),
                catch_up_60_63=_optional(entry.get("catch_up_60_63")))

        return Limit(amount=None, rule=rule or "not-entered",
                     tax_year=self.tax_year, ruleset_ref=self.ref)

    def phase_out_for(self, account_kind: str) -> Optional[Limit]:
        """Income eligibility, which is not a contribution ceiling.

        A Roth contribution can be reduced or disallowed entirely by income —
        a different refusal from exceeding a limit, and one that cannot be
        evaluated until filing status and modified AGI are represented.
        """
        entry = self.income_phase_outs.get(account_kind.lower())
        if entry is None:
            return None
        return Limit(amount=None, rule=f"{account_kind.lower()}_income_phase_out",
                     tax_year=self.tax_year, ruleset_ref=self.ref,
                     verified=bool(entry.get("verified_against_source", False)),
                     requires=tuple(entry.get("requires") or ()))


def _optional(value) -> Optional[float]:
    return None if value is None else float(value)


def load(tax_year: int, *, jurisdiction: str = JURISDICTION,
         version: int = 1, directory: Path = RULES_DIR) -> Ruleset:
    """The published ruleset for one tax year, or a refusal naming the year."""
    ref = f"{jurisdiction}-{tax_year}@{version}"
    path = directory / f"{ref}.yaml"
    if not path.exists():
        published = sorted(p.stem for p in directory.glob(f"{jurisdiction}-*"))
        raise RulesetNotFound(
            f"no account rules published for tax year {tax_year}. Published: "
            f"{', '.join(published) or 'none'}. The nearest year is not "
            "substituted, because a refusal computed from the wrong year's "
            "figures looks identical to a correct one")

    payload = yaml.safe_load(path.read_text())
    return Ruleset(
        name=f"{jurisdiction}-{tax_year}", version=version,
        tax_year=int(payload["tax_year"]),
        source=payload.get("source") or {},
        rules=payload.get("rules") or {},
        income_phase_outs=payload.get("income_phase_outs") or {},
        unlimited=tuple(payload.get("unlimited") or ()),
        not_yet_entered=tuple(payload.get("not_yet_entered") or ()))


def published_years(*, jurisdiction: str = JURISDICTION,
                    directory: Path = RULES_DIR) -> Sequence[int]:
    years = []
    for path in directory.glob(f"{jurisdiction}-*@*.yaml"):
        stem = path.stem.split("@")[0]
        years.append(int(stem.rsplit("-", 1)[1]))
    return tuple(sorted(set(years)))


@dataclass(frozen=True)
class ContributionDecision:
    """What the account permits, what it refused, and what it could not check.

    `refused` is an amount, not a boolean: a plan contributing $24,000 into an
    account permitting $7,500 is wrong by $16,500, and the size is what tells a
    user whether they mis-stated the cadence or the account.
    """

    requested: float
    permitted: float
    refused: float
    limit: Limit
    tax_year: int
    missing_inputs: Sequence[str] = ()

    @property
    def exceeds_on_this_account_alone(self) -> bool:
        """Certain overage. One account already over a shared limit is over it
        however the rest of the picture looks, so this refuses safely without
        the missing inputs."""
        return self.refused > 0

    @property
    def compliance_established(self) -> bool:
        """Whether staying inside the limit was actually *proven*.

        False when the limit is shared and the other accounts were never
        described. Not the same as being over — it is the honest "unknown" that
        a single-account check produces, and reporting it as compliant is how a
        doubled IRA contribution passes.
        """
        return not self.exceeds_on_this_account_alone and not self.missing_inputs

    @property
    def enforcement(self) -> Enforcement:
        if self.limit.amount is None:
            return Enforcement.NOT_ENFORCED
        if not self.limit.verified or self.missing_inputs:
            return Enforcement.PARTIAL
        return Enforcement.ENFORCED

    def to_json(self) -> Dict[str, Any]:
        return {"requested": self.requested, "permitted": self.permitted,
                "refused": self.refused, "tax_year": self.tax_year,
                "exceeds_on_this_account_alone":
                    self.exceeds_on_this_account_alone,
                "compliance_established": self.compliance_established,
                "missing_inputs": list(self.missing_inputs),
                "limit": self.limit.to_json()}
