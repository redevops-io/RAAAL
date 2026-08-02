"""What the system recognises, can compare, and actually enforces — per rule.

    Recognised     the words were read and represented
    Comparable     a run pins it, so a comparison can evaluate the dimension
    Enforced       the declared behaviour executes

Three claims about one account, kept apart because a single "supported" badge
collapses them and the third is the one a reader assumes from the first two.

**Per behaviour, not per account.** "Roth IRA supported" tells a user nothing
actionable. "Contribution limit ENFORCED, shared IRA limit PARTIAL, early
withdrawal NOT MODELLED, state tax NOT MODELLED" tells them exactly which
figures to trust and which question to answer next.

**Input, assumption and output are labelled**, because users blur them and the
blur is where a PARTIAL becomes invisible:

    input       $7,000 into a Roth IRA for 2026
    assumption  no other IRA contributions were declared
    output      accepted; compliance partially established

Without the middle line, "partially established" reads as a defect in the
system rather than a consequence of what the user did not say.

Nothing here is computed. Every state is read from the runtime declarations and
realization checks that already decide it, so the confirmation card, the result
and the worksheet cannot describe one run differently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class Enforcement(str, Enum):
    ENFORCED = "ENFORCED"
    PARTIAL = "PARTIAL"
    """Runs, and cannot establish what it claims. The state that matters most,
    and the one a badge would round to whichever neighbour it resembles."""

    NOT_MODELLED = "NOT_MODELLED"
    UNRESOLVED = "UNRESOLVED"
    """Could be modelled; an input nobody supplied is missing. Answerable by
    asking rather than by building."""


class Facet(str, Enum):
    INPUT = "INPUT"
    ASSUMPTION = "ASSUMPTION"
    OUTPUT = "OUTPUT"


@dataclass(frozen=True)
class RuleDisclosure:
    """One declared behaviour, and how far it is actually carried out."""

    rule: str
    enforcement: Enforcement
    why: str = ""
    missing_inputs: Sequence[str] = ()

    def to_json(self) -> Dict[str, Any]:
        return {"rule": self.rule, "enforcement": self.enforcement.value,
                "why": self.why, "missing_inputs": list(self.missing_inputs)}


@dataclass(frozen=True)
class Statement:
    """One line, labelled by what kind of thing it is."""

    facet: Facet
    text: str

    def to_json(self) -> Dict[str, str]:
        return {"facet": self.facet.value, "text": self.text}


@dataclass(frozen=True)
class ScopeDisclosure:
    """Recognition, comparability and enforcement for one plan."""

    recognised: Sequence[str] = ()
    comparable: Sequence[str] = ()
    not_comparable: Sequence[str] = ()
    rules: Sequence[RuleDisclosure] = ()
    statements: Sequence[Statement] = ()
    jurisdiction: str = ""
    tax_year: Optional[int] = None
    output_basis: str = ""
    """Pre-tax, post-withholding, or after-tax. Stated because the three are
    routinely quoted as though interchangeable."""

    @property
    def coverage(self) -> Dict[str, Any]:
        """Declared behaviours that are actually realized.

        Coverage, not confidence. It says how much of what this system claims
        to model it currently carries out, and nothing about whether a figure
        is right.

        `NOT_MODELLED` is excluded from the denominator. A stated boundary is
        not a shortfall — counting "state tax is not modelled" as a coverage
        failure would make a correct, deliberate exclusion look like a defect,
        which is the miscategorisation `Exclusion` exists to prevent. Those are
        reported separately so they stay visible.
        """
        in_scope = [one for one in self.rules
                    if one.enforcement is not Enforcement.NOT_MODELLED]
        excluded = len(self.rules) - len(in_scope)

        if not in_scope:
            return {"realized": 0, "declared": 0, "fraction": None,
                    "out_of_scope": excluded,
                    "note": "no behaviour is claimed within scope for this plan"}

        realized = sum(1 for one in in_scope
                       if one.enforcement is Enforcement.ENFORCED)
        return {"realized": realized, "declared": len(in_scope),
                "fraction": realized / len(in_scope),
                "out_of_scope": excluded,
                "note": ("share of in-scope declared behaviours currently "
                         "realized; coverage of the model, not confidence in "
                         "the figure. Deliberate exclusions are counted "
                         "separately as out_of_scope")}

    def by_enforcement(self, state: Enforcement) -> Sequence[RuleDisclosure]:
        return tuple(one for one in self.rules if one.enforcement is state)

    def to_json(self) -> Dict[str, Any]:
        return {"recognised": list(self.recognised),
                "comparable": list(self.comparable),
                "not_comparable": list(self.not_comparable),
                "rules": [one.to_json() for one in self.rules],
                "statements": [one.to_json() for one in self.statements],
                "jurisdiction": self.jurisdiction, "tax_year": self.tax_year,
                "output_basis": self.output_basis,
                "coverage": self.coverage}


#: Behaviours the account runtimes declare and nothing performs. Named rather
#: than omitted: a reader assumes an unmentioned rule is handled.
ACCOUNT_NOT_MODELLED: Mapping[str, str] = {
    "early withdrawal penalties": "not applied at any age",
    "required minimum distributions": "a long horizon shows no forced withdrawal",
    "state and local tax": "not modelled in any jurisdiction",
    "income phase-out eligibility": "filing status and modified AGI are not "
                                    "represented, so a contribution reduced or "
                                    "disallowed by income is not detected",
}

RSU_NOT_MODELLED: Mapping[str, str] = {
    "final tax liability": "withholding is a statutory remittance rate, not a "
                           "marginal rate",
    "capital-gains lots": "delivered shares are not tracked as tax lots",
    "wash sales": "interactions with other holdings are not modelled",
    "section 83(b) and ESPP": "not modelled",
}


def for_account(declared: str, *, tax_year: int,
                support=None) -> ScopeDisclosure:
    """Disclosure for one account, read from the realization machinery."""
    from ..runtime.account_limits import Enforcement as LimitEnforcement
    from ..runtime.account_limits import RulesetNotFound
    from ..runtime.account_limits import load as load_rules
    from .account_support import support_for
    from .environment import ACCOUNT_KINDS

    support = support or support_for(declared, year=tax_year)
    rules: List[RuleDisclosure] = []
    statements: List[Statement] = []
    comparable: List[str] = []
    not_comparable: List[str] = []

    kind = ACCOUNT_KINDS.get(declared)
    limit = None
    if kind is not None:
        try:
            limit = load_rules(tax_year).limit_for(kind.value)
        except RulesetNotFound:
            limit = None

    if limit is not None and limit.amount is not None:
        state = limit.enforcement(known={})
        rules.append(RuleDisclosure(
            rule="contribution limit",
            enforcement=(Enforcement.ENFORCED
                         if state is LimitEnforcement.ENFORCED
                         else Enforcement.PARTIAL),
            why=limit.why_not_enforced(known={}) or "applied to this account",
            missing_inputs=tuple(limit.missing_inputs({}))))

        if limit.combined_across:
            shared = ", ".join(one.replace("_", " ").title()
                               for one in limit.combined_across)
            rules.append(RuleDisclosure(
                rule="shared limit across related accounts",
                enforcement=Enforcement.PARTIAL,
                why=(f"this limit is shared with {shared}, and only this "
                     "account was described"),
                missing_inputs=tuple(limit.missing_inputs({}))))
            not_comparable.append(
                f"contributions to {shared} were not declared")

    # `unenforced_behaviours` restates in prose what the limit rules above
    # already say. Emitting both puts one fact on the screen twice, the second
    # time as a sentence pretending to be a rule name.
    stated = {one.why for one in rules}
    for behaviour in getattr(support, "unenforced_behaviours", ()):
        text = str(behaviour)
        if text in stated or any(text[:40] in one.why for one in rules):
            continue
        rules.append(RuleDisclosure(
            rule=text.split(",")[0][:60], enforcement=Enforcement.PARTIAL,
            why=text))

    for rule, why in ACCOUNT_NOT_MODELLED.items():
        rules.append(RuleDisclosure(rule=rule,
                                    enforcement=Enforcement.NOT_MODELLED,
                                    why=why))

    if getattr(support, "comparable", None) is not None:
        comparable.extend(["account kind", "contribution year"])

    statements.append(Statement(
        Facet.INPUT, f"{getattr(support, 'label', declared)}, tax year "
                     f"{tax_year}"))
    if not_comparable:
        statements.append(Statement(
            Facet.ASSUMPTION,
            "No other applicable account contributions were declared."))
    statements.append(Statement(
        Facet.OUTPUT,
        "Contribution accepted where within the stated limit; compliance with "
        "the shared limit is not established."
        if not_comparable else
        "Contribution accepted where within the stated limit."))

    return ScopeDisclosure(
        recognised=tuple(filter(None, [getattr(support, "label", declared),
                                       f"tax year {tax_year}"])),
        comparable=tuple(comparable), not_comparable=tuple(not_comparable),
        rules=tuple(rules), statements=tuple(statements),
        jurisdiction="US-federal", tax_year=tax_year,
        output_basis="pre-tax account value; no tax is applied to the figure")


def for_rsu(runtime, *, implemented: Sequence[str] = ()) -> ScopeDisclosure:
    """Disclosure for a vesting plan, read from the runtime's own declarations."""
    unrealized = set(runtime.unrealized(implemented))
    rules = [
        RuleDisclosure(
            rule=one.name.replace("-", " "),
            enforcement=(Enforcement.PARTIAL if one.name in unrealized
                         else Enforcement.ENFORCED),
            why=one.statement)
        for one in runtime.assumptions
    ]
    rules += [RuleDisclosure(rule=rule, enforcement=Enforcement.NOT_MODELLED,
                             why=why)
              for rule, why in RSU_NOT_MODELLED.items()]

    return ScopeDisclosure(
        recognised=("RSU vest", "employer withholding"),
        comparable=("withholding policy", "corporate-action snapshot"),
        not_comparable=(),
        rules=tuple(rules),
        statements=(
            Statement(Facet.INPUT, "a vesting schedule and a withholding rate"),
            Statement(Facet.ASSUMPTION,
                      "The withholding rate is the statutory remittance rate, "
                      "not this person's marginal rate."),
            Statement(Facet.OUTPUT,
                      "Shares delivered to the modelled account after employer "
                      "share withholding."),
        ),
        jurisdiction="US-federal",
        output_basis=("value delivered to the modelled account after employer "
                      "share withholding; not gross compensation and not final "
                      "federal, state or local tax liability"))
