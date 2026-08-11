"""Three claims about an account, kept apart.

    RECOGNIZED   the compiler can read it and represent it
    COMPARABLE   a run pins a versioned account runtime, so comparisons can
                 evaluate the dimension rather than skip it
    ENFORCED     the declared behaviour actually executes

A single "supported" badge collapses three materially different facts. For a
Roth 401(k) today the honest answer is yes, yes, no — and the third is the one a
user would assume from the first two.

`ENFORCED` is **derived from realization checks**, never from a maintained
support list. The project has been caught twice by a list that had to be
remembered: `ISOLATION_DIMENSIONS` was hand-curated and wrong, and the compiler's
account vocabulary drifted from the runtime's. A third list would drift too.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..runtime import ACCOUNT_IMPLEMENTED, AccountRuntime
from .environment import ACCOUNT_KINDS


class Support(str, Enum):
    YES = "YES"
    NO = "NO"
    PARTIAL = "PARTIAL"
    """Some declared behaviour executes and some does not. Distinct from NO,
    because "nothing works" and "half of it works" lead somewhere different."""


@dataclass(frozen=True)
class AccountSupport:
    """What the system can honestly say about one account type."""

    declared: str
    """What the compiler read, e.g. `ROTH_401K`, or empty if it read nothing."""

    label: str
    recognized: Support
    comparable: Support
    enforced: Support

    declared_behaviours: Sequence[str] = ()
    unenforced_behaviours: Sequence[str] = ()

    @property
    def summary(self) -> str:
        """The sentence a user reads. Three claims, never one."""
        if self.recognized is Support.NO:
            return "not recognised, so nothing about it is modelled"
        parts = [f"identified as {self.label}",
                 ("pinned across comparisons"
                  if self.comparable is Support.YES
                  else "not pinned, so comparisons cannot evaluate it")]
        if self.enforced is Support.YES:
            parts.append("its rules are enforced")
        elif self.enforced is Support.PARTIAL:
            parts.append("some of its rules are enforced")
        else:
            parts.append("its rules are recorded and not yet enforced")
        return "; ".join(parts)

    def to_json(self) -> Dict[str, Any]:
        return {
            "declared": self.declared, "label": self.label,
            "recognized": self.recognized.value,
            "comparable": self.comparable.value,
            "enforced": self.enforced.value,
            "declared_behaviours": list(self.declared_behaviours),
            "unenforced_behaviours": list(self.unenforced_behaviours),
            "summary": self.summary,
        }


LABELS: Mapping[str, str] = {
    "TAXABLE": "Taxable brokerage account",
    "ROTH": "Roth IRA",
    "ROTH_401K": "Roth 401(k)",
    "TRADITIONAL_IRA": "Traditional IRA",
    "TRADITIONAL_401K": "401(k)",
}


def support_for(declared: str,
                implemented: Sequence[str] = ACCOUNT_IMPLEMENTED,
                *, year: Optional[int] = None) -> AccountSupport:
    """The three states for one declared account, all derived.

    `implemented` is injected so a test can assert the states move when
    realizations arrive — a derivation nobody has seen change is a derivation
    nobody has tested.
    """
    import datetime as _dt

    from ..runtime.account_limits import Enforcement, RulesetNotFound
    from ..runtime.account_limits import load as load_rules

    year = year or _dt.date.today().year
    if not declared or declared == "NONE_APPLIED":
        return AccountSupport(declared=declared or "", label="none stated",
                              recognized=Support.NO, comparable=Support.NO,
                              enforced=Support.NO)

    kind = ACCOUNT_KINDS.get(declared)
    label = LABELS.get(declared, declared.replace("_", " ").title())
    if kind is None:
        # Read from the text and not representable as a runtime. Recognised,
        # and nothing beyond that is true.
        return AccountSupport(declared=declared, label=label,
                              recognized=Support.YES, comparable=Support.NO,
                              enforced=Support.NO)

    # The governing figure comes from the ruleset pinned to the tax year, so an
    # account whose limit this system actually applies declares it. Constructed
    # without one, every account looked as though it had no contribution rule to
    # enforce, which is why nothing here ever moved off NO.
    try:
        limit = load_rules(year).limit_for(kind.value)
    except RulesetNotFound as absent:
        return AccountSupport(
            declared=declared, label=label, recognized=Support.YES,
            comparable=Support.YES, enforced=Support.NO,
            unenforced_behaviours=(str(absent).split(".")[0],))

    runtime = AccountRuntime(name=f"account/{kind.value.lower()}", version=1,
                             account_kind=kind,
                             annual_contribution_limit=limit.amount)
    declared_behaviours = tuple(a.name for a in runtime.assumptions)
    unenforced = list(runtime.unrealized(implemented))

    # A mechanism that runs against an unchecked number, or against a limit
    # shared with accounts nobody described, is not enforcement. It is
    # enforcement-shaped, which is worse: the display says the limit is applied
    # and nothing says what the check could not see.
    #
    # `known` is empty because a compiled scenario describes one account. That
    # is the point — the reason is derived from what the scenario actually
    # carries, not asserted.
    figure_caveat = None
    if "contribution-limit" in declared_behaviours \
            and "contribution-limit" not in unenforced:
        if limit.enforcement(known={}) is not Enforcement.ENFORCED:
            figure_caveat = limit.why_not_enforced(known={})

    missing_mechanisms = len(unenforced)
    if figure_caveat:
        unenforced.append(figure_caveat)
    unenforced = tuple(unenforced)
    unverified_figure = figure_caveat is not None

    if not declared_behaviours:
        # Nothing declared is not the same as nothing to declare. A Roth IRA has
        # a contribution limit whether or not this runtime instance names one,
        # so reporting YES here would derive certainty from an absence — the
        # exact move the rest of this system refuses.
        return AccountSupport(
            declared=declared, label=label, recognized=Support.YES,
            comparable=Support.YES, enforced=Support.NO,
            declared_behaviours=(),
            unenforced_behaviours=("no account rules are declared for this kind "
                                   "yet, so none can be enforced",))
    # Derived from missing *mechanisms* only. An unchecked figure is a separate
    # objection: it cannot make a working mechanism absent, and it must not be
    # counted as one, or an account with a single rule would report NO when that
    # rule runs.
    if missing_mechanisms == 0:
        enforced = Support.PARTIAL if unverified_figure else Support.YES
    elif missing_mechanisms < len(declared_behaviours):
        enforced = Support.PARTIAL
    else:
        enforced = Support.NO

    return AccountSupport(
        declared=declared, label=label,
        recognized=Support.YES, comparable=Support.YES, enforced=enforced,
        declared_behaviours=declared_behaviours,
        unenforced_behaviours=unenforced)
