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
                implemented: Sequence[str] = ACCOUNT_IMPLEMENTED
                ) -> AccountSupport:
    """The three states for one declared account, all derived.

    `implemented` is injected so a test can assert the states move when
    realizations arrive — a derivation nobody has seen change is a derivation
    nobody has tested.
    """
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

    runtime = AccountRuntime(name=f"account/{kind.value.lower()}", version=1,
                             account_kind=kind)
    declared_behaviours = tuple(a.name for a in runtime.assumptions)
    unenforced = tuple(runtime.unrealized(implemented))

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
    if not unenforced:
        enforced = Support.YES
    elif len(unenforced) < len(declared_behaviours):
        enforced = Support.PARTIAL
    else:
        enforced = Support.NO

    return AccountSupport(
        declared=declared, label=label,
        recognized=Support.YES, comparable=Support.YES, enforced=enforced,
        declared_behaviours=declared_behaviours,
        unenforced_behaviours=unenforced)
