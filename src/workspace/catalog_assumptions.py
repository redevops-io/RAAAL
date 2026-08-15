"""What the catalogue supplies for what its sentences do not say.

Measured over all 43 offered strategies: 3 ran, 37 asked a question first, 3
refused. The dominant question was `assets`, asked 22 times — and in all 22 the
sentence genuinely names no holding. "I withdraw $20,000 from the portfolio each
year" does not say what the portfolio holds, and a human advisor would ask too.

So the reading was right and the *interaction* was wrong. Somebody picks a
withdrawal rule to model withdrawals and has to invent a portfolio before the
question they came with can be answered. The picked strategy is structured
evidence — the product knows which family it offered — and a family knows what
a reasonable stand-in portfolio looks like.

**An assumption is not a fact, and the difference is the whole design.** These
values are authored `DEFAULT`, whose contract says "nobody asserted it; a
declared default applied — the value a consumer is most entitled to question,
and the one most often mistaken for a choice". They are never authored `USER`.
`Author.USER` dominates every other author and is never overwritten by a
re-read, so recording a guess there would make the product offer its own
assumption back as the user's choice, permanently and invisibly.

**Confirming one changes its authority, not its history.** A person who edits or
accepts an assumed value settles it as `USER` — and the earlier
`CATALOG_ASSUMED` entry stays in the settled record. The plan can then say both
things that are true: this value is now the user's, and it did not start that
way.

**Only what is incidental to the strategy is assumed.** A withdrawal rule is
about the withdrawals; the portfolio it draws from is scenery, and supplying it
lets the rule be seen. Nothing here supplies a value the strategy *is* — no
assumption invents a withdrawal ordering for a strategy about ordering, or a
rebalancing band for a strategy about bands. That line is why the assumed set is
small and why it is per-family rather than global.

Assumed values are drawn from what the engine can actually run, so an assumption
cannot hand somebody a refusal they did not ask for. `tests/` checks that.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: How an assumed value is spelled in the settled record. Distinct from
#: `USER_ANSWERED` so a reader of a stored plan can tell a supplied value from a
#: chosen one without consulting anything else.
CATALOG_ASSUMED = "CATALOG_ASSUMED"

#: And after somebody accepts one. Distinct from `USER_ANSWERED` too: both are
#: the user's, and only one of them started as our guess.
CATALOG_CONFIRMED = "CATALOG_CONFIRMED"


@dataclass(frozen=True)
class Assumption:
    """One value the catalogue supplies, and why this family gets it."""

    dimension: str
    value: str
    because: str
    """Shown beside the value. Not "a default" — the reason has to be specific
    enough that somebody can tell whether it suits them, because the whole point
    is that they can change it."""

    @property
    def detail(self) -> str:
        return f"assumed by the catalogue, not stated: {self.because}"


def _portfolio(what: str) -> Tuple[Assumption, ...]:
    return (
        Assumption("assets", "VTI and BND",
                   f"{what} needs something to hold, and the sentence names "
                   "no holding — a total-market fund and a bond fund stand in "
                   "for an ordinary portfolio"),
    )


#: By catalogue group, because a family is what knows the answer. A global
#: default would supply a contribution amount to a withdrawal strategy and a
#: withdrawal to a contribution one.
#:
#: **Two obvious candidates are deliberately absent**, and they are the reason
#: this list is short. `stated_weights = 60% VTI and 40% BND` reads like the
#: most natural assumption in the file, and supplying it earned a refusal
#: naming `stated_weights` in every family that used it — this build allocates
#: equally at purchase. `periodic_rebalancing = once a year` did the same. Both
#: would have handed somebody a refusal they did not ask for, in place of a
#: question they could have answered.
#:
#: That is the rule `test_catalog_assumptions` enforces: an assumption may
#: never cause a refusal naming the dimension it supplies. Refusals on *other*
#: dimensions are welcome and are the point — supplying a portfolio lets the
#: engine reach "this build only buys; withdrawing is not modelled" and say so
#: by name, instead of asking for a holding and refusing afterwards.
ASSUMPTIONS: Mapping[str, Tuple[Assumption, ...]] = {
    "money-in": _portfolio("a contribution schedule") + (
        Assumption("amount", "$500",
                   "a round monthly contribution, so the schedule has a scale; "
                   "the shape of the result does not depend on it"),
        Assumption("cadence", "monthly",
                   "the commonest contribution rhythm, and the one payroll "
                   "follows"),
    ),
    "allocation": _portfolio("an allocation rule") + (
        Assumption("amount", "$10,000",
                   "an opening balance for the allocation to apply to, since "
                   "an allocation rule describes proportions rather than sums"),
    ),
    "money-out": _portfolio("a withdrawal rule"),
    "accounts": _portfolio("an account strategy"),
    "other": _portfolio("this strategy") + (
        Assumption("amount", "$10,000",
                   "a starting sum, since the sentence describes a rule rather "
                   "than an amount"),
    ),
}


def for_group(group_key: str) -> Tuple[Assumption, ...]:
    return ASSUMPTIONS.get(group_key, ())


def group_of(entry_key: str) -> str:
    """Which family an offered entry belongs to.

    Read from the library rather than passed in from the page: the page carries
    what was clicked, and a page that also carried the family could disagree
    with the catalogue about which family that was.
    """
    from .strategy_library import LIBRARY

    for group in LIBRARY:
        for entry in group.entries:
            if entry.key == entry_key:
                return group.key
    return ""


def applicable(reading, entry_key: str) -> Tuple[Assumption, ...]:
    """The assumptions that would settle a question this reading actually asked.

    Never everything the family declares. An assumption for a dimension the
    sentence already states would overwrite what somebody said with what we
    guessed — which is the one thing this must not do, and the reason this
    filters on `questions` rather than on the family alone.
    """
    asked = set(getattr(reading, "questions", ()) or ())
    if not asked:
        return ()
    return tuple(one for one in for_group(group_of(entry_key))
                 if one.dimension in asked)


def assume(reading, entry_key: str):
    """Settle what the family can stand in for, leaving the rest asked.

    Returns the reading unchanged when nothing applies, so a caller does not
    have to decide whether it was worth calling — and so a strategy whose open
    questions are all substantive still asks them.
    """
    from .pilot import settle
    from runtime_contracts import Author

    supply = applicable(reading, entry_key)
    if not supply:
        return reading
    return settle(
        reading, {one.dimension: one.value for one in supply},
        author=Author.DEFAULT, provenance=CATALOG_ASSUMED,
        witness="catalogue",
        detail={one.dimension: one.detail for one in supply})


def confirm(reading, values: Mapping[str, Any]):
    """A person accepts or edits what was assumed.

    Authored `USER` from here on, which is what makes it authoritative. The
    earlier `CATALOG_ASSUMED` entry is not removed — `settle` appends — so the
    record still says the value did not begin as theirs. Rewriting history to
    make a confirmed assumption look like an original statement would destroy
    the only evidence that the product guessed.
    """
    from .pilot import settle
    from runtime_contracts import Author

    return settle(reading, values, author=Author.USER,
                  provenance=CATALOG_CONFIRMED, witness="user",
                  detail="confirmed on the plan page, having been assumed")


def assumed_in(reading) -> Sequence[str]:
    """Dimensions this reading holds on the catalogue's word alone.

    A dimension that was assumed and later confirmed is not listed: the settled
    record carries both entries and the later one is the user's. Reading only
    the first would keep warning about a value somebody has already accepted.
    """
    settled = list(getattr(reading, "settled", ()) or ())
    latest: Dict[str, str] = {}
    for one in settled:
        latest[one.field] = one.provenance
    return tuple(sorted(name for name, provenance in latest.items()
                        if provenance == CATALOG_ASSUMED))


def describe(reading) -> Optional[str]:
    """One sentence for the result, or nothing to say.

    Placed with the figure rather than only in the table, because the table is
    above the result and a person reading a number has already scrolled past it.
    A figure resting on values nobody chose has to say so where the figure is.
    """
    names = assumed_in(reading)
    if not names:
        return None
    spelled = ", ".join(name.replace("_", " ") for name in names)
    return (f"This uses {len(names)} value{'s' if len(names) > 1 else ''} the "
            f"catalogue assumed rather than read from the sentence: {spelled}. "
            "Change any of them above and run it again.")
