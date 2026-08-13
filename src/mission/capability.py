"""What this build can actually execute, declared once and checked.

    Discovery may understand more than Mission can execute.
    Mission may never execute less than the verified intent
    while pretending they are equivalent.

This module is the second clause. It is the list of semantic dimensions the
engine executes, the values it executes for each, and — as first-class entries —
the ones it refuses and why.

**Why a manifest at all.** The engine's capabilities used to be implicit,
spread across an ordered regex table, an if-chain in a web route, a vocabulary
dict of menu choices and a renderer's word map. Nothing compared them, so they
drifted, and the drift was silent in the only direction that matters: the
product *offered* `quarterly`, `annual` and `daily` in a confirmation menu,
verbalised them as "every quarter" and "every year", and executed all three as
a single contribution. "$1,000 every year over five years" reported $1,000
contributed. No refusal, no caveat, no coverage flag, and ~3,900 tests covering
no path through the function that turned a schedule into money.

**Derived, not restated.** Where a dimension's executable set is a fact about
a table in the executor, this module imports that table rather than repeating
it (`cadence`, `day_rule`). A restated list is a second copy that can disagree,
and the disagreement is exactly the defect. Where a capability is a fact about
whether a code path exists at all (`stated_weights`, `rebalancing`), the claim
is declared here and `tests/test_capability_manifest.py` proves it by
exercising the engine — a claim nobody exercises is a comment.

**It must not constrain what may be understood.** Nothing upstream should read
this to decide what a user is allowed to mean. A reader taught to see only
executable dimensions does not refuse the others; it renders them as the
nearest thing it can say, which is how "by inverse volatility" becomes an equal
split. The manifest's job is to make a refusal loud, not to make an intent
unspeakable.

Two things may read it: the executor's own gate, and any surface that offers
the user a closed set to choose from. A menu is a promise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from .schedule import (
    EXECUTABLE_CADENCES,
    EXECUTABLE_DAY_RULES,
    REFUSED_CADENCES,
)

#: Bumped when the *shape* of the manifest changes, not when a build's
#: capabilities do. A consumer pins this; the content hash tracks the rest.
MANIFEST_SCHEMA = "quantify/capability-manifest@1"


# Support levels are plain module constants rather than an enum: they are
# consumed by templates and JSON as much as by Python, and an enum
# round-tripping through both grew more conversion code than it saved.
EXECUTED = "EXECUTED"
"""The engine runs this dimension, for the values listed."""

REFUSED = "REFUSED"
"""The engine models this dimension and will not run it. A user who states it
gets a refusal naming it — not a substitution."""

NOT_MODELLED = "NOT_MODELLED"
"""The engine has no representation for it at all. Distinct from REFUSED: one
is a decision, the other is an absence, and telling a user "we don't do that"
should not be confused with "we chose not to"."""


@dataclass(frozen=True)
class Dimension:
    """One semantic dimension, and exactly what this build does with it."""

    name: str
    support: str
    values: Sequence[str] = ()
    """The closed set this build executes. Empty means the dimension is either
    unsupported, or open-valued (a number, a date) — `closed` says which."""

    closed: bool = True
    refuses: Mapping[str, str] = field(default_factory=dict)
    """value -> why. Populated for values the vocabulary offers, or that users
    demonstrably write, which this build will not run."""

    why: str = ""
    """Why the dimension itself is refused or unmodelled. Read by a user."""

    derived_from: str = ""
    """The executor table this entry was read from, or empty when the claim is
    behavioural and proven by test instead."""

    @property
    def is_offerable(self) -> bool:
        """Whether a surface may present this as a choice.

        A menu is a promise: offering a value the engine cannot run is the
        defect this module exists for.
        """
        return self.support == EXECUTED and self.closed and bool(self.values)

    def executes(self, value: Any) -> bool:
        if self.support != EXECUTED:
            return False
        if not self.closed:
            return True
        return value in self.values

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "support": self.support,
                "values": list(self.values), "closed": self.closed,
                "refuses": dict(self.refuses), "why": self.why,
                "derived_from": self.derived_from}


def _d(name, support, **kw) -> Dimension:
    return Dimension(name=name, support=support, **kw)


#: The manifest. One entry per semantic dimension the compiler can produce.
#:
#: Ordered as a reader would ask about them: when money arrives, what it buys,
#: what happens to holdings, over what period.
MANIFEST: Mapping[str, Dimension] = {
    # ---- what the person is trying to find out -----------------------
    #
    # Added after a twenty-family sweep showed the hosted reader gets this
    # right and the manifest then waved it through. "draw down 3% a year in
    # retirement" was read correctly as `assess_withdrawal` and executed,
    # because `decide` treats an unclassified dimension as not forbidden —
    # which is the correct default and was the wrong answer here.
    #
    # Only `assess_withdrawal` is refused. The engine buys and holds, so
    # `sell_action` is already REFUSED, and an objective that requires selling
    # cannot be satisfied by a build whose every mechanism adds to a position.
    # The rest stay executable: refusing `other` would refuse on absence of
    # information rather than on a stated request, and `other` is what a reader
    # says when the sentence does not name an objective at all.
    "objective": _d(
        "objective", EXECUTED,
        values=("evaluate_investment_strategy", "compare_strategies",
                "plan_contributions", "other"),
        refuses={
            "assess_withdrawal":
                "this build only buys, so it cannot assess what taking money "
                "out would do; the contributions and the holdings can be "
                "modelled but the withdrawal itself cannot",
            "assess_conversion":
                "moving money between account types is not modelled: no tax "
                "is computed, and a conversion whose whole effect is a tax "
                "one would be simulated as an ordinary contribution",
            "assess_debt_repayment":
                "the engine values market holdings; a debt has no price "
                "series, so paying one down cannot be compared against "
                "investing without inventing the return it avoids"}),

    # ---- when money arrives ------------------------------------------
    "cadence": _d(
        "cadence", EXECUTED,
        values=EXECUTABLE_CADENCES,
        refuses=dict(REFUSED_CADENCES),
        derived_from="mission.schedule.EXECUTABLE_CADENCES"),

    "day_rule": _d(
        "day_rule", EXECUTED,
        values=EXECUTABLE_DAY_RULES,
        derived_from="mission.schedule.EXECUTABLE_DAY_RULES"),

    "trigger_semantics": _d(
        "trigger_semantics", EXECUTED,
        values=("crossing_event", "persistent_condition"),
        derived_from="mission.signals"),

    "execution_timing": _d(
        "execution_timing", EXECUTED,
        values=("next_session_open",),
        refuses={"same_session_close":
                 "acting on the close that produced the signal reads one bar "
                 "into the future"},
        derived_from="mission.funding.SUPPORTED_TIMING"),

    "amount": _d("amount", EXECUTED, closed=False),

    # The two coarse funding elements `coverage` reports on. They exist
    # alongside the finer dimensions above rather than instead of them:
    # coverage asks "was the funding you declared executed", the manifest asks
    # "can this shape of funding be executed at all", and a plan can fail
    # either question independently.
    "scheduled_funding": _d(
        "scheduled_funding", EXECUTED, closed=False,
        why="money on a calendar; see `cadence` for which calendars"),

    "event_triggered_funding": _d(
        "event_triggered_funding", EXECUTED, closed=False,
        why=("money on a condition; see `trigger_semantics` for how the "
             "condition is read and `execution_timing` for when it fills")),

    "conditional_amount": _d(
        "conditional_amount", REFUSED,
        why=("this build contributes a fixed amount; it does not vary the "
             "amount with the condition that triggered it")),

    # ---- what it buys -------------------------------------------------
    "assets": _d("assets", EXECUTED, closed=False,
                 why="subject to the snapshot actually holding a price series"),

    "allocation_method": _d(
        "allocation_method", EXECUTED,
        values=("equal_weight_at_purchase",),
        refuses={
            "inverse_volatility": "this build allocates equally at purchase",
            "risk_parity": "this build allocates equally at purchase",
            "minimum_variance": "this build allocates equally at purchase",
            "maximum_diversification": "this build allocates equally at purchase",
            "volatility_target": "this build allocates equally at purchase",
            "hierarchical_risk_parity": "this build allocates equally at purchase",
        }),

    "stated_weights": _d(
        "stated_weights", REFUSED,
        why=("a split such as 60/40 is read but not attached to anything: the "
             "sentences that state one usually name no holdings, and `60/40` "
             "with nothing saying which side is which cannot be executed. The "
             "engine divides a purchase by stated weights when it has them — "
             "what is missing is the relation between the numbers and the "
             "holdings, not the arithmetic")),

    # ---- what happens to holdings -------------------------------------
    # Named as `coverage` names it, so the two range over one vocabulary.
    "periodic_rebalancing": _d(
        "periodic_rebalancing", EXECUTED,
        refuses={
            "threshold_band":
                "rebalancing when a weight drifts past a band is not "
                "modelled: this build restores the split on a calendar, and "
                "running a drift rule as an annual one would trade on "
                "different days and produce a different figure",
        },
        why=("restored on a calendar — annual, quarterly, monthly, weekly or "
             "biweekly. An instruction that states no cadence is refused "
             "rather than defaulted: rebalancing sells, and picking a "
             "frequency would invent a schedule of sales nobody described")),

    "selection_rule": _d(
        "selection_rule", NOT_MODELLED,
        why="choosing holdings by ranking them is not modelled: this build "
            "buys what it was told to buy, so a rule that picks a different "
            "holding each period has nothing to act on — and executing it as "
            "a purchase of every candidate is a different strategy"),

    "holding_period": _d(
        "holding_period", NOT_MODELLED,
        why="this build buys and holds to the end of the evaluation period, "
            "so a stated holding length has no effect it could have; honouring "
            "it would require selling, which is not modelled either"),

    "sell_action": _d(
        "sell_action", REFUSED,
        why=("this build only buys; selling, withdrawing and harvesting are "
             "not modelled")),

    "dividend_policy": _d(
        "dividend_policy", REFUSED,
        why=("the engine runs on price series only, so dividends are neither "
             "paid nor reinvested; the choice is recorded and distinguishes "
             "two strategies from each other without changing a figure")),

    # The three families the drift lane found execution-unstable. Discovery
    # can now state them; this is where they stop. Named individually rather
    # than folded into one "unsupported structure" entry because a person told
    # "buckets are not modelled" learns something, and a person told "your
    # sentence contains an unsupported structure" does not.
    "asset_location": _d(
        "asset_location", NOT_MODELLED,
        why="placing particular holdings in particular accounts is not "
            "modelled: this build simulates one pool of holdings and computes "
            "no tax, so where each one sits changes nothing it can calculate"),

    "reserve_policy": _d(
        "reserve_policy", NOT_MODELLED,
        why="money held back from investment is not modelled: the engine "
            "simulates what is invested, and a reserve sized against expenses "
            "has no price series to value it against"),

    "bucket_policy": _d(
        "bucket_policy", NOT_MODELLED,
        why="time-segmented buckets are not modelled: this build holds one "
            "portfolio, and spending from a near-term pot while a long-term "
            "one grows is a withdrawal rule, which it also does not model"),

    "portfolio_sleeves": _d(
        "portfolio_sleeves", NOT_MODELLED,
        why="sleeves with their own allocations or gearing are not modelled: "
            "this build divides each purchase equally between the named "
            "holdings, so a levered or separately-weighted sleeve cannot be "
            "honoured"),

    "account_transition": _d(
        "account_transition", NOT_MODELLED,
        why="moving money between account types is not modelled: no tax is "
            "computed, so a conversion whose entire effect is a tax one would "
            "be simulated as an ordinary contribution"),

    "tax_treatment": _d(
        "tax_treatment", NOT_MODELLED,
        why=("account type is recorded and compared, but no tax is computed: "
             "no rates, no lots, no wash sales, no withdrawal ordering")),

    # ---- over what period ----------------------------------------------
    # Declared EXECUTED and consumed by nothing. The stranded-dimension check
    # found it on the first realistic pilot sentence — "over the past 5 years"
    # was read correctly, carried into the intent, and refused, because no part
    # of `compile_intent` consults it.
    #
    # The engine evaluates over whatever price history it is given;
    # `period_start` and `period_end` are *reported* from the index that came
    # back, not *chosen* from a stated window. Honouring a stated period means
    # selecting the prices, which is an execution-path change — and until that
    # exists, claiming EXECUTED here is the manifest asserting a capability the
    # compiler does not have.
    #
    # Refused by name, so a person asking for five years is told this build
    # evaluates over the data it has rather than being silently given a
    # different window. Queued in docs/Benchmark-Queue.md.
    "evaluation_period": _d(
        "evaluation_period", REFUSED,
        why=("this build evaluates over the whole price history it holds and "
             "cannot yet restrict a run to a stated window; the period you "
             "asked for would be recorded and not honoured, and a figure "
             "labelled with it would be wrong")),
}


# --------------------------------------------------------------------------


def dimension(name: str) -> Optional[Dimension]:
    return MANIFEST.get(name)


def offerable_values(name: str) -> Sequence[str]:
    """The closed set a surface may present for this dimension.

    Empty for anything not executed — and a caller that offers a choice anyway
    is making a promise this build cannot keep.
    """
    d = MANIFEST.get(name)
    return tuple(d.values) if d is not None and d.is_offerable else ()


@dataclass(frozen=True)
class Refusal:
    """A refusal that names what it refused.

    Shaped after `runtime_contracts.CapabilityRefusal`; kept local while that
    package is archived and unreachable as a dependency, and deliberately
    field-compatible so adopting it is a rename.
    """

    kind: str
    dimension: str
    stated_value: Any = None
    executable_values: Sequence[str] = ()
    detail: str = ""

    @property
    def message(self) -> str:
        """What a reader is told. Names the dimension and the value, and says
        what this build could have run instead — without applying it."""
        spoken = self.dimension.replace("_", " ")
        if self.stated_value is None:
            parts = [f"This build does not model {spoken}"]
        else:
            parts = [f"You described {spoken} as {self.stated_value!r}, and "
                     "this build does not execute that"]
        if self.detail:
            parts.append(self.detail)
        if self.executable_values:
            parts.append("it can run: " + ", ".join(
                str(v) for v in self.executable_values))
        return "; ".join(parts) + "."

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind, "dimension": self.dimension,
                "stated_value": self.stated_value,
                "executable_values": list(self.executable_values),
                "detail": self.detail, "message": self.message}


UNSUPPORTED_DIMENSION = "UNSUPPORTED_DIMENSION"
UNSUPPORTED_VALUE = "UNSUPPORTED_VALUE"


def decide(name: str, value: Any = None) -> Optional[Refusal]:
    """Whether this build executes `name` at `value`. `None` means yes.

    An unknown dimension returns `None` deliberately: this manifest describes
    what the engine does, and a dimension nobody has classified is not thereby
    forbidden. The build-time check in `tests/test_capability_manifest.py` is
    what keeps the list complete — failing there is loud, and failing here
    would block a user for a bookkeeping omission.
    """
    d = MANIFEST.get(name)
    if d is None:
        return None

    if d.support in (REFUSED, NOT_MODELLED):
        return Refusal(kind=UNSUPPORTED_DIMENSION, dimension=name,
                       stated_value=value, detail=d.why)

    if value is not None and not d.executes(value):
        return Refusal(
            kind=UNSUPPORTED_VALUE, dimension=name, stated_value=value,
            executable_values=tuple(d.values),
            detail=d.refuses.get(value) or _why_not(name, value, d))
    return None


def _why_not(name: str, value: Any, dimension: "Dimension") -> str:
    """A reason for a value nobody wrote a reason for.

    `refuses` carries the values somebody anticipated. A reader can return one
    nobody did — gpt-5.4 answered `day_rule` with `calendar_first_rolled_forward`,
    which is a perfectly sensible English description of a rule this build does
    not have — and the detail was then the empty string. The page rendered
    "day_rule:" followed by nothing: a refusal by name with no name given for
    the refusing, which is worse than no refusal at all, because the reader is
    told something is wrong and not what.
    
    Everything needed is already on the dimension. Refusing by name means
    saying which name, and a closed set is the most useful thing to say next.
    """
    executable = ", ".join(str(v) for v in dimension.values)
    if executable:
        return (f"{value!r} is not something this build executes for "
                f"{name}. It runs {executable} — stating one of those would "
                "let this plan run")
    return (f"{value!r} was stated for {name} and this build does not execute "
            "it. Nothing is substituted, because a plan that ran on a "
            "different rule would answer a question you did not ask")


def refusals_for(declared: Mapping[str, Any]) -> Sequence[Refusal]:
    """Every refusal a set of declared dimensions earns, in manifest order.

    All of them, not the first. A user who declared three unsupported things
    should be told three times rather than discovering them one deploy apart.
    """
    out = []
    for name in MANIFEST:
        if name in declared:
            refusal = decide(name, declared[name])
            if refusal is not None:
                out.append(refusal)
    return tuple(out)


def manifest_json() -> Dict[str, Any]:
    return {"schema": MANIFEST_SCHEMA,
            "dimensions": {k: v.to_json() for k, v in MANIFEST.items()}}
