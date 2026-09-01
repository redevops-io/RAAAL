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
from .strategy_methods import STRATEGY_ALLOCATION_METHODS

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
    demonstrably write, which this build will not run.

    A key wrapped in angle brackets names a *shape* rather than a literal —
    `<a split with no holding named>` — for dimensions whose refusal depends on
    the form of the value instead of its content. `decide` matches literals, so
    such a key never fires by lookup; it is here because a dimension that
    refuses something has to say so where every reader of the manifest looks,
    and "executes, refuses nothing" was not true of `stated_weights`."""

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
        if value in self.values:
            return True
        # A parameterised value: `calendar_day:15` is the `calendar_day` rule
        # carrying the day it names. The family is what the manifest declares,
        # because the day is the user's and cannot be enumerated.
        #
        # Only the family is checked here. Whether *this* day is one the engine
        # can honour is the engine's question, and `schedule.day_of_month`
        # answers it — a rule naming day 99 is not executable and must not
        # reach a simulation by passing a prefix test.
        if isinstance(value, str) and ":" in value:
            family, _, _ = value.partition(":")
            if family in self.values:
                from .schedule import day_of_month
                return day_of_month(value) is not None
        return False

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
        # `stated_weights` joins the executable set: the engine divides each
        # purchase by a split whenever the sentence attaches one to its
        # holdings. It was refused while nothing could bind `60/40` to VTI and
        # BND, and that was a fact about the compiler rather than the executor.
        #
        # The computed allocations (risk parity, minimum variance, the momentum
        # and factor families, …) join it too: the research engine already had
        # every one, and `rebalance.strategy_driven` now restores to the weights
        # they compute each period. The refusals here described a compiler that
        # could not route to them, not an executor that could not run them. They
        # carry a rebalancing cadence, so a plan naming one but no calendar is
        # refused at evaluation with the cadence it is missing.
        values=("equal_weight_at_purchase", "stated_weights",
                *sorted(STRATEGY_ALLOCATION_METHODS)),
        refuses={
            "hierarchical_risk_parity":
                "this build has no hierarchical-risk-parity kernel; risk "
                "parity, minimum variance and equal risk contribution are "
                "available and compute a related split",
        }),

    "stated_weights": _d(
        "stated_weights", EXECUTED, closed=False,
        refuses={
            "<a split with no holding named>":
                "a bare ratio such as 60/40 could mean either assignment, and "
                "guessing runs 40/60 under the name 60/40 — a wrong figure "
                "nothing downstream can detect. Naming the holdings, as in "
                "'60% in VTI and 40% in BND', is what makes it executable"},
        why=("each purchase is divided by the stated weights. The split has to "
             "say which holding takes which share — '60% in VTI and 40% in "
             "BND' rather than a bare '60/40', which names no holdings and "
             "could mean either assignment. An unattachable split is refused "
             "by name rather than guessed at")),

    # ---- what happens to holdings -------------------------------------
    # Named as `coverage` names it, so the two range over one vocabulary.
    "periodic_rebalancing": _d(
        "periodic_rebalancing", EXECUTED,
        # The calendar cadences `rebalance.weighted` restores the split on. The
        # dimension was EXECUTED but carried no values, so `executes()` refused
        # every cadence while the executor was quietly able to run them — the
        # stale half of a migration the executor already finished. `threshold_band`
        # stays refused: drift-triggered rebalancing is a different rule on
        # different days. An instruction that states no cadence is still refused
        # rather than defaulted, upstream, because rebalancing sells and picking a
        # frequency would invent a schedule of sales nobody described.
        values=("annual", "quarterly", "monthly", "weekly", "biweekly"),
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

    # The two unsupported strategy families, so a reading that carries one is
    # refused by name rather than compiled from whatever fragment survived.
    #
    # NOT_MODELLED rather than REFUSED, and the distinction is the one this
    # module draws: REFUSED is a decision about something the engine
    # represents, and neither of these has a representation at all. A tilt
    # names a factor where the engine names instruments; an age-based
    # allocation changes over time where the engine holds one allocation for
    # the whole evaluation.
    "factor_tilt": _d(
        "factor_tilt", NOT_MODELLED,
        why="a tilt names a factor or style rather than the holdings to buy, "
            "and this build divides each purchase between named instruments; "
            "executing it as a purchase of whatever the words happened to "
            "name is a different strategy"),

    "age_based_allocation": _d(
        "age_based_allocation", NOT_MODELLED,
        why="the allocation changes with age or elapsed time, and this build "
            "holds one allocation for the whole evaluation; running it as a "
            "fixed split silently drops the thing that made it a glide path"),

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
        "dividend_policy", EXECUTED, values=("reinvested",),
        refuses={
            "held_as_cash":
                "distributions paid out and left uninvested are not modelled: "
                "the engine credits reinvested distributions by running on a "
                "total-return series, and has no path that pays them out and "
                "leaves them idle. Running this as reinvested would report a "
                "different and better-performing strategy"},
        why=("reinvested distributions are credited: the run uses the "
             "snapshot's total-return series, so a plan that reinvests "
             "compounds its position and one that does not is refused rather "
             "than reported as though it had")),

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
    # Open-valued and EXECUTED, not blanket-refused: the engine restricts a
    # replay to a *trailing* window ("the past 5 years") — `time_window.resolve`
    # slices the snapshot to it and the figure is reported over exactly that
    # period. It cannot yet resolve an open-ended ("since 2021"), explicit-range
    # or rolling window, so those are refused *by value* in `decide` — the same
    # shape as `stated_weights`: the boundary is a property of the value, not of
    # the dimension. A dimension that runs for some values and refuses others is
    # not one this manifest may mark REFUSED, or it would refuse the ones that
    # work.
    "evaluation_period": _d(
        "evaluation_period", EXECUTED, closed=False,
        why=("this build restricts a replay to a trailing period — 'the past 5 "
             "years' — and cannot yet resolve an open-ended, explicit-range or "
             "rolling window")),
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

    # A split that names no holding, on a dimension that otherwise executes.
    #
    # `stated_weights` runs — the engine divides each purchase by the weights —
    # and what it cannot do is decide which side of `60/40` is which. That is a
    # property of the *value*, not of the dimension, so it is refused here
    # rather than in the compiler: `executable_check`, the family sweep and the
    # pilot all ask this function, and a rule living in one caller is a rule the
    # others do not have.
    if name == "stated_weights" and value is not None and "=" not in str(value):
        return Refusal(
            kind=UNSUPPORTED_VALUE, dimension=name, stated_value=value,
            detail=f"{str(value)!r} states a split without saying which "
                   "holding takes which share. Writing it as '60% in VTI and "
                   "40% in BND' would let this plan run — this build divides "
                   "each purchase by stated weights, and will not guess which "
                   "way round they go")

    # A stated evaluation window this build cannot resolve.
    #
    # `evaluation_period` runs for a trailing duration — the engine restricts
    # the replay to "the past 5 years" and reports the figure over exactly that
    # period. It cannot resolve an open-ended ("since 2021"), explicit-range or
    # rolling window, and reading one as trailing would answer a different
    # question with a plausible number. So the supported kind runs and the rest
    # are refused here, by value — the same shape as `stated_weights`, a
    # property of the value rather than of the dimension. Placed before the
    # generic `executes` check because the dimension is open-valued, so that
    # check accepts every value and would let an unresolvable window through.
    if name == "evaluation_period" and value is not None:
        from .time_window import from_canonical

        window = from_canonical(str(value))
        if window is not None and window.supported and (window.years or window.months):
            return None
        return Refusal(
            kind=UNSUPPORTED_VALUE, dimension=name, stated_value=value,
            detail=("this build replays a trailing period — say 'the past 5 "
                    "years' or 'the past 6 months' — and cannot yet restrict a "
                    "run to that window; reading it as a trailing period would "
                    "report a figure for a period you did not ask about"))

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
