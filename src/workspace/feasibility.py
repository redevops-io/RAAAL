"""Whether a confirmed scenario can actually execute.

    READY                        every named holding has price history
    READY_WITH_LIMITATIONS       something optional is missing; the analysis runs
    BLOCKED_MISSING_MARKET_DATA  nothing can execute

The distinction this exists for: a Roth plan described in VOO saved
successfully with zero runs and no worksheet. The confirmation screen said
"no price history" *and* "Ready to save", so the caveat was shown and the plan
was committed anyway — the user got a plan page with nothing on it and no
explanation of why.

**A missing benchmark is a limitation. A missing holding is a refusal.** The
first still answers the question that was asked; the second means there is no
question left to answer. Treating them the same is how "Ready" came to mean
"the description parsed", which is not what a reader takes it to mean.

One implementation, consulted by both the screen and the save path. Two would
be two opinions about whether a plan can run, and the one that mattered would
be whichever the save path happened to hold.
"""
from __future__ import annotations

from . import historical_lots

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Sequence, Tuple


class ItemState(str, Enum):
    """What the user is looking at, and whether they can act on it.

    Six states, not one list. The confirmation screen rendered every open item
    identically, so "which account is this in?" — which the user can answer in
    one click — sat beside "forward projection is not modelled", which they
    cannot answer at all, in the same grey block with the same wording. A
    reader cannot tell progress from a wall.

    The Run button is a function of three of these being empty, so the
    interaction has a gradient rather than a verdict.
    """

    RECOGNIZED = "recognized"
    """Understood from the description. Nothing to do."""

    NEEDS_CONFIRMATION = "needs_confirmation"
    """The system chose something. One click to accept or change."""

    NEEDS_ANSWER = "needs_answer"
    """The user knows this and the system does not. One click or one value."""

    NEEDS_CAPABILITY = "needs_capability"
    """Quantify does not model it. The user cannot answer it, and saying
    "please answer" of something unanswerable is what makes a page feel like a
    rejection. They may be able to proceed without it."""

    BLOCKED = "blocked"
    """Nothing can run: no priceable instrument, an existing holding, a
    forward projection that is the whole request."""

    READY = "ready"


class Resolution(str, Enum):
    """What a user may do about one thing the compiler could not settle."""

    REQUIRED_CLARIFICATION = "REQUIRED_CLARIFICATION"
    """The engine cannot determine a required input — an amount, a cadence, an
    account. The user must answer; there is nothing to proceed without."""

    UNSUPPORTED_SEPARABLE = "UNSUPPORTED_SEPARABLE"
    """A phrase the compiler cannot represent, which can be excluded while the
    rest of the scenario still answers the question that was asked."""

    MATERIAL = "MATERIAL"
    """Excluding it would change the question enough that the result could
    mislead — the projection when the projection *is* the request, or an
    exclusion that leaves nothing executable. Blocked, pending revision."""


#: Fields whose absence the engine cannot work around. Enumerated rather than
#: inferred from the question text: a required input misclassified as separable
#: would let a user dismiss the one thing the result depends on.
REQUIRED_FIELDS = frozenset({
    "amount", "cadence", "account_type", "starting_capital", "weighting",
    "trigger_semantics", "funding_source", "benchmark_set",
    "contribution_day_rule", "dividends",
})

#: Phrases that describe the *whole* request rather than a part of it. Excluding
#: one leaves a plan that answers a different question than the user asked, so
#: they are material even though they look separable.
#:
#: Deliberately narrow. The first version included "compare" and "comparison",
#: which blocked any description using the word — including "compare the two
#: account results", where comparison is what the product does. A marker list
#: broad enough to catch every material phrase is a list that refuses ordinary
#: requests, and a refusal a user cannot act on is worse than a limitation
#: they can see: this is the discriminating-strictness rule applied to prose.
#:
#: What remains are the genuinely forward-looking asks. Quantify replays
#: history; a projection is not a part of the question that can be set aside,
#: it is a different question.
_MATERIAL_MARKERS = ("projection", "projected", "forecast", "future value",
                     "after-tax", "after tax")


class Feasibility(str, Enum):
    READY = "READY"
    READY_WITH_LIMITATIONS = "READY_WITH_LIMITATIONS"
    BLOCKED_MISSING_MARKET_DATA = "BLOCKED_MISSING_MARKET_DATA"


@dataclass(frozen=True)
class OpenItem:
    """One unsettled thing, and what the user is permitted to do about it."""

    field: str
    question: str
    why_it_matters: str
    resolution: Resolution
    #: Whether the plan can produce a result at all. Carried on the item so
    #: that "you cannot set this aside" and "this is not something we model"
    #: stay separate reasons.
    executable: bool = True

    @property
    def dismissible(self) -> bool:
        # Nothing may be set aside while the plan cannot run at all: the
        # result of dismissing every item would still be no result.
        return (self.resolution is Resolution.UNSUPPORTED_SEPARABLE
                and self.executable)

    @property
    def state(self) -> "ItemState":
        """Which of the six this item is in."""
        # Deliberately not "blocked because the plan is". Whether the plan can
        # run is a property of the plan and belongs in one banner; folding it
        # into every item is what made "which account is this in?" read as an
        # unmodelled capability. An item is what it is, and the user can
        # answer it now so that the plan is ready the moment the blocker
        # clears.
        if self.resolution is Resolution.REQUIRED_CLARIFICATION:
            return ItemState.NEEDS_ANSWER
        if self.resolution is Resolution.UNSUPPORTED_SEPARABLE:
            return ItemState.NEEDS_CAPABILITY
        return ItemState.BLOCKED

    @property
    def answerable(self) -> bool:
        """Whether supplying a value here would move the plan forward.

        A required field the user can answer, as against a phrase the
        compiler could not place. The confirmation screen renders an input for
        the first and not the second — asking a question with no way to answer
        it is what forces somebody back to rewriting their description.
        """
        return self.resolution is Resolution.REQUIRED_CLARIFICATION

    @property
    def label(self) -> str:
        """What the acknowledgement says, in the user's terms.

        Named here rather than in the template, because "continue without
        modelling this" is a statement about what the result will mean and the
        screen must not be the thing that decides it.
        """
        return f"Continue without modelling {self.subject}"

    @property
    def subject(self) -> str:
        if self.field.startswith("unclear:"):
            return self.field[len("unclear:"):]
        if self.field.startswith(historical_lots.HISTORICAL_LOTS_NOT_AVAILABLE):
            # The code is for an operator; the user gets a sentence. Rendering
            # the field verbatim put "historical lots not available:units
            # held" on the page as though it were English.
            return "an existing holding"
        return self.field.replace("_", " ")

    def to_json(self) -> dict:
        return {"field": self.field, "question": self.question,
                "why_it_matters": self.why_it_matters,
                "resolution": self.resolution.value,
                "dismissible": self.dismissible, "subject": self.subject}


def classify(unresolved: Sequence[Any], *, executable: bool = True
             ) -> Tuple[OpenItem, ...]:
    """Sort what is outstanding into what a user may act on.

    The rule the pilot needs: ordinary extra prose must not create a dead end,
    and acknowledging it must not mean pretending it was understood. A phrase
    the compiler could not place is separable; a required input is not; and
    anything that describes the whole request is material because excluding it
    answers a different question.
    """
    items = []
    for one in unresolved:
        field = getattr(one, "field", "")
        question = getattr(one, "question", "")
        why = getattr(one, "why_it_matters", "")
        items.append(OpenItem(field, question, why,
                              _resolution_for(field, executable),
                              executable=executable))
    return tuple(items)


def _resolution_for(field: str, executable: bool) -> Resolution:
    # A plan that cannot run does not turn its open questions into unmodelled
    # capabilities. This returned MATERIAL for everything when `executable`
    # was false, so a missing price for SPX made "how much are you
    # contributing?" read as "a capability Quantify does not currently model".
    # Amount is modelled. The plan simply had nothing to price.
    #
    # Non-dismissibility while unrunnable is still correct, and is now carried
    # by `dismissible` rather than by rewriting what the item *is*.
    if field.startswith("unclear:"):
        subject = field[len("unclear:"):].lower()
        if any(marker in subject for marker in _MATERIAL_MARKERS):
            return Resolution.MATERIAL
        return Resolution.UNSUPPORTED_SEPARABLE
    if field.startswith("asset_identity:"):
        return Resolution.REQUIRED_CLARIFICATION
    if field in REQUIRED_FIELDS:
        return Resolution.REQUIRED_CLARIFICATION
    # Unknown fields are treated as required. Failing towards "you must answer"
    # is the safe direction: a dismissible classification is a permission, and
    # a permission granted by omission is one nobody decided to give.
    return Resolution.REQUIRED_CLARIFICATION


@dataclass(frozen=True)
class Verdict:
    state: Feasibility
    executable: Tuple[str, ...] = ()
    unavailable: Tuple[str, ...] = ()

    @property
    def can_execute(self) -> bool:
        return self.state is not Feasibility.BLOCKED_MISSING_MARKET_DATA

    @property
    def detail(self) -> str:
        """What to tell the user. Names the instruments, because "no data" sends
        them back to a description they cannot debug."""
        if self.state is Feasibility.BLOCKED_MISSING_MARKET_DATA:
            if not self.unavailable:
                return ("This plan names no instrument that can be priced, so "
                        "there is nothing to simulate.")
            names = ", ".join(self.unavailable)
            return (f"There is no price history for {names}, so this plan "
                    "cannot be simulated. Saving it would store a plan with no "
                    "result. Try a different instrument, or remove it.")
        if self.state is Feasibility.READY_WITH_LIMITATIONS:
            return (f"No price history for {', '.join(self.unavailable)}. The "
                    "rest of the plan still runs, and those holdings are "
                    "excluded from the result.")
        return ""

    def to_json(self) -> dict:
        return {"state": self.state.value, "executable": list(self.executable),
                "unavailable": list(self.unavailable), "detail": self.detail}


def assess(scenario: Any, frame: Optional[Any]) -> Verdict:
    """Judge one compiled scenario against the data actually delivered.

    `frame` is the resolved market data, or `None` when the gate returned
    nothing at all — a denied policy, an unresolvable snapshot. That is blocked
    too: a plan cannot be ready to run when no data was delivered to run it
    against.
    """
    assets = tuple(_assets_of(scenario))
    if not assets:
        return Verdict(Feasibility.BLOCKED_MISSING_MARKET_DATA)

    if frame is None:
        return Verdict(Feasibility.BLOCKED_MISSING_MARKET_DATA,
                       unavailable=assets)

    available = set(frame.columns)
    executable = tuple(one for one in assets if one in available)
    unavailable = tuple(one for one in assets if one not in available)

    if not executable:
        return Verdict(Feasibility.BLOCKED_MISSING_MARKET_DATA,
                       unavailable=unavailable)
    if unavailable:
        return Verdict(Feasibility.READY_WITH_LIMITATIONS, executable,
                       unavailable)
    return Verdict(Feasibility.READY, executable)


def _assets_of(scenario: Any) -> Sequence[str]:
    rule = getattr(scenario, "allocation_rule", None)
    return tuple(getattr(rule, "assets", ()) or ())


#: What Quantify does not do, in the user's terms. A material blocker that says
#: only "this is not supported" sends someone back to a description they cannot
#: debug; naming the capability tells them what to change.
_CAPABILITY_NOTES = (
    (("projection", "project", "forecast", "20-year", "future"),
     "Quantify replays historical periods. It does not produce forward "
     "balance projections."),
    (("percentage", "%", "70/20/10", "allocation percentages"),
     "Quantify allocates by instrument, not by fixed percentage weights."),
    (("employer match", "matching"),
     "Employer matching contributions are not modelled as a separate funding "
     "source."),
    (("after-tax", "after tax", "tax outcome"),
     "After-tax outcomes are not modelled for this scenario type."),
)


def capability_note(subject: str) -> str:
    lowered = subject.lower()
    for markers, note in _CAPABILITY_NOTES:
        if any(marker in lowered for marker in markers):
            return note
    return ("This part of the description has no equivalent in the model, and "
            "removing it would change what the result answers.")


@dataclass(frozen=True)
class Blockers:
    """Everything standing between a submission and a saved plan.

    Produced once, from the same classification the confirmation screen
    renders, and consumed by both the screen and the refusal. Rebuilding the
    explanation in the route would be a second opinion about why a plan cannot
    be saved, and the one the user sees would be whichever the route held.
    """

    material: Tuple[OpenItem, ...] = ()
    required: Tuple[OpenItem, ...] = ()
    unconfirmed: Tuple[str, ...] = ()
    separable: Tuple[OpenItem, ...] = ()
    """Dismissible items the user has *not* dismissed. They still block —
    `is_complete` counts every unresolved item — and the first version of this
    left them out, so this reported "nothing blocking" while the store refused
    the save. Two checks disagreeing about whether a plan can be saved is the
    failure this module was written to prevent, reproduced inside it."""

    @property
    def any(self) -> bool:
        return bool(self.material or self.required or self.unconfirmed
                    or self.separable)

    def detail(self) -> str:
        """The refusal a user reads. Names each item and what can be done."""
        parts = ["This plan cannot be saved yet."]
        if self.material:
            parts.append("\nUnsupported and material:")
            for one in self.material:
                parts.append(f"  - \u201c{one.subject}\u201d")
                parts.append(f"    {capability_note(one.subject)}")
                parts.append("    Remove or revise this requirement.")
        if self.required:
            parts.append("\nRequired clarification:")
            for one in self.required:
                parts.append(f"  - {one.question or one.subject}")
        if self.separable:
            parts.append("\nNot yet decided — tick to continue without "
                         "modelling, or revise the description:")
            for one in self.separable:
                parts.append(f"  - \u201c{one.subject}\u201d")
        if self.unconfirmed:
            parts.append("\nUnconfirmed assumption:")
            for field in self.unconfirmed:
                parts.append(f"  - {field.replace('_', ' ')}")
        return "\n".join(parts)

    def to_json(self) -> dict:
        return {"material": [one.to_json() for one in self.material],
                "required": [one.to_json() for one in self.required],
                "unconfirmed": list(self.unconfirmed),
                "separable": [one.to_json() for one in self.separable],
                "detail": self.detail()}


def blockers(scenario: Any, *, executable: bool = True,
             stated_text: str = "") -> Blockers:
    """What is outstanding, sorted by what the user can do about it.

    `stated_text` is read for described holdings the compiler has no field
    for. Without it, "I already own 500 shares of AAPL that I bought in 2019
    at $50" produced no material blocker at all and the holding was silently
    dropped; a description mentioning a past purchase was asked how much it
    was *starting with*, folding a share count into a cash amount.
    """
    provenance = scenario.provenance
    items = list(classify(provenance.unresolved, executable=executable))

    text = stated_text or getattr(scenario, "stated_text", "") or ""
    for signal in historical_lots.detect(text):
        items.append(OpenItem(
            field=f"{historical_lots.HISTORICAL_LOTS_NOT_AVAILABLE}:{signal.matched}",
            question=signal.question,
            why_it_matters=signal.why_it_matters,
            # Material, and therefore not dismissible. An existing holding is
            # not extra prose alongside the request — it is a claim about what
            # the figure covers, and excluding it answers a question the user
            # did not ask.
            resolution=Resolution.MATERIAL,
            executable=executable))

    items = tuple(items)
    return Blockers(
        material=tuple(one for one in items
                       if one.resolution is Resolution.MATERIAL),
        required=tuple(one for one in items
                       if one.resolution is Resolution.REQUIRED_CLARIFICATION),
        separable=tuple(one for one in items
                        if one.resolution is Resolution.UNSUPPORTED_SEPARABLE),
        unconfirmed=tuple(one.field for one in provenance.unconfirmed))


class NotExecutable(ValueError):
    """A plan was submitted that cannot produce a result.

    Raised by the save path rather than reported by the screen alone. A screen
    that warns and a route that accepts is a caveat, and the whole point of
    this module is that the state it describes was reachable while the warning
    was on display.
    """
