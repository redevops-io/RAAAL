"""Every material thing a plan declares, and whether execution reached it.

    declared → compiled → executed → evidenced

A run may publish a figure only when every material declared element is in one
of two terminal states:

    EXECUTED_AND_EVIDENCED
    EXCLUDED_BY_USER

Anything else blocks the result.

**Why one control rather than three guards.** A recorder found three prompts
returning an identical figure — $25,000 contributed, $103,393 final — for three
materially different plans. A three-month window was replayed over ten years, a
monthly $500 contribution vanished, and a sell leg vanished. Each was a
separate omission and all three had the same shape:

    declared → stored correctly → never reached execution → nothing said so

Guards written per element would have been three fixes for one defect, and the
fourth element to arrive would have shipped without a fourth guard. This asks
the general question instead: for everything the plan declares, did execution
reach it, and can the run prove it.

**Absence of evidence blocks.** An element whose execution cannot be evidenced
is not treated as executed. That is deliberately strict: the failure being
closed is precisely a system that believed it had done something it had not,
and a coverage record that trusted the same layer would inherit the belief.

The four verification classes this codebase has collected all apply to the
record itself. `evidenced` must come from a different place than `executed` —
a ledger row, a sliced frame's own bounds — or the record is the executor's
opinion of its own work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class ElementState(str, Enum):
    EXECUTED_AND_EVIDENCED = "EXECUTED_AND_EVIDENCED"
    """Ran, and something other than the runner says so."""

    EXCLUDED_BY_USER = "EXCLUDED_BY_USER"
    """Declared, not executed, and the user agreed to proceed without it."""

    DECLARED_NOT_EXECUTED = "DECLARED_NOT_EXECUTED"
    """The state that produced three identical figures for three plans."""

    EXECUTED_NOT_EVIDENCED = "EXECUTED_NOT_EVIDENCED"
    """The runner believes it ran and can show nothing. Blocks, because this
    is the belief the whole record exists to stop being taken on trust."""

    NOT_DECLARED = "NOT_DECLARED"
    """Absent from the plan. Not a gap — most plans declare few of these."""


#: States a publishable run may hold. Everything else blocks the figure.
TERMINAL = frozenset({ElementState.EXECUTED_AND_EVIDENCED,
                      ElementState.EXCLUDED_BY_USER,
                      ElementState.NOT_DECLARED})


@dataclass(frozen=True)
class DeclaredElement:
    """One thing the plan says, and what became of it."""

    element_id: str
    declared: bool
    compiled: bool
    executed: bool
    evidenced: bool
    exclusion_authorized: bool = False
    reason: str = ""
    """Why it is in the state it is in, in a sentence a user can read. This is
    the text a refusal quotes, so it names the element rather than the
    subsystem: "the three-month period you asked for", not "time_window"."""

    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def state(self) -> ElementState:
        if not self.declared:
            return ElementState.NOT_DECLARED
        if self.exclusion_authorized:
            return ElementState.EXCLUDED_BY_USER
        if self.executed and self.evidenced:
            return ElementState.EXECUTED_AND_EVIDENCED
        if self.executed:
            return ElementState.EXECUTED_NOT_EVIDENCED
        return ElementState.DECLARED_NOT_EXECUTED

    @property
    def blocks(self) -> bool:
        return self.state not in TERMINAL

    def to_json(self) -> Dict[str, Any]:
        return {"element_id": self.element_id, "declared": self.declared,
                "compiled": self.compiled, "executed": self.executed,
                "evidenced": self.evidenced,
                "exclusion_authorized": self.exclusion_authorized,
                "state": self.state.value, "reason": self.reason,
                "detail": dict(self.detail)}


@dataclass(frozen=True)
class Coverage:
    """The whole plan's declared-to-executed record."""

    elements: Sequence[DeclaredElement] = ()

    @property
    def blocking(self) -> Sequence[DeclaredElement]:
        return tuple(one for one in self.elements if one.blocks)

    @property
    def publishable(self) -> bool:
        return not self.blocking

    @property
    def declared_count(self) -> int:
        return sum(1 for one in self.elements if one.declared)

    @property
    def covered_count(self) -> int:
        return sum(1 for one in self.elements
                   if one.declared and not one.blocks)

    def ratio(self) -> str:
        return f"{self.covered_count}/{self.declared_count}"

    def refusal(self) -> str:
        """What to show instead of a figure.

        Names the elements rather than describing the shortfall, because the
        message a user needed was never "this cannot be shown" — it was "the
        three-month period you asked for was not applied".
        """
        blocking = self.blocking
        if not blocking:
            return ""
        reasons = "; ".join(one.reason or one.element_id for one in blocking)
        return ("This result is unavailable. Part of what you described was "
                "not carried into the calculation, so any figure would answer "
                f"a different plan: {reasons}. Your description and answers "
                "remain saved.")

    def to_json(self) -> Dict[str, Any]:
        return {"elements": [one.to_json() for one in self.elements],
                "publishable": self.publishable,
                "coverage": self.ratio(),
                "blocking": [one.element_id for one in self.blocking],
                "version": COVERAGE_VERSION}


#: Bumped when the set of material elements changes, so a stored coverage
#: record keeps meaning what it meant. A run recorded under `@1` was checked
#: against the elements `@1` knew about, and nothing else.
COVERAGE_VERSION = "coverage/declared-to-executed@1"


# --- assessment ------------------------------------------------------------
#
# Detection of a declared element is deliberately text-first for the two the
# compiler has no representation for. A sell leg and a second funding mode are
# things this build cannot compile, so there is no compiled artifact to inspect
# — and "we found no representation of it" is exactly the reasoning that let
# them vanish. The description is what the user actually said.

import re

_SELL_LEG = re.compile(
    r"\bsell\b[^.]{0,80}\b(?:when|whenever|if|once|and then)\b"
    r"|\b(?:when|whenever|if|once)\b[^.]{0,80}\bsell\b"
    r"|\bsell everything\b|\bsell it all\b|\bswitch (?:to|into)\b",
    re.IGNORECASE)

#: A purchase tied to a condition, read from the user's own sentence.
#:
#: Deliberately broad where the compiler's trigger vocabularies are narrow.
#: This answers "did the user describe a conditional purchase at all", which is
#: a lower bar than "which of two readings did they mean" — and the whole point
#: is that it must still hold when the second question cannot be answered.
_CONDITIONAL_PURCHASE = re.compile(
    r"\b(?:when|whenever|each time|every time|any time|if|once)\b[^.]{0,90}"
    r"\b(?:below|above|under|over|cross\w*|drops?|falls?|rises?|breaks?)\b"
    r"|\bcross\w*\b[^.]{0,40}\b(?:below|above|under|over)\b",
    re.IGNORECASE)

#: Not a set of "material unresolved fields". That was the first attempt and
#: it was wrong: it counted every unanswered question as a declared element,
#: so a first submission with `funding_source` open — which is nearly all of
#: them — published no figure at all, and the ask-and-refine loop the product
#: is built on stopped working.
#:
#: A compiler-raised question is not something the user declared. The four
#: shapes that must block are each derived from the user's own words already:
#: a stated period (`evaluation_period`), a stated condition
#: (`_CONDITIONAL_PURCHASE`), a stated second funding source
#: (`_SCHEDULED_ALSO`), and a stated sell leg (`_SELL_LEG`).

#: An amount that changes on a condition — "double it when the market falls".
#:
#: `_CONDITIONAL_PURCHASE` covers *whether* a purchase happens. This covers
#: *how much*, which is a separate dimension the compiler cannot express: the
#: base plan runs, the doubling vanishes, and coverage reported 1/1 because the
#: element had never entered the denominator.
#:
#: Written knowing this list is the wrong long-term shape — the recurring
#: failure is that a semantic dimension the compiler cannot represent falls out
#: of the inventory entirely, and the fix for that is a compiler-derived
#: inventory rather than one more entry. Added anyway, because the alternative
#: is shipping a known wrong figure while the better abstraction is designed.
_CONDITIONAL_AMOUNT = re.compile(
    r"\b(?:double|triple|quadruple|halve|increase|decrease|reduce|raise|"
    r"lower|boost|step\s+up|scale\s+up)\b[^.]{0,70}"
    r"\b(?:when|whenever|if|after|once|any\s+(?:month|week|quarter|year|time))\b"
    r"|\b(?:when|whenever|if|after|once)\b[^.]{0,70}"
    r"\b(?:double|triple|quadruple|halve|increase|decrease|reduce|raise|"
    r"lower|boost)\b",
    re.IGNORECASE)

_SCHEDULED_ALSO = re.compile(
    r"\b(?:also|and)\b[^.]{0,60}\b(?:invest|contribute|put|add)\b[^.]{0,40}"
    r"\b(?:every|each|per)\s+(?:month|week|quarter|year|paycheck)"
    r"|\b(?:monthly|weekly|quarterly|annually)\b[^.]{0,30}\b(?:as well|too|also)\b",
    re.IGNORECASE)


def assess(scenario: Any, *, stated_text: str = "",
           resolved_window: Any = None, frame_sessions: int = 0,
           ledger: Any = None, excluded_items: Sequence[str] = ()) -> Coverage:
    """Build the record for one run.

    `resolved_window` and `ledger` are the *evidence* arguments and come from
    somewhere other than the thing they attest to: the resolved window carries
    the session bounds the frame was actually cut to, and the ledger carries
    rows the engine filled. Passing the scenario alone would produce a record
    asserting that everything declared was done, which is what the declaration
    already said.
    """
    excluded = set(excluded_items or ())
    elements: List[DeclaredElement] = []

    # --- the evaluation period ---------------------------------------------
    window = getattr(scenario.provenance, "time_window", None)
    declared_window = window is not None
    window_executed = resolved_window is not None
    # Evidenced by the frame having been cut to it: a resolved window nothing
    # sliced is a computation, not an execution.
    window_evidenced = bool(
        window_executed and frame_sessions
        and getattr(resolved_window, "start", None) is not None)
    elements.append(DeclaredElement(
        element_id="evaluation_period",
        declared=declared_window,
        compiled=declared_window and bool(getattr(window, "supported", False)),
        executed=window_executed,
        evidenced=window_evidenced,
        exclusion_authorized="evaluation_period" in excluded,
        # Two different failures behind one blocked element, and a reader has
        # to be able to tell them apart. "We could not read the period you
        # asked for" sends you to rephrase; "we read it and ignored it" sends
        # you to distrust the number. Reporting the second for the first was
        # the wording this file shipped with.
        reason=("" if not declared_window or window_evidenced else
                (f"this version cannot replay {getattr(window, 'observed', 'that period')!r} — "
                 f"it replays a trailing period such as 'the past 5 years'")
                if not getattr(window, "supported", False) else
                f"you asked for {getattr(window, 'label', 'a period')} and the "
                f"calculation used the whole available history instead"),
        detail={"sessions_evaluated": frame_sessions,
                "resolved": (resolved_window.to_json()
                             if hasattr(resolved_window, "to_json") else None)}))

    # --- event-triggered funding -------------------------------------------
    #
    # `declared` comes from the user's words; `compiled` from what the compiler
    # made of them. It used to come from `is_event_funded`, which is a compiled
    # output — so a plan whose trigger the compiler could not resolve declared
    # *nothing*, coverage reported 1/1, and a buy-and-hold figure was published
    # for a conditional strategy.
    #
    # That is the loophole in the honesty gate itself: the harder an element
    # was to understand, the easier it became for the record to forget the
    # user had said it. Failing to compile a declared element must raise
    # uncertainty, never erase the element.
    event_funded = bool(getattr(scenario, "is_event_funded", False))
    stated_condition = bool(stated_text and _CONDITIONAL_PURCHASE.search(stated_text))
    declares_trigger = event_funded or stated_condition
    rows = len(getattr(ledger, "rows", ()) or ())
    elements.append(DeclaredElement(
        element_id="event_triggered_funding",
        declared=declares_trigger,
        compiled=event_funded,
        executed=event_funded and rows > 0,
        evidenced=rows > 0,
        exclusion_authorized="event_triggered_funding" in excluded,
        reason=("" if not declares_trigger or rows else
                "you described buying when a condition occurs, and this "
                "reading of your description was not settled, so no such "
                "purchase was made"
                if not event_funded else
                "the purchases your condition would trigger were not carried "
                "out"),
        detail={"purchases": rows, "compiled_a_policy": event_funded}))

    # --- a second, scheduled funding source --------------------------------
    #
    # Only material when it sits *beside* event funding: a plain scheduled plan
    # is the ordinary case and executes correctly. The dangerous shape is both
    # at once, where one is silently dropped — and the dropped one was larger
    # than the executed one in the case that found this.
    scheduled_alongside = bool(
        event_funded and stated_text and _SCHEDULED_ALSO.search(stated_text))
    elements.append(DeclaredElement(
        element_id="scheduled_funding",
        declared=scheduled_alongside,
        compiled=False,
        executed=False,
        evidenced=False,
        exclusion_authorized="scheduled_funding" in excluded,
        reason=("" if not scheduled_alongside else
                "you described regular contributions as well as the triggered "
                "ones, and this version can execute only one funding source "
                "per plan"),
        detail={}))

    # --- an amount that changes on a condition ------------------------------
    conditional_amount = bool(stated_text
                              and _CONDITIONAL_AMOUNT.search(stated_text))
    elements.append(DeclaredElement(
        element_id="conditional_amount",
        declared=conditional_amount,
        compiled=False,
        executed=False,
        evidenced=False,
        exclusion_authorized="conditional_amount" in excluded,
        reason=("" if not conditional_amount else
                "you described changing how much you invest when a condition "
                "occurs, and this version contributes a fixed amount"),
        detail={}))

    # --- a sell leg ---------------------------------------------------------
    sell_leg = bool(stated_text and _SELL_LEG.search(stated_text))
    elements.append(DeclaredElement(
        element_id="sell_action",
        declared=sell_leg,
        compiled=False,
        executed=False,
        evidenced=False,
        exclusion_authorized="sell_action" in excluded,
        reason=("" if not sell_leg else
                "you described selling under a condition, and this version "
                "executes purchases only"),
        detail={}))

    return Coverage(elements=tuple(elements))
