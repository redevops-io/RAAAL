"""Well-formed statements to pick from and edit.

The harvest found that real strategy statements routinely omit what to buy —
"I contribute $750/month to my 401k" is a complete thought to the person writing
it, and the runtime cannot run it. Asking people to guess the vocabulary that
gets them a plan is the interaction defect underneath most of the follow-up
burden. A list of statements they can pick and then edit removes the guessing
without removing the typing: the text box stays the primary input, and a picked
sentence lands in it as ordinary editable text.

Two rules hold this together, and both exist because a catalogue is the easiest
place in a product to make a claim nobody checks.

**Everything offered must execute.** An entry in a dropdown is the product
saying "this works". Offering a strategy that compiles to a refusal is a false
claim of support, and worse than the user typing the same sentence unprompted,
because we suggested it. `tests/test_strategy_library.py` runs every offered
entry through the whole pipeline and fails if one does not produce a plan.

**Nothing here advertises what the engine cannot do.** A list of refused
strategies was rendered beside this one for exactly one revision. It was the
wrong answer to a real problem: the fix for "this build cannot evaluate
momentum" is to evaluate momentum, not to explain the gap more clearly. A
sentence the engine will not run is still refused by name when somebody writes
it, which is where the boundary belongs — at the point of the request, not as a
standing advertisement of absence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Entry:
    """One statement somebody can pick.

    `text` is what lands in the box, and it is deliberately a whole sentence in
    ordinary words rather than a fill-in-the-blanks form. The runtime reads
    prose; a template with slots would be a second grammar to maintain, and the
    thing the pilot is measuring is whether prose works.
    """

    key: str
    title: str
    text: str
    #: What this entry exists to demonstrate. Shown to nobody — it is the
    #: reason the entry is in the catalogue, for whoever edits it next.
    demonstrates: str


@dataclass(frozen=True)
class Group:
    key: str
    title: str
    #: Why these belong together, in the user's terms rather than the schema's.
    note: str
    entries: Tuple[Entry, ...]


#: The offered catalogue.
#:
#: Grouped by the distinction the engine actually makes — what puts money in,
#: and what it buys — because that is what changes the answer. Grouping by the
#: names the industry uses would be grouping by a distinction this build cannot
#: act on, and every group but one would be empty.
LIBRARY: Tuple[Group, ...] = (
    Group(
        key="regular",
        title="Money in on a schedule",
        note="A fixed amount on a calendar. The most common shape by far, and "
             "the one the engine models most directly.",
        entries=(
            Entry("monthly-single", "Monthly into one fund",
                  "invest $500 a month into VTI",
                  "the baseline: one asset, one cadence, one amount"),
            Entry("monthly-pair", "Monthly, split between two funds",
                  "invest $500 a month split equally between VTI and BND",
                  "a SET-valued holding; equal split is what the engine does"),
            Entry("weekly-single", "Weekly into one fund",
                  "put $100 into VOO every week",
                  "a non-monthly cadence, phrased as a person would"),
            Entry("quarterly-single", "Quarterly into one fund",
                  "invest $3,000 into QQQ every quarter",
                  "a cadence with a thousands separator in the amount"),
            Entry("first-of-month", "Monthly, on the first trading day",
                  "invest $750 into VTI on the first trading day of each month",
                  "day_rule stated explicitly rather than defaulted"),
        ),
    ),
    Group(
        key="triggered",
        title="Money in when the market does something",
        note="A condition rather than a date. The engine fills on the next "
             "session's open — never the close that produced the signal.",
        entries=(
            Entry("below-ma", "Buy when it falls below its moving average",
                  "buy $1,000 of SPY every time it closes below its 200 day "
                  "moving average, on the next trading day",
                  "crossing_event, and the sentence the first pilot user "
                  "actually typed"),
            Entry("stays-below", "Buy while it stays below its average",
                  "buy $1,000 of SPY on any day it is trading below its 200 "
                  "day moving average",
                  "persistent_condition — the other reading of the same shape, "
                  "and a different strategy"),
            # A drawdown entry — "invest $2,000 into VTI whenever it drops 10%
            # below its highest close of the last year" — was written here and
            # removed. It runs under claude-sonnet-5 and is refused under
            # gpt-4.1-2025-04-14, which reads the fixed $2,000 as a
            # `conditional_amount` and refuses a strategy whose amount does not
            # in fact vary. Offering a sentence that works on one provider is
            # offering a coin toss, so it is out until the false refusal is
            # fixed; docs/Benchmark-Queue.md carries it.
        ),
    ),
    Group(
        key="one-off",
        title="A single purchase",
        note="One amount, once. Useful as a baseline to compare a "
             "contribution plan against.",
        entries=(
            Entry("lump-sum", "Invest a lump sum now",
                  "invest $10,000 into VTI as a lump sum",
                  "cadence=once; the phrase that broke the clarification loop"),
            Entry("lump-sum-pair", "A lump sum across two funds",
                  "invest $10,000 equally into VTI and BND as a one-off "
                  "purchase",
                  "once + SET holding together"),
        ),
    ),
)


def entry(key: str) -> Optional[Entry]:
    for group in LIBRARY:
        for candidate in group.entries:
            if candidate.key == key:
                return candidate
    return None


def offered() -> Sequence[Entry]:
    return [e for group in LIBRARY for e in group.entries]


#: How a sentence got into the box. Recorded on every attempt, because without
#: it the cohort measures the catalogue rather than the runtime: a high success
#: rate over sentences we wrote, read by a reader we wrote, is a closed loop.
#: `TYPED` is the only origin that carries evidence about people's own words,
#: and `EDITED` is the interesting middle — it says the catalogue got them
#: close and something was missing.
PICKED, EDITED, TYPED = "PICKED", "EDITED", "TYPED"


def origin_of(text: str, picked_key: str) -> str:
    """Derived from the text and the pick, never taken from the client.

    A hidden field saying "PICKED" is a claim the browser makes, and the one
    thing it cannot be trusted about is whether the user then changed the
    sentence — which is exactly the distinction this is for. Comparing against
    the catalogue is cheap and cannot be wrong about it.
    """
    if not picked_key:
        return TYPED
    chosen = entry(picked_key)
    if chosen is None:
        return TYPED
    return PICKED if " ".join(text.split()) == " ".join(
        chosen.text.split()) else EDITED
