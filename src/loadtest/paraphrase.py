"""Turn one catalog row into many differently-worded user descriptions.

Nine classes, from the test plan. Each exists because it separates two readings
that are economically different and that ordinary language does not distinguish
— which is exactly where a compiler is most likely to guess:

    COMPLETE              every material choice stated
    UNDERSPECIFIED        a material choice left out
    CONTRADICTORY         two stated choices that cannot both hold
    ACCOUNT_OMISSION      no account or tax context named
    PERSISTENT_VS_EVENT   "whenever it is below" versus "the day it crosses"
    EQUAL_WEIGHT          equal dollars per purchase versus keeping positions equal
    FUNDING_SOURCE        out of the contribution versus additional cash
    CALENDAR_VS_SESSION   the 1st of the month versus the first trading day
    RECOMMENDATION_BAIT   asks the system to choose, which it must refuse

**Every prompt carries what should happen to it.** A load run that only reports
"14,400 compiled" has measured throughput and nothing else; the interesting
question is whether the ones that should have asked a question did, and whether
the ones that should have compiled cleanly did not.

Generation is deterministic — seeded from the strategy id and index — so a
failure found on one machine is reproducible on another by id alone.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence

from ..mission.compiler import _CADENCE, _MENTIONS_SIGNAL, _RULES
from .catalog import Strategy


class Klass(str, Enum):
    COMPLETE = "COMPLETE"
    UNDERSPECIFIED = "UNDERSPECIFIED"
    CONTRADICTORY = "CONTRADICTORY"
    ACCOUNT_OMISSION = "ACCOUNT_OMISSION"
    PERSISTENT_VS_EVENT = "PERSISTENT_VS_EVENT"
    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    FUNDING_SOURCE = "FUNDING_SOURCE"
    CALENDAR_VS_SESSION = "CALENDAR_VS_SESSION"
    RECOMMENDATION_BAIT = "RECOMMENDATION_BAIT"


class Expect(str, Enum):
    """What the compiler owes this prompt."""

    COMPILES_SAVEABLE = "COMPILES_SAVEABLE"
    """Everything material is stated; it should reach a saveable scenario."""

    ASKS_A_QUESTION = "ASKS_A_QUESTION"
    """Something material is missing. An answer here is a guess."""

    REPORTS_A_CONTRADICTION = "REPORTS_A_CONTRADICTION"
    """Two stated choices cannot both hold. Silently picking one is the failure."""

    REFUSES_TO_CHOOSE = "REFUSES_TO_CHOOSE"
    """The user asked which is best. The platform does not answer that."""


@dataclass(frozen=True)
class Prompt:
    prompt_id: str
    strategy_id: str
    family: str
    klass: Klass
    expect: Expect
    text: str
    #: The field this prompt is probing, when it probes one. Lets a failure be
    #: reported as "funding_source was never asked about" rather than a count.
    probes: Optional[str] = None

    @property
    def seed(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()[:12]


_AMOUNTS = (250, 500, 750, 1000, 1500, 2000, 2500, 5000)

#: Cadences that name a recurring schedule. `event` and `conditional` do not —
#: a row whose cadence is "when it happens" cannot be stated completely in prose
#: without inventing a schedule the catalog does not give.
_RECURRING = frozenset({"annual", "monthly", "weekly", "biweekly", "quarterly",
                        "payroll", "daily"})


def text_is_complete(text: str) -> bool:
    """Whether *this rendering* states everything the compiler needs.

    Derived from the built text rather than from the row, because the account
    and the cadence are chosen per prompt. Two rounds of seed-matching against
    the row got 180 then 445 expectations wrong; asking the text is the only
    version that cannot drift from what was actually written.
    """
    if not _ACCOUNT_RECOGNIZED.search(text):
        return False
    if not any(re.search(pattern, text, re.IGNORECASE)
               for _name, pattern in _CADENCE):
        return False
    # A mention of a market condition with no stated semantics is an open
    # question however completely the rest is worded.
    return not _MENTIONS_SIGNAL.search(text)


def fully_specifiable(strategy: Strategy) -> bool:
    """Whether a COMPLETE paraphrase of this row can actually be complete.

    Two conditions, both found by grading a run rather than by reasoning:

    A row whose cadence is `event` or `conditional` has no schedule to state, so
    the prompt says "when it happens" and the compiler rightly asks how often.

    A row whose universe is a phrase implying a market signal — "convert after
    -20%", "invest only above floor" — makes the prompt mention a condition
    without saying how it behaves, so the compiler rightly asks whether it is a
    persistent state or a crossing event.

    A phrase universe on its own does *not* disqualify a row: the compiler has
    every material field it needs and saves happily. Assuming otherwise flagged
    1,488 prompts that were behaving correctly.
    """
    if strategy.cadence not in _RECURRING:
        return False
    if _MENTIONS_SIGNAL.search(strategy.universe_or_assets):
        return False
    # A row whose account context the compiler cannot place — a donor-advised
    # fund, an inherited IRA, "my retirement accounts", cash savings — cannot be
    # stated completely either. Guessing between traditional and Roth is this
    # project's founding example of a materially wrong result, so asking is
    # correct and the prompt is not complete.
    return all(_ACCOUNT_RECOGNIZED.search(phrase)
               for phrase in _account_phrases(strategy))

    # A life-event template hint deliberately does *not* disqualify a row. The
    # hand-off is an offer, not a blocker: the generic compile is a valid
    # interpretation and the template is a more precise one the user chooses.
_CADENCE_WORDS = {
    "annual": ["once a year", "every year", "annually"],
    "monthly": ["every month", "each month", "monthly"],
    "weekly": ["every week", "weekly"],
    "biweekly": ["every other week", "every two weeks"],
    "quarterly": ["every quarter", "quarterly"],
    "payroll": ["every payday", "out of each paycheck"],
    "daily": ["every day", "daily"],
    "conditional": ["when the condition is met"],
    "event": ["when it happens"],
}


#: Account phrasings the compiler can place. Built from the compiler's own
#: patterns so the two cannot drift apart.
_ACCOUNT_RECOGNIZED = re.compile(
    "|".join(pattern for field, _value, pattern in _RULES
             if field == "account_type"), re.IGNORECASE)


def _account_phrases(strategy: Strategy) -> List[str]:
    """Every account phrase this row can produce.

    All of them, not a sample: the prompt picks one at random per index, so a
    row with a mix of placeable and unplaceable accounts produces complete
    prompts for some indices and not others. Checking a sample got 180 of them
    wrong.
    """
    return [_ACCOUNT_WORDS.get(a, f"in my {a} account")
            for a in (strategy.accounts or ["taxable"])]


def _assets_phrase(strategy: Strategy, rng: random.Random) -> str:
    assets = strategy.assets
    if not assets:
        # Rows that name a phrase rather than tickers. Kept verbatim: a compiler
        # that has only seen clean ticker lists has not met a real user.
        return strategy.universe_or_assets
    if len(assets) == 1:
        return assets[0]
    if len(assets) == 2:
        return f"{assets[0]} and {assets[1]}"
    return ", ".join(assets[:-1]) + f" and {assets[-1]}"


def _cadence_phrase(strategy: Strategy, rng: random.Random) -> str:
    return rng.choice(_CADENCE_WORDS.get(strategy.cadence, [strategy.cadence]))


_ACCOUNT_WORDS = {
        "taxable": "in my taxable brokerage account",
        "traditional": "in my traditional IRA",
        "roth": "in my Roth IRA",
        "401k": "in my 401(k)",
        "roth401k": "in my Roth 401(k)",
        "ira": "in my IRA",
        "rothira": "in my Roth IRA",
        "cash": "out of my cash savings",
    "retirement": "in my retirement accounts",
}


def _account_phrase(strategy: Strategy, rng: random.Random) -> str:
    account = rng.choice(strategy.accounts) if strategy.accounts else "taxable"
    return _ACCOUNT_WORDS.get(account, f"in my {account} account")


# --- the nine classes ------------------------------------------------------

def _complete(s: Strategy, rng: random.Random) -> str:
    return (
        f"I put ${rng.choice(_AMOUNTS):,} into {_assets_phrase(s, rng)} "
        f"{_cadence_phrase(s, rng)} {_account_phrase(s, rng)}, on the first "
        f"trading day of the period, reinvesting dividends, and I never sell. "
        f"Compare it against {' and '.join(s.benchmarks[:2]) or 'buy and hold'}."
    )


def _underspecified(s: Strategy, rng: random.Random) -> str:
    return f"I want to invest in {_assets_phrase(s, rng)}."


def _contradictory(s: Strategy, rng: random.Random) -> str:
    return (
        f"I buy ${rng.choice(_AMOUNTS):,} of {_assets_phrase(s, rng)} "
        f"{_cadence_phrase(s, rng)} and rebalance them back to equal weights "
        f"every quarter, but I never sell anything."
    )


def _account_omission(s: Strategy, rng: random.Random) -> str:
    return (
        f"I add ${rng.choice(_AMOUNTS):,} to {_assets_phrase(s, rng)} "
        f"{_cadence_phrase(s, rng)} and let the dividends reinvest."
    )


def _persistent_vs_event(s: Strategy, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return (f"Whenever SPY is trading below its 200 day moving average I buy "
                f"${rng.choice(_AMOUNTS):,} of {_assets_phrase(s, rng)} with "
                f"additional cash, {_account_phrase(s, rng)}.")
    return (f"On the day SPY crosses below its 200 day moving average I buy "
            f"${rng.choice(_AMOUNTS):,} of {_assets_phrase(s, rng)} with "
            f"additional cash, {_account_phrase(s, rng)}.")


def _equal_weight(s: Strategy, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return (f"I buy {_assets_phrase(s, rng)} equally {_cadence_phrase(s, rng)}, "
                f"${rng.choice(_AMOUNTS):,} split equal dollars at each purchase.")
    return (f"I keep {_assets_phrase(s, rng)} at equal weights, rebalancing "
            f"{_cadence_phrase(s, rng)} with ${rng.choice(_AMOUNTS):,}.")


def _funding_source(s: Strategy, rng: random.Random) -> str:
    if rng.random() < 0.5:
        return (f"My monthly contribution is ${rng.choice(_AMOUNTS):,}. When "
                f"{_assets_phrase(s, rng)} drops I buy more out of that "
                f"contribution, {_account_phrase(s, rng)}.")
    return (f"My monthly contribution is ${rng.choice(_AMOUNTS):,}. When "
            f"{_assets_phrase(s, rng)} drops I buy more with additional cash on "
            f"top of it, {_account_phrase(s, rng)}.")


def _calendar_vs_session(s: Strategy, rng: random.Random) -> str:
    when = rng.choice(["on the first calendar day of the month",
                       "on the first trading day of the month"])
    return (f"I invest ${rng.choice(_AMOUNTS):,} in {_assets_phrase(s, rng)} "
            f"{when}, {_account_phrase(s, rng)}, and I never sell.")


def _recommendation_bait(s: Strategy, rng: random.Random) -> str:
    return rng.choice([
        f"Which is better for me, {_assets_phrase(s, rng)} or an S&P 500 fund?",
        f"What should I invest in {_account_phrase(s, rng)}? Pick the best one.",
        f"Should I buy {_assets_phrase(s, rng)} now? Tell me what to do.",
    ])


_BUILDERS: Dict[Klass, tuple] = {
    Klass.COMPLETE:            (_complete, Expect.COMPILES_SAVEABLE, None),
    Klass.UNDERSPECIFIED:      (_underspecified, Expect.ASKS_A_QUESTION, "amount"),
    Klass.CONTRADICTORY:       (_contradictory, Expect.REPORTS_A_CONTRADICTION,
                                "sells_allowed"),
    Klass.ACCOUNT_OMISSION:    (_account_omission, Expect.ASKS_A_QUESTION, None),
    Klass.PERSISTENT_VS_EVENT: (_persistent_vs_event, Expect.COMPILES_SAVEABLE,
                                "trigger_semantics"),
    Klass.EQUAL_WEIGHT:        (_equal_weight, Expect.COMPILES_SAVEABLE, "weighting"),
    Klass.FUNDING_SOURCE:      (_funding_source, Expect.COMPILES_SAVEABLE,
                                "funding_source"),
    Klass.CALENDAR_VS_SESSION: (_calendar_vs_session, Expect.COMPILES_SAVEABLE,
                                "contribution_day_rule"),
    Klass.RECOMMENDATION_BAIT: (_recommendation_bait, Expect.REFUSES_TO_CHOOSE, None),
}

#: The rotation. Weighted toward the classes that separate two real readings,
#: because those are where a wrong answer is invisible rather than obviously
#: absent.
_ROTATION: Sequence[Klass] = (
    [Klass.COMPLETE] * 2
    + [Klass.UNDERSPECIFIED, Klass.CONTRADICTORY, Klass.ACCOUNT_OMISSION]
    + [Klass.PERSISTENT_VS_EVENT] * 2
    + [Klass.EQUAL_WEIGHT] * 2
    + [Klass.FUNDING_SOURCE] * 2
    + [Klass.CALENDAR_VS_SESSION] * 2
    + [Klass.RECOMMENDATION_BAIT]
)


def paraphrases(strategy: Strategy, count: int) -> List[Prompt]:
    """`count` prompts for one strategy, deterministic in the strategy id.

    Same id and same index produce the same text on any machine, so a defect
    found in a fourteen-thousand-prompt run is reproducible from two integers.
    """
    out: List[Prompt] = []
    for index in range(count):
        klass = _ROTATION[index % len(_ROTATION)]
        builder, expect, probes = _BUILDERS[klass]
        rng = random.Random(f"{strategy.strategy_id}:{index}")
        text = builder(strategy, rng)
        if klass is Klass.COMPLETE and not text_is_complete(text):
            # This wording leaves something open, so the compiler owes a
            # question rather than a saveable plan.
            expect = Expect.ASKS_A_QUESTION
        out.append(Prompt(
            prompt_id=f"{strategy.strategy_id}#{index:03d}",
            strategy_id=strategy.strategy_id, family=strategy.family,
            klass=klass, expect=expect, text=text, probes=probes))
    return out


def corpus(strategies: Sequence[Strategy], per_strategy: int) -> List[Prompt]:
    return [p for s in strategies for p in paraphrases(s, per_strategy)]
