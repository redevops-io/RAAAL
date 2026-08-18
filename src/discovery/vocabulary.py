"""What Quantify's dimensions need, and which of its words carry two meanings.

Finance vocabulary, and the reason it has its own module: it lived in
`fusion.py` beside the generic machinery, and that file goes away now that
`discovery-runtime` provides the machinery. Vocabulary is not machinery.

Which terms people demonstrably use for two things — and between which
dimensions — is a fact about *this* domain, observed in how people write rather
than predicted from how the code is shaped, which is why every entry carries a
source. What a dimension needs before a value means anything is the same kind
of fact.

Both tables are read by `discovery.adapter` and passed to the runtime as its
`ambiguity` and `material` seams. The runtime is handed the observation and
provides the outcome; it never learns the words.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


AMBIGUOUS_TERMS: Mapping[str, Mapping[str, Any]] = {
    "rebalance": {
        "readings": "rebalance back to target | change the target allocation",
        "between": ("periodic_rebalancing", "stated_weights"),
        "evidence": "Bogleheads thread 'Don't Know How To Rebalance/Reallocate'",
        "source": "https://www.bogleheads.org/forum/viewtopic.php?t=459742"},
    "reallocate": {
        "readings": "rebalance back to target | change the target allocation",
        "between": ("periodic_rebalancing", "stated_weights"),
        "evidence": "same thread; the two verbs are used interchangeably",
        "source": "https://www.bogleheads.org/forum/viewtopic.php?t=459742"},
}
"""Terms people demonstrably use for more than one thing.

`between` names the contract fields the ambiguity is *between*, and it is what
keeps this outcome from firing on its own vocabulary. "rebalanced annually"
carries the word and no ambiguity at all: the reading is
`periodic_rebalancing=annual`, and the competing reading — that the target is
being changed — needs a target, which the sentence does not contain. "rebalance
to 70/30" does contain one, and there the two readings are both available and
neither is chosen by the words.

So the rule is not "the word appeared". It is "both readings are on the table",
which is what ambiguity means."""



@dataclass(frozen=True)
class Requirement:
    """What a dimension needs before a value means anything.

    `binds` names the relation a value is meaningless without. `60/40` alone is
    a fact; `50/50` in a sentence naming three accounts is not, until something
    says which account it belongs to.
    """

    material: bool = True
    binds: Optional[str] = None
    compare_as: str = "TEXT"
    """How two readers' values for this dimension are the same value. Mirrors
    `Dimension.compare_as` in the schema, and is declared rather than guessed
    for the same reason it is there."""


#: Declared per dimension rather than inferred. A dimension absent here is
#: material and unbound — the conservative reading, since treating an unknown
#: dimension as immaterial would let anything new proceed unexamined.
REQUIREMENTS: Mapping[str, Requirement] = {
    "cadence": Requirement(material=True),
    "amount": Requirement(material=True, compare_as="NUMBER"),
    "assets": Requirement(material=True, compare_as="SET"),
    "allocation_method": Requirement(material=True),
    "moving_average_window": Requirement(material=True,
                                        compare_as="NUMBER"),
    "trigger_semantics": Requirement(material=True),
    "execution_timing": Requirement(material=True),
    "day_rule": Requirement(material=False),
    "evaluation_period": Requirement(material=True),
    "periodic_rebalancing": Requirement(material=False),
    "objective": Requirement(material=True),
    "stated_weights": Requirement(material=True, compare_as="WEIGHTS"),
    "account_allocation": Requirement(material=True, binds="account"),
    "dividend_policy": Requirement(material=False),
}


#: Dimensions where a trailing `m` counts periods rather than millions.
#:
#: `m` is the one genuinely ambiguous magnitude letter. A reader writing `12m`
#: for an amount means twelve million; writing `12m` for a moving-average
#: window it means twelve months, and scaling that produced a twelve-million
#: session window that disagreed with syntax's 12 — a case that had been
#: answered correctly for months, broken by the fix for `2.5k`.
#:
#: `k`, `b` and `bn` are not ambiguous and are scaled everywhere.
PERIOD_DIMENSIONS = frozenset({
    "moving_average_window", "evaluation_period", "holding_period",
    "rebalancing_period", "lookback_window",
})
