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


# --- strategy families this build does not model -----------------------------
#
# Two of these executed on the live lane under gpt-5.4 and were refused under
# gpt-4.1, and the reason the old model refused was not that anything here
# recognised the family. gpt-4.1 happened to also report a `portfolio_sleeves`
# relation, which is unsupported, so the plan failed to compile for an
# unrelated reason. gpt-5.4 sometimes omits that relation and the remaining
# fragment — some holdings and a percentage — looks like an ordinary
# accumulation plan and compiles.
#
# A refusal that depends on another dimension happening to fail first is not a
# refusal. These make the semantic itself detectable, so the refusal is caused
# by the thing being refused.
#
# **Vocabulary only.** What the words mean, with sources. Whether a sentence
# contains one is `derived_readers.unsupported_family`, which reads the parse;
# what to do about it is Mission's, which refuses any dimension nothing
# consumes. Neither decision belongs in a word list.


@dataclass(frozen=True)
class Family:
    """A strategy family, the words that name it, and where that is written."""

    dimension: str
    #: Noun phrases that name the family on their own. Terms of art — nobody
    #: writes "small cap value" or "my age in bonds" about anything else.
    terms: frozenset = frozenset()
    #: Words that express a tilt, in any position. On their own they prove
    #: nothing: `overweight` is ordinary English about a person, and Stanza
    #: tags it ADJ even in "I overweight value", so a predicate-position rule
    #: could never fire for it.
    markers: frozenset = frozenset()
    #: The factors and styles a tilt can be toward. A marker and one of these
    #: together name the family; either alone does not. "I am overweight and
    #: want to retire early" has the marker and no style; "hold 40% in value
    #: stocks" has the style and no marker, and is an ordinary holding.
    styles: frozenset = frozenset()
    why: str = ""
    source: str = ""


#: The families, by the dimension a refusal will name.
#:
#: Deliberately narrow. Every entry is a family the corpus already declares
#: `REFUSED_BY_NAME` with a cited definition, and the words are the ones those
#: cited definitions use. A family added here without a source is a family
#: somebody guessed at, and `tests/test_unsupported_families.py` asserts both.
UNSUPPORTED_FAMILIES = {
    "factor_tilt": Family(
        dimension="factor_tilt",
        terms=frozenset({
            "small cap value", "small-cap value", "value tilt", "quality tilt",
            "momentum tilt", "size tilt", "factor tilt", "small cap tilt",
            "value factor", "quality factor", "momentum factor",
            "size factor", "factor exposure", "smart beta",
        }),
        markers=frozenset({"tilt", "tilted", "tilts", "overweight",
                           "underweight", "overweighted", "underweighted"}),
        styles=frozenset({
            "value", "growth", "quality", "momentum", "size", "small cap",
            "small-cap", "large cap", "large-cap", "mid cap", "mid-cap",
            "smallcap", "low volatility", "min vol", "profitability",
            "factor", "factors", "style", "beta",
        }),
        why="a tilt names a factor rather than the holdings to buy, and this "
            "build divides each purchase between named instruments",
        source="https://www.investopedia.com/terms/s/smallcap.asp"),

    "age_based_allocation": Family(
        dimension="age_based_allocation",
        terms=frozenset({
            "my age in bonds", "your age in bonds", "age in bonds",
            "glide path", "glidepath", "age-based", "age based",
            "target date", "target-date", "as i get older", "as i age",
            "as you age", "over time as", "declining equity",
            "rising equity", "reduce equity exposure over time",
            "increase bonds as", "decrease stocks as",
        }),
        why="the allocation changes with age or elapsed time, and this build "
            "holds one allocation for the whole evaluation",
        source="https://benchmarkfg.com/wp-content/uploads/2025/05/"
               "Reducing-Retirement-Risk-with-a-Rising-Equity-Glide-Path-2.pdf"),
}
