"""Semantic stability: do paraphrases of one plan compile to one plan?

Different from the corpus in `paraphrase.py`, and deliberately so. That corpus
generates wordings that *should* compile differently — persistent condition
versus crossing event, contribution versus additional cash — and checks the
compiler tells them apart.

This generates wordings that mean **exactly the same thing** and checks the
compiler cannot tell them apart:

    one semantic fact set
        -> N surface renderings
        -> compile each
        -> every one must produce the same rule_hash

`rule_hash` is the identity of the market rule alone, so it is independent of
the plan's name and version. Two descriptions of the same rule that hash
differently mean the compiler read something from the wording rather than from
the meaning — which is the defect that makes a system feel arbitrary to a user
who rephrased one sentence and got a different answer.

This is a stronger property than recognition accuracy. Accuracy asks whether the
compiler understood; stability asks whether it understood *the same thing every
time*. A parser can be 100% accurate on a benchmark and still be unusable if a
synonym changes the result.

It is also the harness a language model in stage 1 has to be measured against.
The interesting claim is not that a model reads more wordings — it is that
different models, or the same model on different days, still land on one
canonical Mission.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .catalog import Strategy


@dataclass(frozen=True)
class Facts:
    """What the plan *means*, independent of how it is said."""

    assets: Sequence[str]
    amount: int
    cadence: str
    day_rule: str            # "session" | "calendar"
    dividends: str           # "reinvested" | "cash"
    sells_allowed: bool
    account: str

    def key(self) -> str:
        return "|".join([
            "+".join(sorted(self.assets)), str(self.amount), self.cadence,
            self.day_rule, self.dividends, str(self.sells_allowed), self.account])


#: Surface forms. Every entry in a list must mean the same thing — that is the
#: whole contract of this file. A near-synonym that shifts meaning would make a
#: real instability look like a generator bug, and the reverse.
_SAY_AMOUNT = (
    "I put ${amount:,} into {assets}",
    "I invest ${amount:,} in {assets}",
    "${amount:,} goes into {assets}",
    "I buy ${amount:,} of {assets}",
    "I contribute ${amount:,} to {assets}",
)
_SAY_CADENCE = {
    "monthly": ("every month", "each month", "monthly"),
    "quarterly": ("every quarter", "quarterly"),
    "annual": ("every year", "annually", "yearly"),
    "weekly": ("every week", "weekly"),
    "biweekly": ("every other week", "biweekly"),
}
_SAY_DAY_RULE = {
    "session": ("on the first trading day of the period",
                "on the first market day",
                "on the first trading session"),
    "calendar": ("on the first calendar day of the month",
                 "on the 1st",
                 "on the first of the month"),
}
_SAY_DIVIDENDS = {
    "reinvested": ("reinvesting the dividends", "with dividends reinvested",
                   "and I reinvest dividends"),
    "cash": ("holding the dividends as cash", "with dividends held as cash",
             "and dividends are not reinvested"),
}
_SAY_SELLS = {
    False: ("and I never sell", "and I don't sell anything",
            "with no selling"),
    True: ("", "", ""),
}
_SAY_ACCOUNT = {
    "taxable": ("in my taxable brokerage account", "in my taxable account",
                "in my brokerage account"),
    "roth": ("in my Roth IRA", "in my Roth account"),
    "traditional": ("in my traditional IRA", "in my traditional account"),
    "401k": ("in my 401(k)", "in my 401k"),
}


def facts_for(strategy: Strategy, *, seed: int = 0) -> Optional[Facts]:
    """The fact set a stability family is built from, or `None`.

    Rows without tickers or without a recurring cadence are skipped rather than
    forced: a family built on a phrase universe would vary in ways that are the
    generator's doing, and an unstable result would say nothing about the
    compiler.
    """
    if not strategy.assets or strategy.cadence not in _SAY_CADENCE:
        return None
    rng = random.Random(f"{strategy.strategy_id}:{seed}")
    account = next((a for a in strategy.accounts if a in _SAY_ACCOUNT), "taxable")
    return Facts(
        assets=tuple(strategy.assets[:3]),
        amount=rng.choice((500, 1000, 2000)),
        cadence=strategy.cadence,
        day_rule=rng.choice(("session", "calendar")),
        dividends=rng.choice(("reinvested", "cash")),
        sells_allowed=False,
        account=account,
    )


def _assets_phrase(assets: Sequence[str]) -> Iterator[str]:
    """Orderings of the holdings. A set is a set — "VTI and BND" and "BND and
    VTI" are the same portfolio, and a compiler that disagrees is reading order
    as meaning."""
    for permutation in itertools.islice(itertools.permutations(assets), 3):
        if len(permutation) == 1:
            yield permutation[0]
        elif len(permutation) == 2:
            yield f"{permutation[0]} and {permutation[1]}"
        else:
            yield ", ".join(permutation[:-1]) + f" and {permutation[-1]}"


def wordings(facts: Facts, count: int) -> List[str]:
    """`count` surface renderings of one fact set.

    Varies the verb, the cadence phrase, the day-rule phrase, the dividend
    phrase, the no-selling phrase, the account phrase and the order of the
    holdings — every axis a user would vary without meaning anything by it.
    """
    assets = list(_assets_phrase(facts.assets))
    combos = list(itertools.product(
        _SAY_AMOUNT, assets, _SAY_CADENCE[facts.cadence],
        _SAY_DAY_RULE[facts.day_rule], _SAY_DIVIDENDS[facts.dividends],
        _SAY_SELLS[facts.sells_allowed], _SAY_ACCOUNT[facts.account]))

    # Sampled across the whole product, not the first `count` of it.
    # `itertools.product` varies its rightmost axis fastest, so a prefix holds
    # the verb and the asset ordering fixed — the two axes most likely to break
    # stability. Taking a prefix reported 100% stable while never testing them.
    random.Random(facts.key()).shuffle(combos)

    out: List[str] = []
    for verb, asset_phrase, cadence, day_rule, dividends, sells, account in combos:
        if len(out) >= count:
            break
        clauses = [verb.format(amount=facts.amount, assets=asset_phrase),
                   cadence, account, day_rule, dividends]
        text = ", ".join(c for c in clauses if c)
        if sells:
            text += f", {sells}"
        out.append(text + ".")
    return out


@dataclass
class Family:
    """One meaning, many wordings, and what the compiler made of each."""

    strategy_id: str
    facts_key: str
    wordings: List[str] = field(default_factory=list)
    rule_hashes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def distinct(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for index, digest in enumerate(self.rule_hashes):
            out.setdefault(digest, []).append(index)
        return out

    @property
    def stable(self) -> bool:
        return len(self.distinct) <= 1 and not self.errors

    def divergence(self) -> str:
        """The two wordings that disagree, side by side.

        A count of distinct hashes is not actionable. The pair of sentences that
        produced them is.
        """
        groups = sorted(self.distinct.items(), key=lambda kv: -len(kv[1]))
        if len(groups) < 2:
            return ""
        (major, majority), (minor, minority) = groups[0], groups[1]
        return (f"{len(majority)} wording(s) -> {major[:16]}\n"
                f"        e.g. {self.wordings[majority[0]]!r}\n"
                f"    {len(minority)} wording(s) -> {minor[:16]}\n"
                f"        e.g. {self.wordings[minority[0]]!r}")


def run_family(strategy: Strategy, count: int, *,
               parser=None,
               benchmark_rule: str = "benchmark-policy/public-default@1"
               ) -> Optional[Family]:
    from ..mission.compiler import compile_scenario

    facts = facts_for(strategy)
    if facts is None:
        return None

    family = Family(strategy_id=strategy.strategy_id, facts_key=facts.key())
    for text in wordings(facts, count):
        family.wordings.append(text)
        try:
            parsed = parser(text) if parser else None
            result = compile_scenario(text, name="stability", version=1,
                                      benchmark_rule=benchmark_rule,
                                      parsed=parsed)
        except Exception as exc:                                # noqa: BLE001
            family.errors.append(f"{type(exc).__name__}: {exc}")
            continue
        if result.scenario is None:
            family.errors.append("no scenario produced")
            continue
        family.rule_hashes.append(result.scenario.rule_hash)
    return family


def run(strategies: Sequence[Strategy], count: int, *, parser=None
        ) -> List[Family]:
    families = (run_family(s, count, parser=parser) for s in strategies)
    return [f for f in families if f is not None]


def summarize(families: Sequence[Family]) -> Dict[str, Any]:
    stable = [f for f in families if f.stable]
    return {
        "families": len(families),
        "wordings": sum(len(f.wordings) for f in families),
        "stable": len(stable),
        "unstable": len(families) - len(stable),
        "stability_rate": round(100.0 * len(stable) / len(families), 1)
        if families else 100.0,
        "worst": sorted(((len(f.distinct), f.strategy_id) for f in families),
                        reverse=True)[:5],
    }
