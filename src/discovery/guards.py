"""Structural proof that a material action is present, and nothing more.

A guard answers exactly one question:

    is a material predicate explicitly present in this sentence?

It never answers *what* the predicate means. `sell the loser and buy a similar
fund to avoid a wash sale` has `sell` as its root verb with `loser` as its
object, and that is enough to assert SELL_PRESENT. Turning it into
`sell_action="sell the loser"` would be inferring the value, which is the
model's job and the exact step that would make this a second domain compiler.

**Why presence alone is worth having.** Four of five live draws read that
sentence's `sell_action` and Mission refused it by name. The fifth read no sell
at all and produced an executable plan. The dimension exists, the refusal
exists, and the reader simply dropped it — so nothing downstream had anything
to refuse. Repeated over a cohort that is one person in five getting a figure
for a strategy they did not describe.

The rule this implements:

    if deterministic syntax strongly proves a material dimension is explicitly
    present, Discovery may not seal an intent that omits it

which is enforced through the fusion contract that already exists rather than a
new one. "Syntax proposed a value the model never mentioned" is already
`DISAGREE`, and `DISAGREE` does not proceed. A guard emits that, so the
dimension becomes an open question instead of a silence.

**Deliberately small.** One lemma set per already-refusable dimension, matched
only in predicate position. It cannot grow into a parser by accident: adding a
guard means adding a dimension the manifest already refuses and a verb list,
and `tests/test_syntax_guards.py` asserts both halves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: Dependency relations a predicate occupies. `root` is the main verb;
#: `conj` is the second verb of a coordination, so "sell X and buy Y" is caught
#: whichever way the parser attaches the pair; `xcomp`/`advcl` carry a verb
#: that governs its own clause.
PREDICATE_RELATIONS = frozenset({"root", "conj", "xcomp", "advcl", "ccomp"})


@dataclass(frozen=True)
class SyntaxGuard:
    """One dimension, and the verbs whose presence Discovery may not lose."""

    dimension: str
    lemmas: frozenset
    why: str

    def proves_presence(self, parse) -> bool:
        """Whether this sentence contains one of these verbs as a predicate.

        Position matters. "sell" in predicate position is somebody selling;
        the same token as a noun modifier — "a sell signal", "the sell side" —
        is not, and a guard that fired on it would make every trigger sentence
        unsealable.
        """
        for sentence in getattr(parse, "sentences", ()):
            for token in sentence.tokens:
                if token.upos != "VERB":
                    continue
                if token.relation not in PREDICATE_RELATIONS:
                    continue
                if token.lemma.lower() in self.lemmas:
                    return True
        return False


#: Every guard. One per dimension the manifest already refuses, because a guard
#: on something executable would only ever manufacture questions.
GUARDS: Sequence[SyntaxGuard] = (
    SyntaxGuard(
        dimension="sell_action",
        lemmas=frozenset({
            "sell", "withdraw", "harvest", "convert", "annuitize", "annuitise",
            "liquidate", "redeem", "divest", "drawdown",
        }),
        why="this build only buys, so a sentence that disposes of something "
            "must not compile into one that accumulates"),

    SyntaxGuard(
        dimension="periodic_rebalancing",
        lemmas=frozenset({"rebalance", "reallocate"}),
        why="this build buys and holds; a rebalance that vanished would run as "
            "a plain contribution plan"),
)


def presence(parse) -> Mapping[str, SyntaxGuard]:
    """Dimensions this sentence structurally proves are present."""
    return {guard.dimension: guard for guard in GUARDS
            if guard.proves_presence(parse)}


def missing(parse, decisions: Sequence[Any]) -> Sequence[SyntaxGuard]:
    """Guarded dimensions syntax proves present that fusion has no reading for.

    A dimension already *open* is not missing — it is being asked about, which
    is what a guard wants. Only silence counts: the reader neither settled it
    nor raised it, so nothing downstream would ever mention it.
    """
    spoken = {d.dimension for d in decisions}
    return tuple(guard for dimension, guard in presence(parse).items()
                 if dimension not in spoken)


def as_decisions(parse, decisions: Sequence[Any]) -> Sequence[Any]:
    """One `DISAGREE` per guarded dimension the reader dropped.

    `DISAGREE` rather than a new outcome, because the fusion contract already
    means exactly this: syntax has something to say and the model does not
    corroborate it, so the dimension does not proceed and the intent cannot
    seal. Adding an outcome would have given the same behaviour a second name.

    No value is proposed. The guard proves presence and stops; what the
    predicate *means* stays the reader's answer, and a sealed intent will never
    contain a value this module invented.
    """
    from .fusion import Decision, Fusion
    from .syntax import SyntaxEvidence

    out = []
    for guard in missing(parse, decisions):
        out.append(Decision(
            dimension=guard.dimension,
            outcome=Fusion.DISAGREE,
            value=None,
            material=True,
            model=None,
            syntax=(SyntaxEvidence(
                dimension=guard.dimension,
                proposed_value=None,
                score=1,
                features=("predicate_present",),
                source_span=""),),
            detail=("syntax found this action stated in the sentence and the "
                    f"reader did not report it — {guard.why}")))
    return tuple(out)
