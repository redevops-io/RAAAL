"""Old path against new, with differences classified rather than merely failed.

    Given the same plan and the same relevant market observations, the two
    paths must produce the same evaluator semantics even though they
    intentionally produce different snapshot identities.

**Data identity parity is not expected. Evaluation parity is required.**

The old path resolves the whole panel and lets the engine pick what it needs;
the new one resolves the requested instruments and delivers those. So their
content addresses differ, and that is the new boundary being *more* precise: a
snapshot carrying twenty-two instruments a plan never touches is one whose hash
moves when an unrelated instrument's history is revised, invalidating a stored
figure for no reason.

**Four verdicts, and only three of them pass.**

    EQUIVALENT                     the two agree
    EXPECTED_IDENTITY_DIFFERENCE   they differ because the new path narrows the
                                   dataset on purpose
    APPLICATION_ONLY_DIFFERENCE    they differ outside the evaluator's contract
    EVALUATOR_MISMATCH             they disagree about what the plan does

Classifying rather than whitelisting is the point. A whitelist records that
somebody once decided a difference was fine; a verdict with its reason attached
records *why*, travels with the comparison, and can be read a year later by
somebody deciding whether the reason still holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Sequence, Tuple


class Verdict(str, Enum):
    EQUIVALENT = "EQUIVALENT"
    EXPECTED_IDENTITY_DIFFERENCE = "EXPECTED_IDENTITY_DIFFERENCE"
    APPLICATION_ONLY_DIFFERENCE = "APPLICATION_ONLY_DIFFERENCE"
    EVALUATOR_MISMATCH = "EVALUATOR_MISMATCH"


#: Verdicts a comparison may hold and still pass. `EVALUATOR_MISMATCH` is
#: absent, which is the whole rule.
TOLERATED = (Verdict.EQUIVALENT, Verdict.EXPECTED_IDENTITY_DIFFERENCE,
             Verdict.APPLICATION_ONLY_DIFFERENCE)

#: Identity fields expected to differ, each with the reason encoded here rather
#: than left in a comment. A tolerated difference whose justification lives only
#: in prose is one that outlives its justification.
NARROWED_DATASET = (
    "the new path resolves the requested instruments and delivers only those, "
    "so its content address depends on the evaluation's own inputs. The old "
    "path delivers the whole panel, so its address moves when an unrelated "
    "instrument's history is revised — which would invalidate a stored figure "
    "that never touched that instrument")

EXPECTED_DIFFERENCES: Mapping[str, str] = {
    "market_snapshot_hash": NARROWED_DATASET,
    "market_snapshot_id": NARROWED_DATASET,
    "execution_input_digest": NARROWED_DATASET,
}

#: Identity fields that must be identical. These say *what computed the figure*
#: rather than what it was computed from, and a difference in any of them means
#: the two paths are not the comparison anybody intended.
MUST_MATCH = ("strategy_hash", "evaluator", "evaluator_version",
              "engine_version", "conventions_version",
              "result_schema_version", "evaluation_policy")

#: Differences that are real and outside the evaluator's contract.
#:
#: Benchmarks derive from whichever columns the supplied frame happens to
#: carry, so the two paths build different benchmark sets — application
#: behaviour coupled to the old broad-frame shape, and a separate migration
#: item rather than something allowed to contaminate this comparison.
APPLICATION_ONLY = {
    "benchmarks": "benchmarks are built from the columns present in the "
                  "supplied frame, so a narrowed dataset produces a different "
                  "set. They are application behaviour, not evaluator "
                  "semantics, and their migration is its own item",
}


@dataclass(frozen=True)
class Difference:
    """One field, how the two paths differed, and what that means."""

    field: str
    verdict: Verdict
    reason: str = ""
    old: str = ""
    new: str = ""

    def to_json(self) -> Dict[str, str]:
        return {"field": self.field, "verdict": self.verdict.value,
                "reason": self.reason, "old": self.old, "new": self.new}


@dataclass(frozen=True)
class Comparison:
    """Every field compared, and whether the pair may be called parity."""

    differences: Tuple[Difference, ...]

    @property
    def parity(self) -> bool:
        return all(one.verdict in TOLERATED for one in self.differences)

    @property
    def mismatches(self) -> Tuple[Difference, ...]:
        return tuple(one for one in self.differences
                     if one.verdict is Verdict.EVALUATOR_MISMATCH)

    def by_verdict(self, verdict: Verdict) -> Tuple[Difference, ...]:
        return tuple(one for one in self.differences if one.verdict is verdict)

    def report(self) -> str:
        return "\n".join(
            f"  {one.verdict.value:30} {one.field}"
            + (f"\n      {one.reason}" if one.reason else "")
            + (f"\n      old: {one.old[:110]}\n      new: {one.new[:110]}"
               if one.verdict is Verdict.EVALUATOR_MISMATCH else "")
            for one in self.differences)


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 400 else text[:400] + "…"


def compare(old, new) -> Comparison:
    """Two `EvaluationResult`s, field by field, with every verdict recorded.

    Every compared field appears in the result, equivalent ones included. A
    report listing only differences cannot distinguish "these agreed" from
    "this was never checked", and the second is how a comparison quietly stops
    covering something.
    """
    from .service import STREAMS

    found = []

    for name in MUST_MATCH:
        mine, theirs = getattr(old, name, None), getattr(new, name, None)
        found.append(Difference(
            field=name,
            verdict=(Verdict.EQUIVALENT if mine == theirs
                     else Verdict.EVALUATOR_MISMATCH),
            reason=("" if mine == theirs else
                    "this names what computed the figure rather than what it "
                    "was computed from; a difference here means the two paths "
                    "are not the comparison anybody intended"),
            old=_short(mine), new=_short(theirs)))

    for name, why in EXPECTED_DIFFERENCES.items():
        mine, theirs = getattr(old, name, None), getattr(new, name, None)
        if mine is None and theirs is None:
            continue
        found.append(Difference(
            field=name,
            verdict=(Verdict.EQUIVALENT if mine == theirs
                     else Verdict.EXPECTED_IDENTITY_DIFFERENCE),
            reason=("" if mine == theirs else why),
            old=_short(mine), new=_short(theirs)))

    # The evaluator's contract surface. Compared as `(produced, rows)`, so
    # ordering and counts are part of the comparison rather than a separate
    # check somebody has to remember: a tuple that differs in order differs.
    for name in STREAMS:
        mine = old.streams.get(name)
        theirs = new.streams.get(name)
        same = (mine is not None and theirs is not None
                and (mine.produced, mine.rows) == (theirs.produced, theirs.rows))
        found.append(Difference(
            field=f"stream:{name}",
            verdict=Verdict.EQUIVALENT if same else Verdict.EVALUATOR_MISMATCH,
            reason=("" if same else
                    "the two paths disagree about this stage of the run. The "
                    "streams are the evaluator's contract surface, and a "
                    "difference here is a difference in what the plan did"),
            old=_short(_describe(mine)), new=_short(_describe(theirs))))

    # The publish/refuse disposition, and the reasons when refused. A figure
    # withheld by one path and shown by the other is the most consequential
    # difference there is, and it does not appear in any stream.
    mine, theirs = tuple(old.refusals), tuple(new.refusals)
    found.append(Difference(
        field="refusals",
        verdict=(Verdict.EQUIVALENT if mine == theirs
                 else Verdict.EVALUATOR_MISMATCH),
        reason=("" if mine == theirs else
                "one path withheld a figure the other showed, or refused for a "
                "different reason"),
        old=_short(mine), new=_short(theirs)))

    mine, theirs = dict(old.figures or {}), dict(new.figures or {})
    for key in sorted(set(mine) | set(theirs)):
        if key in APPLICATION_ONLY:
            found.append(Difference(
                field=f"figure:{key}", verdict=Verdict.APPLICATION_ONLY_DIFFERENCE,
                reason=APPLICATION_ONLY[key],
                old=_short(mine.get(key)), new=_short(theirs.get(key))))
            continue
        same = mine.get(key) == theirs.get(key)
        found.append(Difference(
            field=f"figure:{key}",
            verdict=Verdict.EQUIVALENT if same else Verdict.EVALUATOR_MISMATCH,
            reason=("" if same else
                    "a canonical output derived from the streams differs"),
            old=_short(mine.get(key)), new=_short(theirs.get(key))))

    return Comparison(differences=tuple(found))


def _describe(stream) -> str:
    if stream is None:
        return "absent from this result"
    return (f"produced={stream.produced} rows={len(stream.rows)} "
            f"digest={stream.digest}")
