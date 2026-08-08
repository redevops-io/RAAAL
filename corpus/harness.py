"""The correctness instrument, and the check that it is one.

Two corpora live beside this file:

    strategies.json   35 prompts labelled EXECUTE · CLARIFY · REFUSE · UNKNOWN
    catalogue.json    144 generated prompts, unlabelled, with `declares`

They are in this repository, not in the agent that used to run them, because
Phase 0's gate is that they reproduce **from a clean clone**. The vendor
manifest and the licensing record were gitignored until recently and the tests
that depended on them passed only where someone had fetched them by hand; a
corpus outside the repo is the same failure with a longer fuse.

**The `--validate` verb is the point.** Borrowed from DataOpsBench, whose
framing is exact: *"a gate that passed on the defect would be worthless."* For
every labelled row it asks whether the expectation could fail at all — an
EXECUTE row must actually produce a figure, a REFUSE row must actually be
refused, and a row where neither is reachable is removed or rewritten rather
than counted. A corpus of expectations nothing can violate is a list of
opinions.

    python corpus/harness.py validate      # do the expectations discriminate?
    python corpus/harness.py run           # what does this build do with them?
    python corpus/harness.py baseline out/ # freeze a reference to compare to

Nothing here calls a model. It drives the compiler and the capability manifest
directly, so it runs offline, in CI, and against either implementation once
Discovery exists — which is what makes it a *comparison* instrument rather than
a description of the current build.

**Where it probes, and what that costs.** It stops at the executability
decision on the intent: understood, refused by the manifest, or still asking.
It does not run a simulation. So a prompt the product refuses *later* — at
coverage, once a declared element turns out not to have been executed — reads
`CLARIFY` here, because at intent time nothing had refused it yet.

That is why the first baseline reads 145 CLARIFY out of 179 and agrees with
only 6 of 24 labelled expectations. Those numbers are **not** a product score
and must never be quoted as one; the corpus labels describe what the product
should ultimately do with a prompt, and this instrument measures one earlier
stage of it. What it is good for is *movement*: the same probe before and after
a change, on the same rows.

The gap is itself worth reading. A row expecting REFUSE that reads CLARIFY here
is one where the manifest could have refused at intent time and did not — which
is either a dimension missing from the manifest, or a refusal the product
defers longer than it needs to. Both are findings, and neither is visible from
the run-time path alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

BENCHMARK_RULE = "benchmark-policy/public-default@1"

#: What a build can do with a prompt. The four the corpus labels, plus the one
#: only the harness can report.
EXECUTE = "EXECUTE"
CLARIFY = "CLARIFY"
REFUSE = "REFUSE"
UNKNOWN = "UNKNOWN"
HARNESS_ERROR = "HARNESS_ERROR"
"""This file failed, not the build. Kept distinct because an evaluator that
reports its own crash as a product refusal manufactures evidence."""


@dataclass(frozen=True)
class Outcome:
    ref: str
    observed: str
    refusals: Sequence[str] = ()
    open_dimensions: Sequence[str] = ()
    intent_hash: str = ""
    produced_by: str = ""
    detail: str = ""
    expectation: Optional[str] = None

    @property
    def agrees(self) -> Optional[bool]:
        """None when the row carries no expectation — 144 of them do not, and
        `False` there would invent a failure."""
        if self.expectation in (None, "", UNKNOWN):
            return None
        return self.observed == self.expectation

    def to_json(self) -> Dict[str, Any]:
        return {"ref": self.ref, "observed": self.observed,
                "expectation": self.expectation, "agrees": self.agrees,
                "refusals": list(self.refusals),
                "open_dimensions": list(self.open_dimensions),
                "intent_hash": self.intent_hash,
                "produced_by": self.produced_by, "detail": self.detail}


def evaluate(prompt: str, ref: str = "", expectation: Optional[str] = None) -> Outcome:
    """What this build does with one prompt, without running a simulation.

    Deliberately stops at the executability decision. Whether the *figure* is
    right is a different question with a different instrument; what this
    measures is whether the build understood the request and was honest about
    what it could do with it.
    """
    from src.mission.compiler import compile_scenario, parse
    from src.mission.verified_intent import executable_check, from_compiled

    try:
        parsed = parse(prompt)
        result = compile_scenario(prompt, name=ref or "corpus", version=1,
                                  benchmark_rule=BENCHMARK_RULE, parsed=parsed)
        intent = from_compiled(result, parsed)
        refusals = executable_check(intent)
    except Exception as failure:                                  # noqa: BLE001
        return Outcome(ref=ref, observed=HARNESS_ERROR,
                       detail=f"{type(failure).__name__}: {failure}",
                       expectation=expectation)

    open_dimensions = tuple(u.dimension for u in intent.unresolved)
    if refusals:
        observed = REFUSE
    elif open_dimensions:
        observed = CLARIFY
    else:
        observed = EXECUTE

    return Outcome(
        ref=ref, observed=observed,
        refusals=tuple(r.dimension for r in refusals),
        open_dimensions=open_dimensions,
        intent_hash=intent.intent_hash, produced_by=intent.produced_by,
        detail="; ".join(r.message for r in refusals)[:400],
        expectation=expectation)


# --- the corpora -----------------------------------------------------------

def labelled() -> List[Dict[str, Any]]:
    return json.loads((HERE / "strategies.json").read_text())


def catalogue() -> List[Dict[str, Any]]:
    return json.loads((HERE / "catalogue.json").read_text())


# --- validate: do the expectations discriminate? ---------------------------

#: Mutations applied to a prompt to check its expectation can be violated.
#:
#: Each one is a *semantic* change, not a typo: a corpus row whose expectation
#: survives replacing its whole strategy was never testing that strategy.
_MUTATIONS = {
    EXECUTE: ("by inverse volatility", REFUSE),
    REFUSE: (None, None),
}


def validate() -> int:
    """Prove each labelled expectation could fail.

    An EXECUTE row is mutated into something the manifest refuses; if it still
    reads EXECUTE, the row is not measuring executability. A REFUSE row is
    checked against the build directly; a REFUSE that nothing refuses is an
    opinion about a future build.
    """
    rows = labelled()
    checked = failed = 0
    print(f"validating {len(rows)} labelled prompts\n")

    for row in rows:
        expectation = row["expectation"]
        if expectation not in (EXECUTE, REFUSE):
            continue
        checked += 1
        base = evaluate(row["prompt"], row["ref"], expectation)

        if expectation == EXECUTE:
            mutated = evaluate(row["prompt"].rstrip(". ") + " by inverse volatility.",
                               row["ref"], expectation)
            discriminates = mutated.observed != base.observed
            why = (f"base={base.observed} mutated={mutated.observed}")
        else:
            discriminates = base.observed in (REFUSE, CLARIFY)
            why = f"observed={base.observed}"

        if not discriminates:
            failed += 1
            print(f"  NOT DISCRIMINATING  {row['ref']:10s} {expectation:8s} {why}")
            print(f"                      {row['prompt'][:88]}")

    print(f"\nRESULT {checked - failed}/{checked} expectations discriminate")
    if failed:
        print("\n  A row nothing can violate is an opinion, not evidence. "
              "Rewrite or remove it rather than counting it.")
    return 1 if failed else 0


# --- run / baseline --------------------------------------------------------

def run(out: Optional[Path] = None) -> int:
    rows = [dict(r, _corpus="strategies") for r in labelled()]
    rows += [dict(r, _corpus="catalogue", expectation=None) for r in catalogue()]

    outcomes = [evaluate(r["prompt"], r["ref"], r.get("expectation"))
                for r in rows]

    tally: Dict[str, int] = {}
    for o in outcomes:
        tally[o.observed] = tally.get(o.observed, 0) + 1

    judged = [o for o in outcomes if o.agrees is not None]
    agreed = sum(1 for o in judged if o.agrees)

    print(f"{len(outcomes)} prompts")
    for state in (EXECUTE, CLARIFY, REFUSE, HARNESS_ERROR):
        if tally.get(state):
            print(f"  {state:14s} {tally[state]:4d}")
    print(f"\n  against expectation: {agreed}/{len(judged)} agree "
          f"({len(outcomes) - len(judged)} rows carry no expectation)")
    print("\n  NOT A PRODUCT SCORE. This probes the intent-stage decision, so a"
          "\n  prompt the product refuses later — at coverage, once a declared"
          "\n  element turns out not to have executed — reads CLARIFY here. Use"
          "\n  these numbers to compare two builds on the same rows, never to"
          "\n  describe one.")

    deferred = [o for o in judged
                if o.expectation == REFUSE and o.observed == CLARIFY]
    if deferred:
        print(f"\n  {len(deferred)} rows expect REFUSE and read CLARIFY: the"
              "\n  manifest could have refused at intent time and did not."
              "\n  Either a dimension is missing from it, or the refusal is"
              "\n  deferred further than it needs to be. Both are findings.")

    if any(o.observed == HARNESS_ERROR for o in outcomes):
        print("\n  HARNESS_ERROR rows are this file failing, not the build. "
              "They are not refusals and must not be counted as any result.")

    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "outcomes.json").write_text(
            json.dumps([o.to_json() for o in outcomes], indent=1))
        print(f"\nwrote {out}/outcomes.json")
    return 0


def compare(baseline: Path) -> int:
    """This build against a frozen reference, by meaning rather than by count.

    A prompt that moved EXECUTE -> REFUSE is a regression to explain. One that
    moved REFUSE -> EXECUTE needs its figure checked by hand before it counts
    as an improvement, because the cheapest way to raise the execute rate is to
    stop refusing things.
    """
    before = {o["ref"]: o for o in json.loads(
        (baseline / "outcomes.json").read_text())}
    rows = [dict(r) for r in labelled()] + [
        dict(r, expectation=None) for r in catalogue()]
    moved = []
    for row in rows:
        now = evaluate(row["prompt"], row["ref"], row.get("expectation"))
        was = before.get(row["ref"])
        if was and was["observed"] != now.observed:
            moved.append((row["ref"], was["observed"], now.observed))

    if not moved:
        print("no prompt changed outcome")
        return 0

    print(f"{len(moved)} prompts changed outcome\n")
    for ref, was, now in moved:
        arrow = "REGRESSION" if (was == EXECUTE and now != EXECUTE) else \
                "needs a hand-checked figure" if now == EXECUTE else "changed"
        print(f"  {ref:10s} {was:8s} -> {now:8s}  {arrow}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("verb", choices=("validate", "run", "baseline", "compare"))
    parser.add_argument("out", nargs="?", type=Path)
    args = parser.parse_args()

    if args.verb == "validate":
        return validate()
    if args.verb == "run":
        return run(args.out)
    if args.verb == "baseline":
        return run(args.out or (HERE / "baseline"))
    return compare(args.out or (HERE / "baseline"))


if __name__ == "__main__":
    raise SystemExit(main())
