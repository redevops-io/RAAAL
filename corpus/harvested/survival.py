"""Material-semantic survival: what happened to each thing a sentence asserted.

    python corpus/harvested/survival.py

**Why this and not a pass rate.** "37 of 103 executed" answers a question nobody
should ask. A strategy this build cannot model, refused by name, is a correct
outcome; a strategy it half-models and runs is a disaster; and a pass rate
scores them the same way round as the wrong one. So the unit here is not the
sentence, it is the *material semantic* — one thing the person asserted — and
each one ends in exactly one of three states.

    HONOURED   the plan carries it
    NAMED      the runtime cannot do it and said so, by name
    DROPPED    neither

`HONOURED` and `NAMED` both count as survival. That is the whole point: a
person told "this build cannot hold assets in a named account" still knows what
happened to that clause of their sentence. A person told nothing does not, and
the plan they are shown is missing something they asked for with no mark where
it was.

**A caution about what the denominator is.** These 29 sentences are the
strategy statements in a 220-sentence harvest of forum prose. They are attested
and they are not representative of what someone types into a strategy box —
see `annotate.py`. A survival rate here is evidence about language of this
kind, not a claim about the pilot's population.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

ANNOTATIONS = HERE / "annotations.json"
OUT = HERE / "survival.json"

HONOURED, NAMED, DROPPED = "HONOURED", "NAMED", "DROPPED"

#: The sentence stopped for some *other* reason, so this concept never got a
#: verdict at all.
#:
#: Added after the first run counted these as `DROPPED`, which overstated the
#: danger considerably. "I contribute $750/month to my 401k" is asked about —
#: it never says what to buy — and no plan is produced. The account clause did
#: not survive, and it also did not silently become part of a running plan that
#: omits it. Those are different things and a metric that scores them the same
#: way cannot be used to decide what to fix.
#:
#: `DROPPED` now means the narrow, dangerous case only: a plan executed and the
#: concept is not in it and nothing said so.
NOT_REACHED = "NOT_REACHED"


def _readers():
    from src.discovery.hosted_recording import RecordedHostedReader
    from src.discovery.syntax_stanza import RecordedReader

    return RecordedHostedReader(), RecordedReader()


def fate(concept: str, dimension, settled: set, refusals: set,
         questions: set, executed: bool) -> str:
    """What became of one asserted concept.

    `executed` is load-bearing and was missing from the first version. A
    dimension sitting in `settled` means Discovery *recognised* it, which is
    not the same as the plan carrying it — that was the entire rotation defect,
    where the selection was read correctly and then dropped on the way to a
    plan. Crediting recognition as survival would have scored the runtime on
    the half of the journey that was never in question.

    Reading it the other way round is safe now: the compiler refuses any
    settled dimension no part of it consults, so a plan that executed did
    consult everything settled.

    `dimension is None` means this build has no dimension that could carry the
    concept at all. Such a concept can still be `NAMED` — "this build does not
    model contribution limits" is the honest answer — but never `HONOURED`, and
    treating an unrepresentable concept as honoured because the plan happened
    to compile is the silent reduction this metric exists to count.
    """
    if dimension is not None and (dimension in refusals or dimension in questions):
        return NAMED
    if executed:
        if dimension is None:
            return DROPPED
        return HONOURED if dimension in settled else DROPPED
    # No plan. Something was refused or asked about, but not this.
    return NOT_REACHED


def main() -> int:
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.witnesses import BOTH
    from src.workspace.pilot import read

    document = json.loads(ANNOTATIONS.read_text())
    maps_to = document["maps_to"]
    model, syntax = _readers()

    rows, tally = [], {HONOURED: 0, NAMED: 0, DROPPED: 0, NOT_REACHED: 0}
    unrecorded = []
    for entry in document["annotations"]:
        try:
            reading = read(entry["text"], model, schema=QUANTIFY_SCHEMA,
                           profile=BOTH, syntax_reader=syntax)
        except Exception as failure:                            # noqa: BLE001
            # Recorded, never skipped. A sentence that vanishes from the
            # denominator raises the rate by exactly the cases that went worst.
            unrecorded.append({"text": entry["text"],
                               "failed": f"{type(failure).__name__}: "
                                         f"{str(failure)[:120]}"})
            continue

        settled = {f.field for f in reading.settled if f.value is not None}
        compiled = getattr(reading, "compiled", None)
        refusals = {getattr(r, "dimension", "")
                    for r in getattr(compiled, "refusals", ())}
        questions = set(reading.questions)

        fates = {}
        for concept in entry["material"]:
            state = fate(concept, maps_to.get(concept), settled, refusals,
                         questions, bool(getattr(compiled, "scenario", None)))
            fates[concept] = state
            tally[state] += 1

        rows.append({
            "stopped_by": sorted(refusals | questions),
            "text": entry["text"],
            "interpretation": entry["interpretation"],
            "expected_disposition": entry["disposition"],
            "executed": bool(getattr(compiled, "scenario", None)),
            "fates": fates,
            "source": entry["source"]})

    total = sum(tally.values())
    survived = tally[HONOURED] + tally[NAMED]
    # The denominator excludes what never got a verdict. Counting NOT_REACHED
    # as a failure would blame the runtime for stopping, which is the outcome
    # this project prefers; counting it as a success would credit it for a
    # question it answered about something else.
    adjudicated = total - tally[NOT_REACHED]

    # What stopped the sentences that produced no plan, ranked.
    #
    # This turned out to matter more than the survival rate. 27 of 29 attested
    # strategy statements never reach a plan, so the rate is computed over a
    # handful of semantics and a headline percentage taken from it would be
    # arithmetic rather than evidence. What the sentences are stopped *by* is
    # the thing with enough instances behind it to act on.
    stopped_by: dict = {}
    for row in rows:
        if row["executed"]:
            continue
        for name in row["stopped_by"]:
            stopped_by[name] = stopped_by.get(name, 0) + 1

    # Which concepts get dropped, ranked. This is the improvement queue: a
    # concept dropped once is an anecdote, a concept dropped in every sentence
    # that asserts it is a gap with a name.
    dropped_by_concept: dict = {}
    for row in rows:
        for concept, state in row["fates"].items():
            if state == DROPPED:
                dropped_by_concept[concept] = \
                    dropped_by_concept.get(concept, 0) + 1

    OUT.write_text(json.dumps({
        "schema": "quantify-material-survival@1",
        "sentences": len(rows),
        "material_semantics": total,
        "tally": tally,
        "survival_rate": round(survived / adjudicated, 4) if adjudicated else None,
        "adjudicated": adjudicated,
        "unrecorded": unrecorded,
        "reached_a_plan": sum(1 for r in rows if r["executed"]),
        "stopped_by": dict(sorted(stopped_by.items(), key=lambda kv: -kv[1])),
        "dropped_by_concept": dict(sorted(dropped_by_concept.items(),
                                          key=lambda kv: -kv[1])),
        "metric_note": (
            "HONOURED + NAMED over everything adjudicated. A refusal by name "
            "is survival: the person knows what became of that clause. Silence "
            "is not, whatever the plan looks like."),
        "caution": (
            "A survival rate near 100% with no sentence reaching a plan is not "
            "a good result and must not be quoted as one. Nothing was reduced "
            "because nothing ran. It says this build is safe on language of "
            "this kind and cannot yet model it, which are two findings and "
            "only one of them is comfortable."
            if not any(r["executed"] for r in rows) else
            "Some sentences reached a plan, so the rate is measuring what it "
            "claims to measure."),
        "population_note": document["yield_note"],
        "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n")

    print(f"{len(rows)} sentences, {total} material semantics")
    for state in (HONOURED, NAMED, DROPPED, NOT_REACHED):
        print(f"  {state:12} {tally[state]}")
    print(f"  survival     {survived}/{adjudicated} = "
          f"{survived / adjudicated:.1%}" if adjudicated else "  no semantics")
    if unrecorded:
        print(f"\n{len(unrecorded)} sentences could not be read and are "
              "recorded as such, not dropped")
    reached = sum(1 for r in rows if r["executed"])
    print(f"\n{reached}/{len(rows)} reached a plan")
    if not reached:
        print("  NOTE: the survival rate above is high because nothing ran.")
        print("        Safe, and not yet useful on language of this kind.")
    if stopped_by:
        print("what stopped the rest:")
        for name, count in sorted(stopped_by.items(), key=lambda kv: -kv[1]):
            print(f"  {count:3}  {name}")
    if dropped_by_concept:
        print("\ndropped, by concept:")
        for concept, count in sorted(dropped_by_concept.items(),
                                     key=lambda kv: -kv[1]):
            print(f"  {count:3}  {concept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
