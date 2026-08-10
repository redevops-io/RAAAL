"""How each strategy family fares, measured rather than asserted.

    python corpus/parser/strategy_closure.py          # writes strategy_closure.json
    python corpus/parser/strategy_closure.py --print  # and shows the table

**What is measured is the refusal, not a dimension.** The first version scored
a case CARRIED when one nominated dimension appeared in the reading. That is a
proxy for the thing that matters and it was wrong in both directions: it called
asset location understood because `account_type` happened to appear, and it
called a withdrawal unhandled when the model had read
`objective=assess_withdrawal` and Mission would refuse on it. The rule this
tier exists to enforce is:

    if the engine cannot model the semantic, Discovery must preserve enough of
    it for Mission to refuse it BY NAME

so the question asked of each case is now `refusals_for(reading)` — does asking
Mission about what was read produce a refusal. One state per case:

    REFUSED             Mission refuses this, by name. Success for an
                        unsupported family regardless of which dimension
                        carried it.

    EXECUTABLE          the family is supported and Mission raises no
                        objection. Success for a supported family.

    SILENTLY_REDUCED    something was read, the family is unsupported, and
                        Mission refuses nothing. The sentence produces a plan
                        built from whatever fragment survived, and every
                        surviving fragment in this build is
                        accumulation-shaped. This is the dangerous state: no
                        refusal, no approximation flagged as one, and a figure
                        at the end of it.

    NOTHING_READ        nothing at all was recognised. Honest failure. The
                        person is told the sentence could not be read, which is
                        wrong-but-safe rather than wrong-and-confident.

`SILENTLY_REDUCED` is ranked worse than `NOTHING_READ` on purpose. A sentence
that produces nothing sends somebody back to rephrase; a sentence that produces
the wrong plan sends them away with a number.

**What this measures and what it does not.** The witness here is the
deterministic `CompilerReader`, because it needs no provider and no recordings,
so this report runs anywhere. It says nothing about what the *model* reader
sees — under `MODEL_ONLY`, which is the pilot profile, the model is the only
witness and there is no second one to catch it missing the same thing. Closing
that is a separate pass that needs hosted recordings for these sentences.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "strategy_families.json"
OUT = HERE / "strategy_closure.json"

REFUSED = "REFUSED"
EXECUTABLE = "EXECUTABLE"
SILENTLY_REDUCED = "SILENTLY_REDUCED"
NOTHING_READ = "NOTHING_READ"

#: The schema has no dimension for this, so recognition cannot be the problem
#: and cannot be the fix. Scored apart from the rest because counting it as a
#: recognition defect would send the work to the wrong layer.
SCHEMA_GAP = "SCHEMA_GAP"


def _read(reader, schema, text: str) -> dict:
    result = reader.read(text, schema)
    if getattr(result, "failed", ""):
        return {}
    return {r.dimension: r.value for r in result.readings}


#: The reader a deployment actually serves. `compiler` is retained only as a
#: historical comparator: nothing in `src/` constructs `CompilerReader`, and
#: `test_strategy_families` asserts that structurally.
SERVING = "hosted"


def _witness(name: str):
    """Either reader, so the same measurement can be run against both.

    The whole point of running it twice: 11 silent reductions against the
    deterministic reader is a finding about `quantify-compiler@2`, which is the
    reader being deleted. Whether it is also a finding about Discovery is a
    different question with a different answer and a different fix.
    """
    if name == "hosted":
        from src.discovery.hosted_recording import RecordedHostedReader

        return RecordedHostedReader()

    from src.discovery.readers_quantify import CompilerReader

    return CompilerReader()


def measure(witness: str = SERVING) -> dict:
    sys.path.insert(0, str(HERE.parent.parent))

    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.mission.capability import refusals_for

    document = json.loads(CASES.read_text())
    reader = _witness(witness)

    results, by_state, by_family = [], {}, {}
    for case in document["cases"]:
        got = _read(reader, QUANTIFY_SCHEMA, case["text"])
        refusals = refusals_for(got)
        supported = case["must_be"] == "RECOGNISED"

        if not case["carriers"] and not supported:
            state = SCHEMA_GAP
        elif refusals:
            state = REFUSED
        elif supported:
            state = EXECUTABLE
        elif got:
            state = SILENTLY_REDUCED
        else:
            state = NOTHING_READ

        results.append({**case, "state": state, "read": got,
                        "refused": [f"{r.dimension}={r.stated_value!r}"
                                    for r in refusals]})
        by_state[state] = by_state.get(state, 0) + 1
        family = by_family.setdefault(case["family"], {})
        family[state] = family.get(state, 0) + 1

    reduced = [r for r in results if r["state"] == SILENTLY_REDUCED]
    return {
        "schema": "quantify-strategy-closure@1",
        "witness": reader.id,
        "witness_note": (
            "One reader. Under MODEL_ONLY, the pilot profile, the model is "
            "also the only witness, so nothing would catch it missing a "
            "dimension either."),
        "count": len(results),
        "by_state": by_state,
        "by_family": by_family,
        "silently_reduced": [
            {"id": r["id"], "text": r["text"],
             "should_carry": "/".join(r["carriers"]), "read_instead": r["read"]}
            for r in reduced],
        "cases": results}


def main(show: bool = False, witness: str = "compiler") -> int:
    report = measure(witness)
    out = OUT if witness == SERVING else OUT.with_name(
        "strategy_closure_compiler.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"{report['count']} cases, witness {report['witness']}")
    for state in (REFUSED, EXECUTABLE, SILENTLY_REDUCED, NOTHING_READ,
                  SCHEMA_GAP):
        print(f"  {state:18} {report['by_state'].get(state, 0)}")

    if show:
        print("\nsilently reduced — a plan is produced and it is the wrong one:")
        for one in report["silently_reduced"]:
            print(f"  {one['text'][:58]:60}")
            print(f"    should carry {one['should_carry']:22} "
                  f"read instead {one['read_instead']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(
        show="--print" in sys.argv,
        witness="compiler" if "--compiler" in sys.argv else SERVING))
