"""How each strategy family fares, measured rather than asserted.

    python corpus/parser/strategy_closure.py          # writes strategy_closure.json
    python corpus/parser/strategy_closure.py --print  # and shows the table

One state per case, and the middle one is the whole reason this exists:

    CARRIED             the dimension that should carry the request was read.
                        For a REFUSED_BY_NAME family that is success — Mission
                        can now refuse the thing the person asked for.

    SILENTLY_REDUCED    the carrying dimension was NOT read, but something else
                        was. The sentence produces a plan built from whatever
                        fragment survived, and every surviving fragment in this
                        build is accumulation-shaped. This is the dangerous
                        state: no refusal, no approximation flagged as one, and
                        a figure at the end of it.

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

CARRIED = "CARRIED"
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


def measure() -> dict:
    sys.path.insert(0, str(HERE.parent.parent))

    from src.discovery.readers_quantify import CompilerReader
    from src.discovery.schema import QUANTIFY_SCHEMA

    document = json.loads(CASES.read_text())
    reader = CompilerReader()

    results, by_state, by_family = [], {}, {}
    for case in document["cases"]:
        got = _read(reader, QUANTIFY_SCHEMA, case["text"])
        if not case["carriers"]:
            state = SCHEMA_GAP
        elif any(c in got for c in case["carriers"]):
            state = CARRIED
        elif got:
            state = SILENTLY_REDUCED
        else:
            state = NOTHING_READ

        results.append({**case, "state": state, "read": got})
        by_state[state] = by_state.get(state, 0) + 1
        family = by_family.setdefault(case["family"], {})
        family[state] = family.get(state, 0) + 1

    reduced = [r for r in results if r["state"] == SILENTLY_REDUCED]
    return {
        "schema": "quantify-strategy-closure@1",
        "witness": reader.id,
        "witness_note": (
            "The deterministic reader only. Says nothing about what the model "
            "reader sees — and under MODEL_ONLY, the pilot profile, the model "
            "is the only witness, so nothing would catch it missing the same "
            "dimension."),
        "count": len(results),
        "by_state": by_state,
        "by_family": by_family,
        "silently_reduced": [
            {"id": r["id"], "text": r["text"],
             "should_carry": "/".join(r["carriers"]), "read_instead": r["read"]}
            for r in reduced],
        "cases": results}


def main(show: bool = False) -> int:
    report = measure()
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    print(f"{report['count']} cases, witness {report['witness']}")
    for state in (CARRIED, SILENTLY_REDUCED, NOTHING_READ, SCHEMA_GAP):
        print(f"  {state:18} {report['by_state'].get(state, 0)}")

    if show:
        print("\nsilently reduced — a plan is produced and it is the wrong one:")
        for one in report["silently_reduced"]:
            print(f"  {one['text'][:58]:60}")
            print(f"    should carry {one['should_carry']:22} "
                  f"read instead {one['read_instead']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(show="--print" in sys.argv))
