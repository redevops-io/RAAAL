"""Why each pending case is still pending — the metric, not the count.

    python corpus/parser/closure.py

"How many cases left `AWAITING_A_PARSER`" is a number that can be moved by
writing rules. "Why is each one still there" is a question that cannot, and it
is the one worth asking: a case blocked on a missing model reading and a case
blocked on a missing relation want different work, and a shrinking count hides
which.

Every pending case lands in exactly one state:

    MAPPED_AND_AGREED     a candidate was proposed and fusion let it through
    MAPPED_BUT_DISAGREED  a candidate was proposed and fusion refused it
    INSUFFICIENT_RELATION the value exists; nothing binds it
    AMBIGUOUS_BY_LANGUAGE the words carry both readings
    STILL_UNSUPPORTED     no normalised value, so no binding and no candidate

`STILL_UNSUPPORTED` is not a failure of this pipeline and must not be read as
one. "weight by inverse volatility" has no literal in it at all; there is
nothing for a normaliser to find and nothing for a binder to attach. Those
cases belong to the semantic reader, and naming them keeps the deterministic
layers from being blamed for work that was never theirs — or, worse, from
growing a rule to claim it.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from corpus.parser.loader import load                       # noqa: E402
from src.discovery.binding import BindingStatus, bind       # noqa: E402
from src.discovery.fusion import Fusion, Proposal, fuse     # noqa: E402
from src.discovery.semantics import propose                 # noqa: E402
from src.discovery.syntax import normalize                  # noqa: E402
from src.discovery.syntax_stanza import RecordedReader      # noqa: E402

OUT = Path(__file__).resolve().parent / "closure.json"

MAPPED_AND_AGREED = "MAPPED_AND_AGREED"
MAPPED_BUT_DISAGREED = "MAPPED_BUT_DISAGREED"
INSUFFICIENT_RELATION = "INSUFFICIENT_RELATION"
AMBIGUOUS_BY_LANGUAGE = "AMBIGUOUS_BY_LANGUAGE"
STILL_UNSUPPORTED = "STILL_UNSUPPORTED"
NO_PARSE = "NO_PARSE_RECORDED"

STATES = (MAPPED_AND_AGREED, MAPPED_BUT_DISAGREED, INSUFFICIENT_RELATION,
          AMBIGUOUS_BY_LANGUAGE, STILL_UNSUPPORTED, NO_PARSE)


def classify(case, recorded: RecordedReader) -> dict:
    """One case, one state, and the reason in the case's own terms."""
    if not recorded.has(case.text, case.language):
        return {"state": NO_PARSE,
                "reason": f"no {case.language} model has been fetched"}

    values = normalize(case.text, case.language)
    if not values:
        return {"state": STILL_UNSUPPORTED,
                "reason": "no normalised value in the sentence; this needs the "
                          "semantic reader, not a deterministic rule"}

    parse = recorded.parse(case.text, case.language)
    bindings = bind(parse, values)
    candidates = propose(bindings, values)

    if not candidates:
        unbound = [b for b in bindings if not b.established]
        if any(b.status is BindingStatus.AMBIGUOUS for b in unbound):
            return {"state": INSUFFICIENT_RELATION,
                    "reason": "the structure offers more than one target and "
                              "does not choose between them"}
        if unbound:
            return {"state": INSUFFICIENT_RELATION,
                    "reason": "a value was normalised and nothing binds it"}
        return {"state": STILL_UNSUPPORTED,
                "reason": "values normalised but no declared mapping consumes "
                          "this relation for this field"}

    # A candidate exists — but for *which* field?
    #
    # The first version of this took `candidates[0]` when nothing matched the
    # field the case asserts, and reported MAPPED_AND_AGREED for six cases that
    # had been answered with something else entirely: "when SPY crosses below
    # its 200-day average" came back as a 200-day holding period and counted as
    # a success for trigger semantics. That is the comparator-manufactures-
    # agreement defect this project has hit before, reproduced inside the
    # report built to measure it.
    wanted = case.asserts.get("field")
    if wanted is None:
        return {"state": STILL_UNSUPPORTED,
                "reason": "this case asserts a role pair rather than a single "
                          "field; no mapping produces role pairs yet",
                "proposed": sorted({c.field for c in candidates})}

    match = next((c for c in candidates if c.field == wanted), None)
    if match is None:
        return {"state": STILL_UNSUPPORTED,
                "reason": f"candidates were proposed for "
                          f"{sorted({c.field for c in candidates})} and none "
                          f"for {wanted!r}, which is what this case asserts",
                "proposed": sorted({c.field for c in candidates})}
    decision = fuse(match.field,
                    model=Proposal(match.field, match.value,
                                   "deterministic-stand-in@1", case.text))

    if decision.outcome is Fusion.AMBIGUOUS_BY_LANGUAGE:
        return {"state": AMBIGUOUS_BY_LANGUAGE, "reason": decision.detail,
                "field": match.field}
    if decision.proceeds:
        # Fusion's outcome and the corpus's expectation are different axes, and
        # both are reported. "Fusion agreed" says the pipeline was internally
        # consistent; `matches_expected` says it was also right. A state name
        # carrying only the first would be a green number for a wrong value —
        # which is how the six false positives above read before the value
        # check existed.
        produced, expected = _plain(match.value), case.asserts.get("value")
        return {"state": MAPPED_AND_AGREED, "field": match.field,
                "value": produced, "expected": expected,
                "matches_expected": str(produced) == str(expected),
                "reason": "a candidate was proposed and fusion let it through"}
    return {"state": MAPPED_BUT_DISAGREED, "field": match.field,
            "reason": decision.detail}


def _plain(value):
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def main() -> int:
    recorded = RecordedReader()
    pending = [c for c in load()
               if c.tier == "semantics"
               or (c.tier == "dependency" and not recorded.has(c.text, c.language))]

    rows = []
    for case in pending:
        outcome = classify(case, recorded)
        rows.append({"id": case.id, "property": case.property,
                     "text": case.text, "language": case.language, **outcome})

    counts = Counter(row["state"] for row in rows)
    agreed = [r for r in rows if r["state"] == MAPPED_AND_AGREED]
    wrong = [r["id"] for r in agreed if not r.get("matches_expected")]
    by_property: dict = {}
    for row in rows:
        by_property.setdefault(row["property"], Counter())[row["state"]] += 1

    OUT.write_text(json.dumps(
        {"schema": "quantify-parser-closure@1",
         "pending": len(rows),
         "by_state": dict(counts),
         "agreed_and_correct": len(agreed) - len(wrong),
         "agreed_but_wrong_value": wrong,
         "by_property": {k: dict(v) for k, v in sorted(by_property.items())},
         "note": ("STILL_UNSUPPORTED is not a defect of the deterministic "
                  "layers. It names cases with no literal to normalise, which "
                  "belong to the semantic reader — recorded so they are not "
                  "confused with cases a rule could close."),
         "rows": rows}, indent=2, ensure_ascii=False) + "\n")

    print(f"{len(rows)} pending cases -> {OUT}")
    for state in STATES:
        if counts.get(state):
            suffix = ""
            if state == MAPPED_AND_AGREED:
                suffix = f"  ({len(agreed) - len(wrong)} with the expected value)"
            print(f"  {state:24} {counts[state]}{suffix}")
    if wrong:
        print(f"  !! agreed with a value the case does not expect: {wrong}")
    print()
    for prop, states in sorted(by_property.items()):
        print(f"  {prop:32} {dict(states)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
