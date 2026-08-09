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
    NO_LITERAL            nothing to normalise; this is the semantic reader's
    NO_FIELD_MAPPING      a literal and a binding exist, and no rule says what
                          that value *means* in the contract

The last two were one state until the counts made the difference matter. They
want opposite work. `weight by inverse volatility` has no literal at all —
there is nothing for a normaliser to find and nothing for a binder to attach,
and no rule written here would change that. `contribute a fixed $500` has both:
`amount=500` is recognised and bound, and what is missing is a rule saying that
"fixed" makes `amount_kind` FIXED. One needs a reader; the other needs field
derivation from structure already in hand.

Neither is a failure of the deterministic layers, and naming them keeps those
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
NO_LITERAL = "NO_LITERAL"
NO_FIELD_MAPPING = "NO_FIELD_MAPPING"
NO_PARSE = "NO_PARSE_RECORDED"

STATES = (MAPPED_AND_AGREED, MAPPED_BUT_DISAGREED, INSUFFICIENT_RELATION,
          AMBIGUOUS_BY_LANGUAGE, NO_FIELD_MAPPING, NO_LITERAL, NO_PARSE)

#: Which layer owns each state's remaining work. Printed with the counts,
#: because "36 unsupported" is a pile and "36 waiting on the semantic reader"
#: is a queue with an owner.
OWNER = {MAPPED_AND_AGREED: "—",
         MAPPED_BUT_DISAGREED: "adjudication",
         INSUFFICIENT_RELATION: "binder or corpus",
         AMBIGUOUS_BY_LANGUAGE: "the user, via clarification",
         NO_FIELD_MAPPING: "semantics.py — field derivation",
         NO_LITERAL: "the semantic reader",
         NO_PARSE: "stanza.download"}


def classify(case, recorded: RecordedReader) -> dict:
    """One case, one state, and the reason in the case's own terms."""
    if not recorded.has(case.text, case.language):
        return {"state": NO_PARSE,
                "reason": f"no {case.language} model has been fetched"}

    values = normalize(case.text, case.language)
    if not values:
        return {"state": NO_LITERAL,
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
        return {"state": NO_FIELD_MAPPING,
                "reason": "values normalised and bound, and no declared "
                          "mapping consumes this relation for this field",
                **_ticket(case, values, bindings, candidates)}

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
        return {"state": NO_FIELD_MAPPING,
                "reason": "this case asserts a role pair rather than a single "
                          "field; no mapping produces role pairs yet",
                "proposed": sorted({c.field for c in candidates}),
                **_ticket(case, values, bindings, candidates)}

    match = next((c for c in candidates if c.field == wanted), None)
    if match is None:
        return {"state": NO_FIELD_MAPPING,
                "reason": f"candidates were proposed for "
                          f"{sorted({c.field for c in candidates})} and none "
                          f"for {wanted!r}, which is what this case asserts",
                "proposed": sorted({c.field for c in candidates}),
                **_ticket(case, values, bindings, candidates)}
    # The candidate's own span, never the whole sentence. Passing the utterance
    # made fusion's ambiguity check fire on any sentence containing the word
    # "rebalance", whatever field was being decided — "rebalance on the last
    # session of each quarter" has a determinate day rule and an ambiguous verb,
    # and they are different dimensions.
    decision = fuse(match.field,
                    model=Proposal(match.field, match.value,
                                   "deterministic-stand-in@1",
                                   match.source_span))

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


def _ticket(case, values, bindings, candidates) -> dict:
    """What already exists and what is missing, for one unmapped case.

    A `NO_FIELD_MAPPING` row without this is a label. With it the row is close
    to an implementation ticket, and — more usefully — the next person does not
    re-debug normalisation and binding to find out they both worked.
    """
    return {"observed": [f"{v.kind}={v.canonical}" for v in values],
            "bound_to": sorted({f"{b.relation}->{b.target_span}"
                                for b in bindings if b.established}),
            "expected": {k: v for k, v in case.asserts.items()},
            "missing": "semantic_derivation"}


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
         "owner": {state: OWNER[state] for state in STATES},
         "ambiguity_note": (
             "AMBIGUOUS_BY_LANGUAGE has no live instance in this corpus, and "
             "that is correct rather than a gap in fusion. The rebalance / "
             "reallocate ambiguity is about *what the action does*; it does not "
             "touch how often it happens or which session of the period it "
             "happens on, which are the fields these cases assert. The corpus "
             "has no case asserting the field the ambiguity actually affects. "
             "It is exercised synthetically in tests/test_fusion.py, and a "
             "real case would be a corpus addition, not a code change."),
         "note": ("NO_LITERAL and NO_FIELD_MAPPING are not defects of the "
                  "deterministic layers, and they want opposite work: one "
                  "needs the semantic reader, the other needs field derivation "
                  "from structure already in hand. They were one state until "
                  "the counts made the difference matter."),
         "rows": rows}, indent=2, ensure_ascii=False) + "\n")

    print(f"{len(rows)} pending cases -> {OUT}")
    for state in STATES:
        if counts.get(state):
            suffix = ""
            if state == MAPPED_AND_AGREED:
                suffix = f"  ({len(agreed) - len(wrong)} with the expected value)"
            print(f"  {state:22} {counts[state]:3}{suffix}"
                  f"{'' if suffix else '   ' + OWNER[state]}")
    if wrong:
        print(f"  !! agreed with a value the case does not expect: {wrong}")
    print()
    for prop, states in sorted(by_property.items()):
        print(f"  {prop:32} {dict(states)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
