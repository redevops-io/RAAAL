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

#: Both witnesses spoke about the asserted field.
AGREE = "AGREE"
DISAGREE = "DISAGREE"
AMBIGUOUS_BY_LANGUAGE = "AMBIGUOUS_BY_LANGUAGE"

#: Only the model spoke. Kept apart from `AGREE` because a field settled by one
#: witness is a different kind of evidence from one two witnesses reached
#: independently — and collapsing them would let the corpus report agreement it
#: never observed.
MODEL_ONLY_ACCEPTED = "MODEL_ONLY_ACCEPTED"
MODEL_ONLY_UNRESOLVED = "MODEL_ONLY_UNRESOLVED"

#: Syntax spoke and the model did not. Named separately because it was being
#: reported as MODEL_ONLY — the count was right and the label named the wrong
#: witness, which is worse than either being wrong on its own.
SYNTAX_ONLY_UNRESOLVED = "SYNTAX_ONLY_UNRESOLVED"

INSUFFICIENT_RELATION = "INSUFFICIENT_RELATION"
INTERMEDIATE_SEMANTIC = "INTERMEDIATE_SEMANTIC"
NO_LITERAL = "NO_LITERAL"
NO_FIELD_MAPPING = "NO_FIELD_MAPPING"
NO_PARSE = "NO_PARSE_RECORDED"
NO_MODEL_RECORDING = "NO_MODEL_RECORDING"
SCHEMA_GAP = "SCHEMA_GAP"

#: An intermediate the corpus asserts and nothing computes.
#:
#: Split out because the first version classified a case as
#: INTERMEDIATE_SEMANTIC on the strength of its *asserted field* alone, without
#: checking that anything produced it — and then excluded it from the pending
#: count. Four of six were in that state: classified out of the queue and
#: verified by nothing, which is the overclaiming shape this report exists to
#: catch, reproduced in the report.
INTERMEDIATE_NOT_PRODUCED = "INTERMEDIATE_NOT_PRODUCED"

STATES = (AGREE, MODEL_ONLY_ACCEPTED, DISAGREE, MODEL_ONLY_UNRESOLVED,
          SYNTAX_ONLY_UNRESOLVED,
          AMBIGUOUS_BY_LANGUAGE, INSUFFICIENT_RELATION, INTERMEDIATE_SEMANTIC,
          NO_FIELD_MAPPING, NO_LITERAL, NO_PARSE, NO_MODEL_RECORDING,
          SCHEMA_GAP, INTERMEDIATE_NOT_PRODUCED)

#: States whose cases could eventually produce contract semantics.
#:
#: `INTERMEDIATE_SEMANTIC` is deliberately absent. Those cases assert semantics
#: this pipeline computes and the contract does not carry, and counting them as
#: pending measured the wrong boundary — they are not waiting on anything.
#: Excluding them is not making the number look better; the report records that
#: they used to be counted, so the change is visible rather than quiet.
COUNTS_AS_PENDING = frozenset(STATES) - {AGREE, MODEL_ONLY_ACCEPTED,
                                         INTERMEDIATE_SEMANTIC, SCHEMA_GAP}

#: Which layer owns each state's remaining work. Printed with the counts,
#: because "36 unsupported" is a pile and "36 waiting on the semantic reader"
#: is a queue with an owner.
OWNER = {AGREE: "—",
         MODEL_ONLY_ACCEPTED: "— (one witness; a deterministic producer would "
                              "make it two)",
         DISAGREE: "adjudication",
         MODEL_ONLY_UNRESOLVED: "adjudication",
         SYNTAX_ONLY_UNRESOLVED: "— (the asymmetry witness: syntax alone never "
                                 "carries a field)",
         AMBIGUOUS_BY_LANGUAGE: "the user, via clarification",
         INSUFFICIENT_RELATION: "binder or corpus",
         INTERMEDIATE_SEMANTIC: "tests/test_semantics.py — verified at the "
                                "mapper boundary, not through fusion",
         NO_FIELD_MAPPING: "semantics.py — field derivation",
         NO_LITERAL: "neither witness reads this field",
         NO_PARSE: "deferred_multilingual.json — out of declared scope",
         NO_MODEL_RECORDING: "corpus/parser/record_hosted.py",
         SCHEMA_GAP: "the schema — a sayable reading it cannot hold",
         INTERMEDIATE_NOT_PRODUCED: "semantics.py — nothing computes it"}


def classify(case, recorded: RecordedReader,
             hosted=None) -> dict:
    """One case, one state, and the reason in the case's own terms.

    Runs the *whole* pipeline now — both witnesses — rather than the
    deterministic path alone. Before the hosted reader was wired, a case whose
    field nothing normalises could only be `NO_LITERAL`; with a second witness
    that label was measuring the absence of one producer and calling it the
    absence of all of them.
    """
    from src.discovery.pipeline import read
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.semantics import INTERMEDIATE_FIELDS

    wanted = case.asserts.get("field")

    # A reading the contract has no value for. Recorded rather than renamed to
    # the nearest allowed one, which would make the corpus agree with a schema
    # that cannot express the sentence.
    if "schema_gap" in case.asserts:
        return {"state": SCHEMA_GAP, "field": wanted,
                "reason": f"the sentence reads as {case.asserts['schema_gap']!r} "
                          f"and {wanted!r} has no such value"}

    # Outside the contract boundary, and not waiting on anything. Asserted
    # against mapper output by `tests/test_semantics.py` instead.
    if wanted in INTERMEDIATE_FIELDS:
        # And check that something actually computes it. Classifying a case as
        # intermediate on the strength of its asserted field alone, then
        # excluding it from the pending count, is how a case ends up out of the
        # queue and verified by nothing.
        from src.discovery.binding import bind
        from src.discovery.semantics import propose
        from src.discovery.syntax import normalize

        if not recorded.has(case.text, case.language):
            return {"state": NO_PARSE,
                    "reason": f"no {case.language} model has been fetched"}
        values = normalize(case.text, case.language)
        produced = {c.field: c.value for c in
                    propose(bind(recorded.parse(case.text, case.language), values),
                            values)}
        if wanted not in produced:
            return {"state": INTERMEDIATE_NOT_PRODUCED, "field": wanted,
                    "reason": f"the case asserts {wanted!r} and the mapper "
                              f"produced {sorted(produced)}; nothing computes "
                              "it, so being outside the contract is not the "
                              "reason it is unanswered"}
        return {"state": INTERMEDIATE_SEMANTIC, "field": wanted,
                "value": str(produced[wanted]),
                "expected": case.asserts.get("value"),
                "matches_expected": str(produced[wanted]) == str(
                    case.asserts.get("value")),
                "reason": f"{wanted!r} is computed by this pipeline and is not "
                          "a contract dimension; verified at the mapper "
                          "boundary by tests/test_semantics.py",
                "previously_counted_as_pending": True}

    if not recorded.has(case.text, case.language):
        return {"state": NO_PARSE,
                "reason": f"no {case.language} model has been fetched"}
    if hosted is None or not hosted.has(case.text):
        return {"state": NO_MODEL_RECORDING,
                "reason": "no recorded hosted reading; run record_hosted.py"}

    result = read(case.text, recorded.parse(case.text, case.language),
                  hosted.read(case.text, QUANTIFY_SCHEMA), QUANTIFY_SCHEMA,
                  language=case.language)

    if wanted is None:
        return {"state": NO_FIELD_MAPPING,
                "reason": "this case asserts a role pair rather than a single "
                          "field; no mapping produces role pairs yet",
                "proposed": sorted(result.by_field)}

    decision = result.by_field.get(wanted)
    if decision is None:
        # Neither witness produced the asserted field. Which of the two labels
        # applies depends on whether anything at all was read here.
        if result.candidates or result.decisions:
            return {"state": NO_FIELD_MAPPING,
                    "reason": f"readings were produced for "
                              f"{sorted(result.by_field)} and none for "
                              f"{wanted!r}, which is what this case asserts",
                    "proposed": sorted(result.by_field),
                    "intermediate": sorted({c.field for c in result.intermediate})}
        return {"state": NO_LITERAL,
                "reason": "neither witness produced a reading for this "
                          "sentence at all"}

    produced = _plain(decision.value) if decision.proceeds else None
    expected = case.asserts.get("value")
    witnesses = ["model"] if decision.model is not None else []
    witnesses += ["syntax"] if decision.syntax else []

    # Compared by the dimension's own rule, not by string equality. The model
    # renders an amount as `£1k` and the corpus writes `1000`; calling that a
    # mismatch here would reintroduce, in the report, exactly the formatting-as-
    # conflict defect that `compare_as` was added to fusion to remove.
    from src.discovery.fusion import REQUIREMENTS, Requirement, same_value

    rule = REQUIREMENTS.get(wanted, Requirement()).compare_as
    common = {"field": wanted, "witnesses": witnesses,
              "value": produced, "expected": expected, "compared_as": rule,
              "matches_expected": (produced is not None and expected is not None
                                   and same_value(produced, expected, rule))}

    if decision.outcome is Fusion.AMBIGUOUS_BY_LANGUAGE:
        return {"state": AMBIGUOUS_BY_LANGUAGE, "reason": decision.detail,
                **common}
    if decision.outcome is Fusion.INSUFFICIENT_RELATION:
        return {"state": INSUFFICIENT_RELATION, "reason": decision.detail,
                **common}
    if decision.proceeds:
        state = AGREE if len(witnesses) > 1 else MODEL_ONLY_ACCEPTED
        return {"state": state, "reason": decision.detail, **common}
    if len(witnesses) > 1:
        state = DISAGREE
    elif witnesses == ["model"]:
        state = MODEL_ONLY_UNRESOLVED
    else:
        state = SYNTAX_ONLY_UNRESOLVED
    return {"state": state, "reason": decision.detail, **common}


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
    from src.discovery.hosted_recording import RecordedHostedReader

    recorded, hosted = RecordedReader(), RecordedHostedReader()
    pending = [c for c in load()
               if c.tier == "semantics"
               or (c.tier == "dependency" and not recorded.has(c.text, c.language))]

    rows = []
    for case in pending:
        outcome = classify(case, recorded, hosted)
        rows.append({"id": case.id, "property": case.property,
                     "text": case.text, "language": case.language, **outcome})

    counts = Counter(row["state"] for row in rows)
    agreed = [r for r in rows if r["state"] in (AGREE, MODEL_ONLY_ACCEPTED)]
    wrong = [r["id"] for r in agreed if not r.get("matches_expected")]
    by_property: dict = {}
    for row in rows:
        by_property.setdefault(row["property"], Counter())[row["state"]] += 1

    still_pending = [r for r in rows if r["state"] in COUNTS_AS_PENDING]
    reclassified = [r["id"] for r in rows
                    if r.get("previously_counted_as_pending")]

    OUT.write_text(json.dumps(
        {"schema": "quantify-parser-closure@2",
         "cases": len(rows),
         "pending": len(still_pending),
         "excluded_from_pending": {
             "state": INTERMEDIATE_SEMANTIC, "count": len(reclassified),
             "ids": reclassified,
             "classification": "intentionally outside the contract boundary",
             "previously_counted_by_awaiting_a_parser": True,
             "why": ("They assert semantics this pipeline computes and the "
                     "contract does not carry. Counting them as pending "
                     "measured the wrong boundary; the history is recorded "
                     "here so the change is visible rather than quiet.")},
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

    print(f"{len(rows)} cases, {len(still_pending)} pending -> {OUT}")
    for state in STATES:
        if counts.get(state):
            suffix = ""
            if state in (AGREE, MODEL_ONLY_ACCEPTED):
                of_state = [r for r in rows if r["state"] == state]
                right = sum(1 for r in of_state if r.get("matches_expected"))
                suffix = f"  ({right} of {len(of_state)} with the expected value)"
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
