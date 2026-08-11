"""Every compiler-only row, with an explicit disposition. No residuals.

    python corpus/ledger.py            # build and check
    python corpus/ledger.py --write    # write corpus/ledger.json

A compiler-only row is one the legacy reader answered and the hosted reader did
not. It is the shape where a model recall failure silently erases a declared
dimension, so it is the shape that most needs a name attached to each instance
rather than a class attached to the pile.

Four dispositions, and the fourth is the one that earns its keep:

    COMPILER_DEFECT      the compiler's value is wrong, so nothing was lost by
                         the model not matching it
    MODEL_MISS           the sentence says it, the compiler read it, the model
                         should have and did not
    REPRESENTATION_MOVE  the model carried the same information somewhere else
                         — a relation rather than a scalar. Not a miss; a
                         different and usually better representation
    READER_ABSENT        the hosted reader produced no usable output for this
                         prompt at all, so this row is not evidence about its
                         recall

`READER_ABSENT` is "absence is not ignorance" arriving in the evaluator. Of 64
compiler-only rows in the first schema@2 run, 46 came from prompts where the
model never contributed — 16 truncated by this harness's own token ceiling and
two from a dry run that had overwritten a real measurement. Counting those as
recall failures would have been a finding about the harness reported as a
finding about the model.

**This file fails rather than defaulting.** A row it cannot classify is listed
and the exit code is 1. There is no `UNADJUDICATED` bucket, because a bucket
like that fills up and then gets read as "nothing to see".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

COMPILER = "quantify-compiler@2"
HOSTED = "claude-sonnet-5@1"

COMPILER_DEFECT = "COMPILER_DEFECT"
MODEL_MISS = "MODEL_MISS"
REPRESENTATION_MOVE = "REPRESENTATION_MOVE"
READER_ABSENT = "READER_ABSENT"
SCHEMA_GAP = "SCHEMA_GAP"
"""A fifth, and the deviation is deliberate. The other four answer "which
reader was wrong", which every compiler-only row has an answer to. A contested
row need not: "convert my traditional IRA to a Roth" had both readers stating a
true fact and the schema forcing them to disagree. Without this class that row
would have to be blamed on a reader, which is how a representational failure
gets recorded as a reading failure and then fixed in the wrong place."""

DISPOSITIONS = (COMPILER_DEFECT, MODEL_MISS, REPRESENTATION_MOVE,
                READER_ABSENT, SCHEMA_GAP)

#: Dimensions where the compiler's reading is known wrong from adjudication, so
#: a row it alone produced is not a loss. Each entry cites the adjudicated
#: finding it rests on rather than being asserted here.
KNOWN_COMPILER_DEFECTS = {
    "assets": "extracts capitalised fragments as instruments — 'ETF' from "
              "'multi-asset ETF set', 'DD' from 'drawdown-aware'",
    "observed_assets": "names the wrapper rather than the watched series — "
                       "'ETF' where the sentence watches the S&P 500",
    "evaluation_period": "reads cadence-plus-window as a rolling window",
}

#: Rows needing a judgement no rule can make. Keyed (ref, dimension).
#: Empty until the clean run produces rows that fall through, and every entry
#: carries its reason — an unexplained disposition is a vote, not a finding.
BY_HAND: dict = {
    # Six sentences where a rule cannot decide, judged from the prompt text.
    # Five are one class: the compiler reads a frequency out of a clause that
    # is not about contributions — the same shape as the rebalancing/cadence
    # collision and the cadence/window collision before it. That pattern is now
    # three-for-three, and it is the strongest single argument that the legacy
    # reader's failure mode is structural rather than a list of missing rules.
    ("WM-0081", "cadence"): (
        COMPILER_DEFECT,
        "'with annual CPI adjustment' — annual describes the inflation "
        "adjustment, not a contribution cadence"),
    ("WM-0085", "cadence"): (
        COMPILER_DEFECT,
        "'with refill annually' — the refill frequency of a withdrawal "
        "bucket, not a contribution cadence"),
    ("WM-0063", "cadence"): (
        COMPILER_DEFECT,
        "'within daily harvesting' — daily describes the harvesting, not a "
        "contribution cadence"),
    ("WM-0081", "amount"): (
        COMPILER_DEFECT,
        "read 1,000,000 from a sentence containing no such figure; the only "
        "number in it is the 4% withdrawal rate"),
    ("WM-0048", "trigger_semantics"): (
        COMPILER_DEFECT,
        "'when trend + vol filter' does not distinguish a crossing from a "
        "persistent state, and the compiler chose one. The model left it out, "
        "which is what the instructions ask for when a sentence is silent"),

    ("WM-0041", "cadence"): (
        COMPILER_DEFECT,
        "'when SPY<200SMA' is a condition and the sentence names no calendar "
        "at all; the compiler produced 'daily' from nothing in it"),
    ("WM-0076", "cadence"): (
        COMPILER_DEFECT,
        "'convert after -20% each year' — the recurrence belongs to a "
        "conversion, and `cadence` in this schema means how often money is "
        "*contributed*"),

    # Corrected from MODEL_MISS after WM-0076 forced the question. `cadence`
    # means contribution frequency, so a recurrence belonging to gifting is
    # the same defect as one belonging to a refill or a CPI adjustment. The
    # earlier call was inconsistent with the five rows beside it, and a
    # disposition that depends on which row you looked at first is not a
    # judgement.
    ("WM-0123", "cadence"): (
        COMPILER_DEFECT,
        "'gifts upfront to charity each year' — the recurrence belongs to "
        "giving, not to contributing"),

    ("CTRL-1", "trigger_semantics"): (
        MODEL_MISS,
        "'whenever it crosses below its 200-day moving average' says crossing "
        "plainly, and the model did not report it. The same sentence in "
        "another row it did — a non-determinism the plan already flags, "
        "recorded here as the miss it is"),
}


def load(corpus: str) -> dict:
    path = HERE / "shadow" / f"shadow-{corpus}.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing — run the shadow first")
    return json.loads(path.read_text())


def check_validity(payload: dict, corpus: str) -> list:
    """A ledger built on an invalid run is a ledger about the harness."""
    provenance = payload.get("provenance") or {}
    problems = []
    if provenance.get("mode") != "full":
        problems.append(f"{corpus}: mode={provenance.get('mode')!r}, not a full run")
    if provenance.get("truncated"):
        problems.append(f"{corpus}: {provenance['truncated']} truncated replies")
    hosted = (provenance.get("readers") or {}).get(HOSTED, {})
    if not hosted.get("enabled"):
        problems.append(f"{corpus}: the hosted reader was not enabled")
    return problems


#: Dimensions where the schema itself cannot state the sentence, so neither
#: reader can be right. Adjudicated, not inferred.
KNOWN_SCHEMA_GAPS = {
    "account_type": "a conversion names two accounts in named roles and the "
                    "scalar holds one — represented by rel:account_transition "
                    "in schema@2, and rows predating that carry this",
}


def classify(record: dict, dimension: str, state: str) -> tuple:
    """(disposition, reason, evidence_source) for one row needing adjudication.

    `evidence_source` records *how the disposition was reached*, so a reader of
    the ledger can tell a rule from a judgement without asking anyone.
    """
    if not record.get("usable"):
        failed = record.get("failed_readers") or {}
        why = next(iter(failed.values()), "the reader contributed nothing")
        truncated = "unparseable" in str(why)
        return (READER_ABSENT,
                ("hosted reply truncated mid-output by the harness's own token "
                 "ceiling" if truncated else
                 "hosted reader produced no output at all"),
                "rule:truncated" if truncated else "rule:no-output")

    # Did the model carry it in a relation instead?
    for one in record.get("fields", []):
        if not one["dimension"].startswith("rel:"):
            continue
        if HOSTED not in one.get("values", {}):
            continue
        if dimension in ("assets", "account_type", "stated_weights"):
            return (REPRESENTATION_MOVE,
                    f"the model carried it in {one['dimension']} instead of "
                    "the scalar",
                    f"rule:relation-move:{one['dimension']}")

    if dimension in KNOWN_SCHEMA_GAPS:
        return SCHEMA_GAP, KNOWN_SCHEMA_GAPS[dimension], "adjudicated:schema-gap"

    if dimension in KNOWN_COMPILER_DEFECTS:
        return (COMPILER_DEFECT, KNOWN_COMPILER_DEFECTS[dimension],
                "adjudicated:compiler-defect")

    return "", "", ""


def needs_adjudication(one: dict) -> bool:
    """Rows where something is wrong, or where the schema could not say it.

    CONTESTED: both read it, differently — one of them is wrong, or neither is.
    compiler-only: the legacy reader answered and the hosted one did not, which
    is where a recall failure silently erases a declared dimension.

    Model-only rows are *not* here. There is nothing to adjudicate when the
    compiler has no opinion — they are the coverage Discovery adds, counted
    separately as `exceeds`. A model-only reading can still be wrong, and the
    instrument for that is the canonical expectations, not this ledger.
    """
    if one["state"] == "CONTESTED":
        return True
    return one["state"] == "ONE_SIDED" and COMPILER in one.get("values", {})


def build() -> tuple:
    rows, problems, exceeds = [], [], []
    for corpus in ("strategies", "catalogue"):
        payload = load(corpus)
        problems.extend(check_validity(payload, corpus))
        fingerprint = payload.get("schema", "")
        run_id = (payload.get("provenance") or {}).get("prompts_digest", "")

        for record in payload["records"]:
            for one in record["fields"]:
                if one["state"] == "ONE_SIDED" and HOSTED in one.get("values", {}):
                    exceeds.append({"corpus": corpus, "ref": record["ref"],
                                    "dimension": one["dimension"]})
                if not needs_adjudication(one):
                    continue

                dimension = one["dimension"]
                disposition, reason, source = classify(
                    record, dimension, one["state"])
                key = (record["ref"], dimension)
                if not disposition and key in BY_HAND:
                    disposition, reason = BY_HAND[key]
                    source = "by-hand"

                rows.append({
                    "run_id": run_id,
                    "schema_fingerprint": fingerprint,
                    "corpus": corpus,
                    "prompt_id": record["ref"],
                    "dimension": dimension,
                    "comparison_state": one["state"],
                    "compiler_value": one.get("values", {}).get(COMPILER),
                    "model_value": one.get("values", {}).get(HOSTED),
                    "disposition": disposition or "",
                    "reason": reason,
                    "evidence_source": source,
                })
    return rows, problems, exceeds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    rows, problems, exceeds = build()
    if problems:
        print("INVALID RUN — a ledger built on this would describe the harness:")
        for one in problems:
            print(f"  {one}")
        return 2

    counts: dict = {}
    for row in rows:
        counts[row["disposition"] or "UNCLASSIFIED"] = \
            counts.get(row["disposition"] or "UNCLASSIFIED", 0) + 1

    contested = sum(1 for r in rows if r["comparison_state"] == "CONTESTED")
    one_sided = len(rows) - contested
    print(f"{len(rows)} adjudicated rows "
          f"({contested} contested, {one_sided} compiler-only)\n")

    # Every disposition, iterated from DISPOSITIONS rather than a hand-written
    # list. The first version omitted SCHEMA_GAP and printed 73 of 80 rows
    # under a heading that did not say so — a summary that silently drops a
    # category, which is the defect this whole ledger exists to prevent.
    for name in (*DISPOSITIONS, "UNCLASSIFIED"):
        if counts.get(name):
            print(f"  {counts[name]:4d}  {name}")
    assert sum(counts.values()) == len(rows), "the summary lost rows"

    unclassified = [r for r in rows if not r["disposition"]]
    if unclassified:
        print(f"\n{len(unclassified)} rows need a judgement no rule can make. "
              "Add each to BY_HAND with its reason:\n")
        for row in unclassified[:40]:
            print(f"  ({row['prompt_id']!r}, {row['dimension']!r}): "
                  f"compiler said {row['compiler_value']!r}")
        return 1

    print(f"\n  {len(exceeds)} model-only readings — coverage Discovery adds "
          "where the\n  compiler has no opinion. Not adjudicated here: there "
          "is nothing to\n  adjudicate against. Their correctness is the "
          "canonical expectations'\n  job.")

    if args.write:
        (HERE / "ledger.json").write_text(json.dumps(
            {"dispositions": list(DISPOSITIONS), "rows": rows,
             "counts": counts, "exceeds": len(exceeds)}, indent=1))
        print(f"\nwrote {HERE}/ledger.json")
    print("\nno residual UNADJUDICATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
