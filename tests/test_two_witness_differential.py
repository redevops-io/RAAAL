"""Case-level differential: both witness profiles, both implementations.

Aggregate counts hide the cases that matter. This produces a verdict per case
and writes the artifact out, so the evidence exists *before* the serving path
changes rather than being reconstructed afterwards.

    EQUIVALENT           same claims, outcomes, open fields, provenance
    REPRESENTATION_ONLY  differs in how it is written, not in what it means
    NEW_DISAGREEMENT     the runtime opens a question the internal path settled
    LOST_DISAGREEMENT    the runtime settles what the internal path questioned
    OUTCOME_CHANGE       a settled value differs, by the dimension's own rule

`NEW_DISAGREEMENT` and `OUTCOME_CHANGE` must be zero: the first puts a question
in front of somebody who was not being asked, the second changes what their
plan does.

`LOST_DISAGREEMENT` is reported separately and deliberately *not* treated as an
improvement. A disagreement disappearing is what the canonical-claim
aggregation was built to do — the deterministic reader emits one observation
per share and those are now one claim — but "the readers reconcile now" and
"the readers were made to agree" look identical in a count. Each one is listed
with both claims so it can be read rather than assumed.
"""
from __future__ import annotations

import json
import os
import pathlib
from collections import Counter

import pytest

#: The current run. The pre-cutover copy sits beside it under its own name and
#: is never overwritten: "the runtime is substitutable for what serves" and
#: "what serves is the runtime" are different claims, and one artifact cannot
#: carry both.
ARTIFACT = (pathlib.Path(__file__).resolve().parent.parent
            / "corpus" / "parser" / "two_witness_differential.json")

CORPUS = (pathlib.Path(__file__).resolve().parent.parent
          / "corpus" / "parser" / "strategy_closure_hosted.json")

DECLARED = {"QUANTIFY_PILOT_READER": "recorded",
            "QUANTIFY_PARSER_MODE": "RUNTIME",
            "QUANTIFY_PARSER_MODEL": "claude-sonnet-5",
            "ANTHROPIC_API_KEY": "unused",
            "PILOT_DATA_POLICY": "SYNTHETIC_ONLY",
            "QUANTIFY_SYNTAX_WITNESS": "yes"}


def _texts():
    if not CORPUS.exists():
        return ()
    return tuple(c["text"] for c in json.loads(CORPUS.read_text()).get("cases", [])
                 if c.get("text"))


@pytest.fixture(scope="module")
def declared():
    from src.deploy import context as deploy_context

    settings = deploy_context.resolve({**os.environ, **DECLARED})
    original = deploy_context.current
    deploy_context.current = lambda: settings
    yield settings
    deploy_context.current = original


def _profile(decisions):
    """A decision list as the facts the gate compares."""
    from src.discovery.witnesses import BOTH, provenance_of

    claims, outcomes, open_fields, provenance = {}, {}, [], {}
    for one in decisions:
        name = one.dimension
        outcomes[name] = one.outcome.name
        provenance[name] = provenance_of(one, BOTH)
        if one.proceeds:
            claims[name] = str(one.value)
        else:
            open_fields.append(name)
    return {"claims": claims, "outcomes": outcomes,
            "open": sorted(open_fields), "provenance": provenance}


def _both_paths(text):
    """One utterance, two implementations, same frozen witnesses."""
    from src.discovery.adapter import decisions_via_runtime, deterministic_witness
    from src.discovery.hosted_recording import RecordedHostedReader
    from src.discovery.pipeline import read as internal_two_witness
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.syntax_stanza import RecordedReader

    model = RecordedHostedReader().read(text, QUANTIFY_SCHEMA)
    parse = RecordedReader().parse(text)

    internal = list(internal_two_witness(text, parse, model,
                                         QUANTIFY_SCHEMA).decisions)
    evidence, derived = deterministic_witness(text, parse)
    runtime = decisions_via_runtime(model, syntax_evidence=evidence,
                                    derived=derived)
    return _profile(internal), _profile(runtime)


def _verdict(old, new):
    """One verdict per case, by meaning rather than by layout."""
    from src.discovery.adapter import same_value_for

    changed = [name for name in set(old["claims"]) & set(new["claims"])
               if not same_value_for(name, old["claims"][name],
                                     new["claims"][name])]
    if changed:
        return "OUTCOME_CHANGE", changed

    newly_open = sorted(set(new["open"]) - set(old["open"]))
    if newly_open:
        return "NEW_DISAGREEMENT", newly_open

    newly_settled = sorted(set(old["open"]) - set(new["open"]))
    if newly_settled:
        return "LOST_DISAGREEMENT", newly_settled

    # A dimension one path knows about and the other does not, with nothing
    # settled differently, is coverage rather than meaning.
    if set(old["claims"]) != set(new["claims"]):
        return "REPRESENTATION_ONLY", sorted(
            set(old["claims"]) ^ set(new["claims"]))
    if old["provenance"] != new["provenance"]:
        return "REPRESENTATION_ONLY", ["provenance"]
    if old["outcomes"] != new["outcomes"]:
        return "REPRESENTATION_ONLY", ["outcomes"]
    return "EQUIVALENT", []


def test_the_corpus_is_reachable_under_both_witnesses(declared):
    assert len(_texts()) >= 20, "too few cases for this to mean anything"


def test_two_witness_differential(declared, capsys):
    """The gate, and the artifact it leaves behind."""
    rows, counts = [], Counter()
    for text in _texts():
        try:
            old, new = _both_paths(text)
        except Exception as failure:                           # noqa: BLE001
            counts["COULD_NOT_RUN"] += 1
            rows.append({"case": text, "verdict": "COULD_NOT_RUN",
                         "detail": f"{type(failure).__name__}: {failure}"})
            continue
        verdict, detail = _verdict(old, new)
        counts[verdict] += 1
        rows.append({"case": text, "verdict": verdict, "detail": detail,
                     "internal": old, "runtime": new})

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(
        {"profile": "two-witness (recorded model + recorded parse)",
         "scope": ("Establishes substitutability of discovery-runtime for the "
                   "internal implementation under frozen recorded witnesses. "
                   "Does not establish behaviour of the deployed model."),
         "counts": dict(counts), "cases": rows}, indent=2, sort_keys=True))

    print("\n  two-witness differential")
    for verdict, n in sorted(counts.items()):
        print(f"    {verdict:<22} {n}")
    for row in rows:
        if row["verdict"] in ("NEW_DISAGREEMENT", "OUTCOME_CHANGE",
                              "COULD_NOT_RUN"):
            print(f"    {row['verdict']}: {row['case'][:58]!r} {row['detail']}")
    for row in rows:
        if row["verdict"] == "LOST_DISAGREEMENT":
            print(f"    LOST: {row['case'][:50]!r} {row['detail']}")

    assert counts["COULD_NOT_RUN"] == 0, "some cases did not run"
    assert counts["NEW_DISAGREEMENT"] == 0, (
        "the runtime opens questions the internal path settled")
    assert counts["OUTCOME_CHANGE"] == 0, (
        "a settled value differs by the dimension's own comparison rule")
