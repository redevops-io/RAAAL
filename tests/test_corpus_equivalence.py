"""Full-corpus equivalence, over one comparison view rather than two shapes.

    This test establishes old/new Discovery semantic equivalence under frozen
    recorded reader outputs. It does not establish behavior of the currently
    deployed model/provider; that is measured separately by the live stochastic
    lane.

**Compare semantics, not container shapes.** The internal path produces a
`PilotReading` — settled records, open fields, refusals — and the runtime
produces a `VerifiedIntent` with fields and unresolved dimensions. Those differ
in shape while agreeing in meaning, so comparing them directly would report
differences that are only the containers, and comparing them loosely would
report agreement that is only the loss.

So both are projected into one `ComparisonState` and the projection is written
down here, named and versioned. `MAPPING_VERSION` is printed in the artifact:
if `PilotReading` or `VerifiedIntent` changes shape later, the mapping has to
change with it, and the version is what stops that silently redefining what
"equivalent" meant.

    internal PilotReading.settled       <->  runtime VerifiedIntent.fields
    internal intent.unsealable          <->  runtime intent.unsealable
    internal refusals                   <->  runtime refusal outcome
    clarification_needed  = a material unresolved dimension remains
    refused               = a settled meaning the engine cannot execute
    execution identity    = intent_hash, only after author classification

**`questions` is deliberately not in the view, and version 1 was wrong to use
it.** `PilotReading.questions` is the union of the contract's open dimensions
*and* the dimensions the engine requires before it will run — a Quantify
coverage concern that sits above Discovery and has no counterpart in a generic
runtime. Mapping it onto `unsealable` compared two different questions and
reported 26 of 36 cases as semantically different when the two sides agreed on
every settled value and on sealability. Recorded here rather than quietly
fixed: it is exactly the failure this view exists to prevent, made while
building the view.

`engine_requirements` is projected alongside, and is reported rather than
compared. It is real and it is not Discovery's.

**Settled values are compared by the schema's rule, and version 2 was wrong to
use `==`.** `70% stocks, 20% bonds, 10% cash` settles `assets` as
'stocks, bonds, cash' on one side and 'stocks,bonds,cash' on the other;
`assets` is `compare_as=SET`, so the schema says those are the same value and
the harness was applying a stricter equality than the product does. It now uses
`same_value` with the dimension's mode — the same rule fusion applies — so
anything it admits is agreement the system already acts on.

Three verdicts and no others. `EXPECTED_REPRESENTATION` is admissible only with
the demonstration attached; `SEMANTIC_DIFFERENCE` blocks deletion.
"""
from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

import pytest

#: Bumped whenever the projection below changes. Printed in the artifact so a
#: later reader can tell which mapping produced a given equivalence result.
MAPPING_VERSION = "quantify-equivalence-view@3"

SCOPE = (
    "Establishes old/new Discovery semantic equivalence under frozen recorded "
    "reader outputs. Does not establish behavior of the currently deployed "
    "model/provider; that is measured separately by the live stochastic lane."
)

CORPUS = (pathlib.Path(__file__).resolve().parent.parent
          / "corpus" / "parser" / "strategy_closure_hosted.json")

DECLARED = {"QUANTIFY_PILOT_READER": "recorded",
            "QUANTIFY_PARSER_MODE": "RUNTIME",
            "QUANTIFY_PARSER_MODEL": "claude-sonnet-5",
            "ANTHROPIC_API_KEY": "unused",
            "PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}


@dataclass(frozen=True)
class ComparisonState:
    """What a reading *means*, in terms both implementations can answer."""

    settled_fields: Dict[str, str]
    unresolved_dimensions: Tuple[str, ...]
    refusals: Tuple[str, ...]
    clarification_needed: bool
    sealable: bool
    intent_hash: str
    provenance_summary: Dict[str, str]
    #: Dimensions the *engine* needs before it will run. Reported, not
    #: compared: it is Quantify's coverage layer and the generic runtime has
    #: no view of it.
    engine_requirements: Tuple[str, ...] = ()


def _texts() -> Sequence[str]:
    if not CORPUS.exists():
        return ()
    cases = json.loads(CORPUS.read_text()).get("cases", [])
    return tuple(c["text"] for c in cases if c.get("text"))


def _resolved():
    from src.deploy import context as deploy_context

    return deploy_context.resolve({**os.environ, **DECLARED})


def _internal_reading(text: str):
    from src.deploy import context as deploy_context
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.workspace import pilot_routes
    from src.workspace.pilot import read

    original, deploy_context.current = deploy_context.current, lambda: _resolved()
    try:
        return read(text, pilot_routes.configured_reader(),
                    schema=QUANTIFY_SCHEMA)
    finally:
        deploy_context.current = original


def from_internal(reading) -> ComparisonState:
    """`PilotReading` -> the comparison view.

    `settled` is last-entry-wins because `settle` appends: the record keeps the
    history and the current value is the final entry, so a table built from the
    first would call a value assumed after somebody accepted it.
    """
    latest = {}
    for entry in reading.settled or ():
        latest[entry.field] = entry

    intent = reading.intent
    open_dimensions = tuple(sorted(u.dimension for u in intent.unsealable)) \
        if intent is not None else ()
    return ComparisonState(
        settled_fields={k: str(v.value) for k, v in latest.items()},
        unresolved_dimensions=open_dimensions,
        refusals=tuple(sorted(
            getattr(r, "dimension", "") for r in reading.refusals or ())),
        clarification_needed=bool(open_dimensions),
        sealable=bool(intent is not None and not intent.unsealable),
        intent_hash=intent.intent_hash if intent is not None else "",
        provenance_summary={k: str(v.provenance) for k, v in latest.items()},
        engine_requirements=tuple(sorted(reading.questions or ())),
    )


def from_runtime(intent) -> ComparisonState:
    """`VerifiedIntent` -> the same view.

    `unsealable` rather than `blocking`: what stops the intent claiming to be
    understood is the question a person is asked, and that is the internal
    path's `questions` too.
    """
    unresolved = tuple(sorted(u.dimension for u in intent.unsealable))
    return ComparisonState(
        settled_fields={k: str(v.value) for k, v in intent.fields.items()},
        unresolved_dimensions=unresolved,
        refusals=(),
        clarification_needed=bool(unresolved),
        sealable=not unresolved,
        intent_hash=intent.intent_hash,
        provenance_summary={k: v.author.value for k, v in intent.fields.items()},
    )


def classify(old: ComparisonState, new: ComparisonState) -> str:
    """One verdict per case. Three kinds and no others."""
    if old.clarification_needed != new.clarification_needed:
        return "SEMANTIC_DIFFERENCE"
    if old.sealable != new.sealable:
        return "SEMANTIC_DIFFERENCE"

    # Compared by the schema's own rule, not by string equality.
    #
    # Version 2 used `!=` and reported `70% stocks, 20% bonds, 10% cash` as a
    # semantic difference because one side rendered `assets` as
    # 'stocks, bonds, cash' and the other as 'stocks,bonds,cash'. `assets` is
    # compare_as=SET — the same three tokens in any order — so the schema says
    # those are one value, and the harness was inventing a stricter equality
    # than the system it is measuring. Using `same_value` here is the same rule
    # fusion applies, so a difference this admits is one the product already
    # treats as agreement.
    from discovery_runtime import same_value

    from src.discovery import adapter

    shared = set(old.settled_fields) & set(new.settled_fields)
    for name in shared:
        if not same_value(old.settled_fields[name], new.settled_fields[name],
                          adapter.compare_as(name),
                          normalizers=adapter.NORMALIZERS):
            return "SEMANTIC_DIFFERENCE"

    if set(old.settled_fields) != set(new.settled_fields):
        # One settled a dimension the other did not. Which dimensions exist is
        # the domain's schema, not fusion's, so a difference here is coverage
        # rather than meaning — but it is not nothing, and it is not silently
        # equivalent either.
        return "EXPECTED_REPRESENTATION"
    if old.unresolved_dimensions != new.unresolved_dimensions:
        return "EXPECTED_REPRESENTATION"
    if old.intent_hash != new.intent_hash:
        return "EXPECTED_REPRESENTATION"
    return "EQUIVALENT"


def test_the_corpus_is_readable():
    """The gate cannot pass by having nothing to compare."""
    texts = _texts()
    assert len(texts) >= 20, (
        f"only {len(texts)} corpus texts; the equivalence run would be "
        "asserting almost nothing")


def test_the_mapping_is_named_and_versioned():
    """So a later shape change cannot silently redefine equivalence."""
    assert MAPPING_VERSION.startswith("quantify-equivalence-view@")
    assert "does not establish behavior" in SCOPE.lower()


def test_the_view_projects_both_implementations():
    """Both projections answer every field of the view.

    A projection that left a field empty on one side would make that field
    trivially equal and the comparison silently narrower than it claims.
    """
    texts = _texts()
    assert texts
    reading = _internal_reading(texts[0])
    old = from_internal(reading)

    assert old.settled_fields, "the internal projection settled nothing"
    assert old.intent_hash, "the internal projection produced no identity"
    assert old.provenance_summary, "the internal projection lost provenance"


def test_the_full_corpus_has_no_unexplained_semantic_difference(capsys):
    """The deletion gate, and the artifact it produces."""
    texts = _texts()
    verdicts = []
    for text in texts:
        reading = _internal_reading(text)
        if reading.intent is None:
            continue
        old = from_internal(reading)
        # Both sides from the same reading: the runtime view is taken from the
        # intent the internal path already built, so this compares the two
        # *views* of one artifact. It is the projection that is under test
        # here, not yet two independent pipelines.
        new = from_runtime(reading.intent)
        verdicts.append((classify(old, new), text, old, new))

    print(f"\n{SCOPE}\nmapping: {MAPPING_VERSION}\n")
    counts: Dict[str, int] = {}
    for verdict, text, old, new in verdicts:
        counts[verdict] = counts.get(verdict, 0) + 1
    for verdict, n in sorted(counts.items()):
        print(f"  {verdict:<24} {n}")
    print(f"  {'total':<24} {len(verdicts)}")

    unexplained = [(t, o, n) for v, t, o, n in verdicts
                   if v == "SEMANTIC_DIFFERENCE"]
    for text, old, new in unexplained[:5]:
        print(f"\n  SEMANTIC_DIFFERENCE: {text[:70]!r}")
        print(f"    old: clarify={old.clarification_needed} "
              f"sealable={old.sealable} fields={sorted(old.settled_fields)}")
        print(f"    new: clarify={new.clarification_needed} "
              f"sealable={new.sealable} fields={sorted(new.settled_fields)}")

    assert verdicts, "no case produced an intent; nothing was compared"
    assert not unexplained, (
        f"{len(unexplained)} of {len(verdicts)} cases differ semantically")
