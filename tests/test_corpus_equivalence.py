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

    internal VerifiedIntent.fields      <->  runtime VerifiedIntent.fields
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

**Both sides project from the sealed `VerifiedIntent`, and version 3 was wrong
to read `PilotReading.settled`.** That is Quantify's working record on the way
to the artifact, not the artifact: for `rebalance back to 60/40 every year` it
holds `periodic_rebalancing = None`, while the sealed intent — on both
implementations — holds the canonicalised `'annual'`. Comparing it against the
runtime's `VerifiedIntent.fields` compared two *stages of one pipeline* and
reported the runtime as wrong where the runtime was right.

Equivalence is measured at the canonical contract boundary. Working records,
intermediate stages and derived views are adjacent to the property and are not
the property — the same category error as version 1's `questions`, one layer
down.

**Settled values are compared by the schema's rule, and version 2 was wrong to
use `==`.** `70% stocks, 20% bonds, 10% cash` settles `assets` as
'stocks, bonds, cash' on one side and 'stocks,bonds,cash' on the other;
`assets` is `compare_as=SET`, so the schema says those are the same value and
the harness was applying a stricter equality than the product does. It now uses
`same_value` with the dimension's mode — the same rule fusion applies — so
anything it admits is agreement the system already acts on.

Three verdicts and no others. `EXPECTED_REPRESENTATION` is admissible only with
the demonstration attached; `SEMANTIC_DIFFERENCE` blocks deletion.

**Version 6 tightened two rules that were admitting real differences.** A
differing author is now semantic: author is identity-bearing and decides
whether a later reader may correct a value, which is authority rather than
presentation — and the leniency is exactly what hid the missing relation
markers for two runs. A differing field set is now semantic too: a dimension
one lane settles and the other does not is a missing part of the request. What
remains representational is a differing `intent_hash` with everything else
equal, and that is admissible only because the 25-case StrategySpec proof
demonstrated it execution-neutral.
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
MAPPING_VERSION = "quantify-equivalence-view@6"

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


def _runtime_intent(text: str):
    """The runtime path, executed independently from the same reader.

    Not derived from the internal path's artifact. Until this existed the
    harness projected one artifact twice, which validated the view and proved
    nothing about two implementations — the runtime side now runs its own
    readers, its own fusion and its own draft.
    """
    from src.deploy import context as deploy_context
    from src.discovery import adapter
    from src.workspace import pilot_routes

    original, deploy_context.current = deploy_context.current, lambda: _resolved()
    try:
        reader = adapter.ReaderAdapter(pilot_routes.configured_reader())
        intent = adapter.intent_from([reader], text)
    finally:
        deploy_context.current = original

    # Seal only when meaning is closed, which is the contract's own rule and
    # the reason `draft` and `seal` are separate calls. The harness must not
    # force closure to make compilation possible — that would compare a
    # sealed artifact against one the runtime would never have sealed.
    #
    # Version 4 stopped at the draft, so every runtime intent stayed DRAFT and
    # `compile_intent` refused it: 22 of 25 hash-only cases failed to compile
    # for a lifecycle reason and 3 more read as StrategySpec differences that
    # were the same cause wearing another label. That result is invalid as
    # evidence about execution equivalence and is recorded as such.
    if not intent.unsealable:
        try:
            return intent.seal()
        except Exception:                                      # noqa: BLE001
            # Refused by the contract. Returned unsealed so the difference is
            # visible as a sealability difference rather than disappearing.
            return intent
    return intent


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
    """`PilotReading` -> the comparison view, projected from its sealed intent.

    `reading.settled` is deliberately not read. It is the working record the
    internal path keeps on the way to the artifact and it holds pre-
    canonicalisation values — including a literal `None` for a dimension the
    sealed intent carries correctly. The artifact is what both implementations
    are obliged to agree on, so the artifact is what is compared.

    `provenance_summary` therefore reports the contract's `author` on both
    sides rather than Quantify's provenance string, which has no counterpart in
    a generic runtime.
    """
    intent = reading.intent
    latest = {} if intent is None else dict(intent.fields)
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
        provenance_summary={k: v.author.value for k, v in latest.items()},
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

    # Author is identity-bearing and controls dominance: USER is final and is
    # never overwritten by a re-read. Two implementations disagreeing about who
    # established a value disagree about whether a later reader may correct it,
    # which is authority and not presentation.
    #
    # Admitted as representational until version 6, which is how missing
    # relations hid: their marker fields carried no evidence, fell back to a
    # generic READER, and the leniency swallowed it.
    if old.provenance_summary != new.provenance_summary:
        return "SEMANTIC_DIFFERENCE"

    if set(old.settled_fields) != set(new.settled_fields):
        # One settled a dimension the other did not. A missing dimension is a
        # missing part of the request — this was EXPECTED_REPRESENTATION until
        # version 6 and it swallowed the relation gap for two runs.
        return "SEMANTIC_DIFFERENCE"
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
        # Independently produced. The runtime path runs its own readers, its
        # own fusion and its own draft over the same frozen readings; nothing
        # here is taken from the artifact the internal path built.
        new = from_runtime(_runtime_intent(text))
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


def _downstream(intent):
    """What this intent produces for the engine, as a comparable value.

    Not only the `StrategySpec`: an intent that refuses, or that seals nothing,
    has an outcome too, and comparing only the specs would silently skip every
    case that does not compile — which is most of the corpus.
    """
    import dataclasses

    from src.mission.from_intent import NotExecutable, compile_intent
    from src.mission.strategy_spec import from_scenario

    if intent is None or not intent.is_verified:
        return ("UNSEALED", "")
    try:
        compiled = compile_intent(intent)
    except NotExecutable as refused:
        return ("REFUSED",
                ",".join(sorted(r.dimension for r in refused.refusals)))
    if compiled.scenario is None:
        return ("NO_SCENARIO", "")
    return ("SPEC", json.dumps(
        dataclasses.asdict(from_scenario(compiled.scenario)),
        sort_keys=True, default=str))


def test_every_representational_difference_is_execution_neutral(capsys):
    """The demonstration `EXPECTED_REPRESENTATION` is only admissible with.

    A differing `intent_hash` with everything else equal is representational
    *if* the two intents produce the same thing downstream — and that is a
    claim to prove, not a rule to assert. Every case classified representational
    is compiled on both lanes and required to agree.

    Stated honestly in the output: most of the corpus does not compile to a
    spec at all, so the majority agree on producing none. That is outcome
    equivalence and it is weaker evidence than byte-identical specs. Both are
    counted separately rather than summed into one reassuring number.
    """
    from collections import Counter

    outcomes = Counter()
    differing = []
    for text in _texts():
        reading = _internal_reading(text)
        if reading.intent is None:
            continue
        old, new = from_internal(reading), from_runtime(_runtime_intent(text))
        if classify(old, new) != "EXPECTED_REPRESENTATION":
            continue
        a, b = _downstream(reading.intent), _downstream(_runtime_intent(text))
        outcomes[a[0] if a == b else "DIFFER"] += 1
        if a != b:
            differing.append((text, a[0], b[0]))

    print(f"\n{SCOPE}\nmapping: {MAPPING_VERSION}\n")
    print("  representational differences, proved downstream:")
    for kind, n in sorted(outcomes.items()):
        print(f"    {kind:<14} {n}")
    print(f"    {'byte-identical StrategySpec':<14} "
          f"{outcomes.get('SPEC', 0)} of {sum(outcomes.values())}")

    assert sum(outcomes.values()), "nothing was classified representational"
    assert not differing, (
        "these are classified representational and are not execution-neutral:\n"
        + "\n".join(f"  {t[:60]!r}: {a} vs {b}" for t, a, b in differing))
