"""Every semantic seam the runtime exposes is classified by Quantify.

The migration's recurring defect was never a wrong answer. It was an omission:
a place the runtime was ready to receive a domain fact, and Quantify not
sending one. Four times.

    canonicalisation      the runtime sealed `amount='4%'` because the adapter
                          passed no canonicalizer and the upstream default
                          copied the payload verbatim
    relation evidence     relation markers carried no evidence, so authorship
                          fell back to a generic READER against the internal
                          path's MODEL
    ambiguity             "rebalance back to 60/40" settled silently, because
                          nothing told the runtime the word carries two
                          readings
    materiality           an open dimension blocked sealing that the internal
                          path treats as immaterial

Each was found by a corpus case failing, which is late and lucky. This makes
omission structural: a seam that exists and is not classified fails here rather
than in whichever sentence happens to exercise it.

**What this guard does not do, stated so the green does not read as more than
it is.** It detects *omitted existing seams*. It cannot detect semantics the
runtime does not yet expose as a seam at all — three of the four above did not
exist as parameters until a Quantify case demanded them, and this test would
have been green throughout. Coverage of the seam list is not coverage of the
semantics.
"""
from __future__ import annotations

import dataclasses
import inspect

import pytest

from discovery_runtime import DiscoveryRuntime, draft_intent, merge_readings

from src.discovery import adapter

SUPPLIED = "SUPPLIED"
NOT_APPLICABLE = "NOT_APPLICABLE"

#: Seams the runtime exposes as injection points, and what Quantify does about
#: each. A seam classified NOT_APPLICABLE must say why — "we do not need it" is
#: the sentence somebody writes when they have not looked.
CLASSIFICATION = {
    "canonicalize": (
        SUPPLIED,
        "adapter.canonicalizer wraps discovery.canonical.canonicalise and "
        "folds its refusals into unresolved"),
    "fusion_policy": (
        SUPPLIED,
        "adapter.fusion_policy closes over compare modes, normalisers, "
        "ambiguity and materiality"),
    "compare_as": (
        SUPPLIED,
        "adapter.compare_modes, read from QUANTIFY_SCHEMA rather than restated"),
    "normalizers": (
        SUPPLIED,
        "adapter.NORMALIZERS: NUMBER via discovery.syntax.normalize. TEXT and "
        "SET are the runtime's own and deliberately not overridden"),
    "ambiguity": (
        SUPPLIED,
        "adapter.ambiguity, from discovery.fusion.AMBIGUOUS_TERMS, which stays "
        "local with its sources"),
    "material": (
        SUPPLIED,
        "adapter.material, read from discovery.fusion.REQUIREMENTS"),
    "readers": (
        SUPPLIED,
        "adapter.ReaderAdapter presents Quantify readers in the runtime's "
        "protocol and carries relations and their evidence"),
    "schema": (
        SUPPLIED,
        "QUANTIFY_SCHEMA, passed opaque so a caller can introspect its own "
        "dimensions; the runtime never looks inside it"),
    "objective": (
        SUPPLIED,
        "adapter.runtime takes it per call: what a request is for is a domain "
        "statement the runtime cannot invent"),
}

#: Seams Quantify implements around the runtime rather than inside it. Listed
#: because they are semantic responsibilities at the same boundary, and an
#: omission here has the same consequence — but they are not runtime
#: parameters, so nothing can introspect them and this list is maintained by
#: hand. That is a weakness and is written down as one.
ADAPTER_SIDE = {
    "author classification": (
        SUPPLIED,
        "adapter.classify_authors maps witness kind to Author after drafting; "
        "the runtime is right not to guess"),
    "relation handling": (
        SUPPLIED,
        "adapter.as_intent_relation translates RelationReading to the "
        "contract's IntentRelation; adapter.relation_fields flattens kinds for "
        "Mission's compiler"),
    "SET normalisation": (
        SUPPLIED,
        "adapter.one_reading_per_set_dimension unions members a reader emitted "
        "separately, on both lanes, through one function"),
}


def _runtime_seams() -> set:
    """Every injection point the runtime exposes, by introspection.

    Read from the code rather than listed, so a seam added upstream appears
    here without anybody remembering to add it — which is the whole point.
    """
    seams = {f.name for f in dataclasses.fields(DiscoveryRuntime)}
    for function in (merge_readings, draft_intent):
        seams |= {name for name, p in inspect.signature(function).parameters.items()
                  if p.kind is p.KEYWORD_ONLY}
    return seams


def test_every_runtime_seam_is_classified():
    """A new seam upstream fails here until Quantify decides about it."""
    unclassified = sorted(_runtime_seams() - set(CLASSIFICATION))
    assert not unclassified, (
        f"discovery-runtime exposes {unclassified} and Quantify has not "
        "classified them. Every seam is SUPPLIED or NOT_APPLICABLE with a "
        "reason — an unclassified seam is one nothing is sending a domain fact "
        "through, which is how canonicalisation, ambiguity and materiality "
        "were each missed until a corpus case failed.")


def test_no_classified_seam_has_disappeared():
    """A seam removed upstream leaves a classification describing nothing.

    Worse than tidiness: the reason text goes on asserting that Quantify
    supplies something, and the next reader believes it.
    """
    gone = sorted(set(CLASSIFICATION) - _runtime_seams())
    assert not gone, (
        f"{gone} are classified here and no longer exist upstream")


@pytest.mark.parametrize("seam", sorted(set(CLASSIFICATION) | set(ADAPTER_SIDE)))
def test_each_classification_is_usable(seam):
    """SUPPLIED or NOT_APPLICABLE, and never a bare assertion."""
    verdict, reason = {**CLASSIFICATION, **ADAPTER_SIDE}[seam]
    assert verdict in (SUPPLIED, NOT_APPLICABLE), f"{seam}: {verdict}"
    assert len(reason.strip()) > 30, (
        f"{seam} is marked {verdict} without a usable reason. "
        "'we do not need it' is what somebody writes when they have not looked.")


def test_every_supplied_seam_names_something_that_exists():
    """The reasons point at real attributes, not remembered ones.

    A classification naming a function that has been renamed is a claim nobody
    can check, which is the failure mode of every hand-maintained list.
    """
    named = ("canonicalizer", "fusion_policy", "compare_modes", "NORMALIZERS",
             "ambiguity", "material", "ReaderAdapter", "classify_authors",
             "as_intent_relation", "relation_fields",
             "one_reading_per_set_dimension", "runtime")
    missing = [n for n in named if not hasattr(adapter, n)]
    assert not missing, (
        f"the classifications name {missing}, which the adapter does not have")


def test_the_supplied_seams_are_actually_wired():
    """Classified is not the same as connected.

    A seam can be SUPPLIED in this table and still not reach the runtime — the
    ambiguity seam existed for a release before anything passed it. This builds
    the configured runtime and reads back what it carries.
    """
    built = adapter.runtime([], objective="evaluate_investment_strategy")
    assert built.canonicalize is not None
    assert built.fusion_policy is not None
    assert built.schema is not None
    assert built.objective, "objective is empty; the contract requires one"

    # The policy must carry the schema's rules, not merely be present.
    import inspect as _inspect

    closure = _inspect.getclosurevars(built.fusion_policy)
    assert "modes" in closure.nonlocals, (
        "fusion_policy does not close over the compare modes, so a caller gets "
        "exact comparison on amounts")


def test_the_limitation_is_recorded():
    """The guard must say what it cannot do, in itself.

    Three of the four omissions this exists for did not exist as seams when
    they were missed, and this test would have been green throughout. A reader
    seeing it pass must not conclude the boundary is covered.
    """
    # Whitespace-normalised: the sentence wraps in the docstring, and an
    # assertion that depends on where it wraps fails on a reflow that changed
    # nothing.
    flat = " ".join(__doc__.split())
    assert "cannot detect semantics the runtime does not yet expose" in flat
    assert "Coverage of the seam list is not coverage of the semantics" in flat
