"""What the migration deleted, and why the rest stayed.

The deletion was frozen as four modules: `fusion`, `pipeline`, `reader`,
`witnesses`. Two are gone. The other two turned out to hold no Discovery
Runtime semantics at all — they hold Quantify's schema types and its witness
and provenance vocabulary, none of which exists upstream — so deleting them
would have removed domain code and forced it to be written again somewhere
else under a different name.

That is a judgement call, and a judgement call recorded in a commit message is
one nobody can re-check. So it is recorded here as evidence instead:

    for each module on the original list, either it is gone, or it names the
    Quantify-specific content that kept it — and that content is asserted to
    exist here and to have no upstream equivalent

The consequence that matters is the second half. If `SettledField` or the
witness vocabulary is ever moved upstream, this file fails and the deletion
decision gets revisited, rather than the module sitting here forever because
the reason it stayed was true once.

**`Reading` is the one real collision**, and it is a collision of names rather
than of behaviour: Quantify's is one dimension's reading and the runtime's is a
whole reader's output. Asserted below, because "they are different really" is
exactly the sentence that precedes two types drifting into one job.
"""
from __future__ import annotations

import dataclasses
import importlib

import pytest

#: Module -> (what kept it, the names that are the reason). An empty tuple
#: means it was deleted and must stay deleted.
FROZEN_LIST = {
    "src.discovery.fusion": (
        "deleted: reader fusion, comparison semantics and the outcome enum, "
        "every one of which discovery-runtime now owns. Its domain content "
        "left first — the vocabulary to vocabulary.py, the normalisers and "
        "the aggregation seam to adapter.py — so what was deleted was the "
        "generic half and nothing else.", ()),
    "src.discovery.pipeline": (
        "deleted: generic orchestration of readers into a merged reading. "
        "What Quantify still decides — which readers run, in what order, and "
        "how their observations become one claim per dimension — is in "
        "adapter.two_witness_run, which is orchestration and not semantics.",
        ()),
    "src.discovery.reader": (
        "kept: Quantify's schema and its readers' output shape. The runtime "
        "takes `schema` opaque and never looks inside it, so what a dimension "
        "is, which relations exist and what a Quantify reader returns are all "
        "ours to define.",
        ("Schema", "Dimension", "RelationSpec", "ReadingSet",
         "RelationReading", "DiscoveryReader")),
    "src.discovery.witnesses": (
        "kept: which witnesses exist and what to persist about how a field "
        "was settled. `AGREE` is what fusion concluded; `MODEL_ONLY_ACCEPTED` "
        "is what happened, and only Quantify knows the difference.",
        ("Witness", "WitnessProfile", "SettledField", "provenance_of",
         "record", "witnesses_of")),
}

DELETED = [name for name, (_, kept) in FROZEN_LIST.items() if not kept]
SURVIVING = [name for name, (_, kept) in FROZEN_LIST.items() if kept]


@pytest.mark.parametrize("module", DELETED)
def test_a_deleted_module_stays_deleted(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


@pytest.mark.parametrize("module", SURVIVING)
def test_a_surviving_module_still_holds_what_kept_it(module):
    """The names in the record are the names in the module.

    A module kept for `SettledField` and no longer defining it is a module
    whose reason to exist has quietly expired.
    """
    loaded = importlib.import_module(module)
    _, kept = FROZEN_LIST[module]
    missing = [name for name in kept if not hasattr(loaded, name)]
    assert not missing, (
        f"{module} was kept for {missing}, which it no longer defines. The "
        "deletion decision was made on content that is gone.")


@pytest.mark.parametrize("module", SURVIVING)
def test_what_kept_it_has_no_upstream_equivalent(module):
    """The half that can expire.

    Every name here is Quantify's because nothing upstream provides it. Move
    one upstream and this fails — which is the point: the module then holds a
    duplicate, and staying is no longer the right call.

    `Reading` is deliberately absent from every list. It exists in both places
    and is two different things, which the test below asserts rather than
    assumes.
    """
    import discovery_runtime
    import runtime_contracts

    _, kept = FROZEN_LIST[module]
    upstream = set(dir(discovery_runtime)) | set(dir(runtime_contracts))
    collisions = sorted(set(kept) & upstream)
    assert not collisions, (
        f"{collisions} now exist upstream as well as in {module}. Whatever "
        "moved, this module is holding a second copy of it and the deletion "
        "decision needs remaking.")


def test_the_two_readings_are_different_things():
    """The one name that exists in both, asserted as two concepts.

    Quantify's `Reading` is one dimension's reading — a value, a span, a
    confidence. The runtime's is a whole reader's output — a payload of every
    dimension, with evidence. Nothing converts between them by accident
    because they share no field at all, and that is what this checks: if they
    ever grow a common shape, one of them is becoming the other.
    """
    from discovery_runtime import Reading as RuntimeReading

    from src.discovery.reader import Reading as QuantifyReading

    runtime = {f.name for f in dataclasses.fields(RuntimeReading)}
    quantify = {f.name for f in dataclasses.fields(QuantifyReading)}

    assert runtime == {"payload", "evidence", "unresolved", "relations"}
    assert quantify == {"dimension", "value", "confidence", "source_span", "note"}
    assert not runtime & quantify, (
        f"the two Reading types now share {sorted(runtime & quantify)}. They "
        "were kept apart because they answer different questions; a shared "
        "field is the first step to one being passed where the other is meant.")


def test_every_module_on_the_frozen_list_is_accounted_for():
    """Neither list is empty, and no module is on both.

    A record where everything was kept is a record of a deletion that did not
    happen, and one where everything was deleted cannot be true — the schema
    had to survive.
    """
    assert DELETED and SURVIVING
    assert not set(DELETED) & set(SURVIVING)
    assert len(FROZEN_LIST) == 4, (
        "the frozen deletion list had four modules; this record has "
        f"{len(FROZEN_LIST)}")


@pytest.mark.parametrize("module", sorted(FROZEN_LIST))
def test_each_decision_is_argued(module):
    """"We do not need it" is what somebody writes when they have not looked."""
    reason, _ = FROZEN_LIST[module]
    assert len(reason.split()) >= 8, f"{module}: {reason!r}"
    assert reason.startswith(("deleted:", "kept:")), module
