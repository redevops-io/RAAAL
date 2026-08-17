"""The adapter supplies the schema's rules and implements none of them.

Its whole justification is that fusion, comparison and sealing now live in one
place. An adapter that quietly reimplemented any of them would restore the
duplication the migration exists to remove, while looking like configuration —
so "contains no fusion logic" is asserted structurally rather than reviewed.
"""
from __future__ import annotations

import ast
import pathlib
from decimal import Decimal

import pytest

from src.discovery import adapter

ADAPTER = pathlib.Path(adapter.__file__)


def test_every_dimension_has_a_mode_and_it_comes_from_the_schema():
    modes = adapter.compare_modes()
    from src.discovery.schema import QUANTIFY_SCHEMA

    assert len(modes) == len(QUANTIFY_SCHEMA.dimensions)
    for dimension in QUANTIFY_SCHEMA.dimensions:
        assert modes[dimension.name] == dimension.compare_as, (
            f"{dimension.name}: the adapter reports {modes[dimension.name]} "
            f"and the schema says {dimension.compare_as}")


def test_only_the_domain_mode_is_supplied():
    """`TEXT` and `SET` belong to the runtime.

    Supplying our own would replace a generic rule with a domain one that
    happens to agree today and can drift tomorrow — and the drift would be
    invisible, because both would still be called TEXT.
    """
    assert set(adapter.NORMALIZERS) == {"NUMBER"}


def test_the_normaliser_is_the_readers_own():
    """Not a second numeric parser.

    Two places deciding what a written number means is how the deterministic
    path and the model came to disagree about `$500`.
    """
    assert adapter.number("£2.5k") == Decimal(2500)
    assert adapter.number("$500") == Decimal(500)
    assert adapter.number("1,000") == Decimal(1000)
    assert adapter.number("500") == Decimal(500)


def test_an_unreadable_value_is_none_not_a_guess():
    """`None` means "not equal to anything", including another unreadable one."""
    assert adapter.number("bananas") is None
    assert adapter.number("") is None


def test_an_unknown_dimension_compares_exactly():
    assert adapter.compare_as("no_such_dimension") == "TEXT"


def test_the_fusion_policy_carries_the_schemas_rules():
    """The regression the closure exists to prevent.

    A caller that built `merge_readings` itself and forgot `normalizers` would
    get exact comparison on amounts — a clarification question on every `$500`
    a reader had normalised.
    """
    from runtime_contracts import DecisionEvidence, ReaderKind

    from discovery_runtime import Reading

    def ev(reader_id, value):
        return DecisionEvidence(reader_id=reader_id, kind=ReaderKind.RULE,
                                value=value, source_ref="s")

    readings = [
        Reading(payload={"amount": "$500"}, evidence={"amount": [ev("model", "$500")]}),
        Reading(payload={"amount": "500"}, evidence={"amount": [ev("rules", "500")]}),
    ]
    fused = adapter.fusion_policy()(readings)
    assert not fused.unresolved, (
        "the adapter's policy did not carry the schema's NUMBER rule, so two "
        "readers that agree were reported as differing")


def test_the_policy_still_reports_a_real_disagreement():
    """The other half. A policy that reconciles everything is not a policy."""
    from runtime_contracts import DecisionEvidence, ReaderKind

    from discovery_runtime import Reading

    def ev(reader_id, value):
        return DecisionEvidence(reader_id=reader_id, kind=ReaderKind.RULE,
                                value=value, source_ref="s")

    readings = [
        Reading(payload={"amount": "$500"}, evidence={"amount": [ev("model", "$500")]}),
        Reading(payload={"amount": "$600"}, evidence={"amount": [ev("rules", "$600")]}),
    ]
    assert [u.dimension for u in adapter.fusion_policy()(readings).unresolved] == ["amount"]


#: Names that would mean this module had grown its own copy of what it is
#: supposed to be delegating.
FORBIDDEN_DEFINITIONS = {"fuse", "same_value", "merge_readings", "seal",
                         "digest", "Fusion", "Decision", "Proposal"}


def test_the_adapter_defines_no_fusion_or_sealing_of_its_own():
    """Structural, on the syntax tree.

    A grep would match this list and the docstring above it. The property is
    "no definition with these names", which is a fact about the tree.
    """
    tree = ast.parse(ADAPTER.read_text())
    defined = {node.name for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef))}
    clashes = defined & FORBIDDEN_DEFINITIONS
    assert not clashes, (
        f"the adapter defines {sorted(clashes)}. Those belong to "
        "discovery-runtime; defining them here restores the duplication the "
        "migration removes, while looking like configuration.")


def test_the_adapter_gets_its_machinery_from_the_runtime():
    """And that it delegates at all.

    An adapter that imported nothing from `discovery_runtime` would satisfy the
    test above by doing nothing, which is the way that check fails silently.
    """
    tree = ast.parse(ADAPTER.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            if (node.module or "").split(".")[0] == "discovery_runtime":
                imported.update(a.name for a in node.names)
    assert {"merge_readings", "DiscoveryRuntime"} <= imported, (
        f"the adapter imports {sorted(imported)} from discovery_runtime; it is "
        "supposed to be wiring that package up")
