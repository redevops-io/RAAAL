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


def test_the_mode_is_the_one_fusion_uses():
    """`REQUIREMENTS` wins over the schema, because fusion reads the requirement.

    The two disagree for exactly one dimension: `stated_weights` is WEIGHTS in
    the requirement and SET in the schema. Reading the schema compared `60/40`
    against `VTI=60,BND=40` as unordered tokens, called them different, and
    refused a split the compiler had been handed — so the source that matters
    is the one fusion consults.

    Two sources for one fact is its own defect and is not the adapter's to
    resolve. What the adapter must not do is pick the other one.
    """
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.vocabulary import REQUIREMENTS

    modes = adapter.compare_modes()
    assert len(modes) >= len(QUANTIFY_SCHEMA.dimensions)

    from src.discovery.vocabulary import PERIOD_DIMENSIONS

    for name, requirement in REQUIREMENTS.items():
        declared = getattr(requirement, "compare_as", "")
        if not declared:
            continue
        # A numeric dimension that counts periods refines NUMBER to PERIOD.
        # That is the one permitted departure from the requirement and it is
        # enumerated, not a wildcard: any other divergence is the schema
        # winning over fusion again.
        if declared == "NUMBER" and name in PERIOD_DIMENSIONS:
            assert modes[name] == "PERIOD", f"{name}: {modes[name]}"
            continue
        if declared == "SET":
            assert modes[name] == "HOLDINGS", f"{name}: {modes[name]}"
            continue
        assert modes[name] == declared, (
            f"{name}: the adapter reports {modes[name]} and fusion's "
            f"requirement says {declared}")

    assert modes["stated_weights"] == "WEIGHTS", (
        "the known divergence regressed to the schema's SET")
    assert "SET" not in modes.values(), (
        "a SET dimension is comparing generically, so `an SPX ETF` and `SPX "
        "ETF` are two holdings again")


def test_only_domain_modes_are_supplied():
    """`TEXT` and `SET` belong to the runtime, and we do not redefine them.

    Redefining a generic mode would replace a rule that works in any language
    with one that happens to agree today — and the drift would be invisible,
    because both would still be called SET.

    The four supplied here are all facts about English finance: what `£2.5k` is
    worth, that `12m` is twelve months in a window and twelve million in an
    amount, that `60/40` and `VTI=60,BND=40` are the same split with one of
    them saying which holding takes which share, and that `an SPX ETF` and `SPX
    ETF` name one holding.

    `HOLDINGS` is the one that had to be learned. It was left to the runtime's
    SET on the argument that the generic rule agreed with the domain one, and
    two corpus cases showed it does not — `the S&P 500 tracker` against `S&P
    500 tracker` was reported as a disagreement. The fix is a new mode rather
    than an override of SET, so the generic rule stays generic.
    """
    assert set(adapter.NORMALIZERS) == {"NUMBER", "PERIOD", "HOLDINGS", "WEIGHTS"}
    assert not {"TEXT", "SET"} & set(adapter.NORMALIZERS), (
        "a generic mode has been redefined; the runtime's rule and ours now "
        "differ under one name")


def test_the_holdings_rule_absorbs_a_determiner_and_nothing_more():
    """The line the rule must not cross.

    Dropping `a|an|the` removes one closed class of English function word.
    Anything wider starts making two different holdings equal, which is the
    substitution the whole boundary exists to prevent — a reader must never
    turn "an S&P 500 tracker" into VTI on the person's behalf.
    """
    assert adapter.same_value_for("assets", "an SPX ETF", "SPX ETF")
    assert adapter.same_value_for("assets", "the S&P 500 tracker",
                                  "S&P 500 tracker")
    assert adapter.same_value_for("assets", "VTI and BND", "BND, VTI")

    assert not adapter.same_value_for("assets", "SPX ETF", "SPY")
    assert not adapter.same_value_for("assets", "an S&P 500 tracker", "VTI")
    assert not adapter.same_value_for("assets", "VTI, BND", "VTI")


def test_the_period_mode_reaches_the_dimensions_that_count_periods():
    """`PERIOD` is a mode, not a branch inside the comparison.

    The old comparison took the dimension name as an argument and asked
    `dimension in PERIOD_DIMENSIONS` inside itself. The runtime keys
    normalisers by mode, so the distinction has to become a mode — and if the
    assignment were dropped the modes would all read NUMBER and a window
    written `12m` would settle as twelve million sessions.
    """
    from src.discovery.vocabulary import PERIOD_DIMENSIONS

    modes = adapter.compare_modes()
    assert modes["moving_average_window"] == "PERIOD"
    assert modes["amount"] == "NUMBER", (
        "an amount was swept into PERIOD, so `12m` no longer means millions")

    # Only dimensions that compare numerically can be scaled, so only those
    # need protecting from it. The rest are recorded below rather than
    # asserted, because the reachable set is smaller than the list suggests.
    for name in PERIOD_DIMENSIONS:
        assert modes.get(name) in ("PERIOD", "TEXT", None), (
            f"{name} compares as {modes[name]} and would scale `12m` to "
            "twelve million")


def test_most_of_the_period_vocabulary_never_reaches_the_numeric_path():
    """Recorded, because the name `PERIOD_DIMENSIONS` overstates its reach.

    Five dimensions are listed and one compares numerically. Two others
    compare as TEXT — where `12m` and `12` are simply unequal strings, no
    silent scaling but no equality either — and two are not schema dimensions
    at all. None of that changed in the migration; the numeric branch was
    always guarded by the mode. It is written down so nobody reads the list as
    five protected dimensions, and so that promoting one of them to NUMBER
    fails here rather than settling a twelve-million-session window.
    """
    from src.discovery.vocabulary import PERIOD_DIMENSIONS

    modes = adapter.compare_modes()
    numeric = {n for n in PERIOD_DIMENSIONS if modes.get(n) in ("NUMBER", "PERIOD")}
    textual = {n for n in PERIOD_DIMENSIONS if modes.get(n) == "TEXT"}
    absent = {n for n in PERIOD_DIMENSIONS if n not in modes}

    assert numeric == {"moving_average_window"}, numeric
    assert textual == {"evaluation_period", "holding_period"}, textual
    assert absent == {"lookback_window", "rebalancing_period"}, absent


def test_one_dimension_has_one_mode():
    """`compare_as` and `compare_modes` answer the same question.

    They did not. `compare_as` read the schema and fusion read the
    requirement, so `stated_weights` was SET to one caller and WEIGHTS to the
    other — a divergence nothing detected because both answers were plausible
    mode names.
    """
    modes = adapter.compare_modes()
    disagree = {name: (adapter.compare_as(name), mode)
                for name, mode in modes.items() if adapter.compare_as(name) != mode}
    assert not disagree, f"two sources give different modes: {disagree}"


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
