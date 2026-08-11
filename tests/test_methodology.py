"""Conformance tests for Methodology Specification 0.1.

These are the implementation-independent checks the spec promises: identity,
canonical hashing, contract compatibility, and the merge semantics table. If a
second runner is ever written, this file is what it must satisfy.
"""
from __future__ import annotations

import pytest

from src.methodology import (
    Citation,
    ComparabilityStatus,
    ContractStatus,
    EconomicStatus,
    Methodology,
    MethodologyRegistry,
    OutputContract,
    Param,
    Rule,
    StructuralStatus,
    from_dict,
    merge,
)
from src.methodology.spec import FIELD_SEMANTICS, REQUIRED_DISCLOSURE, PerformanceClass


def _base(**overrides) -> Methodology:
    defaults = dict(
        concept="demo",
        version=1,
        title="Demo",
        objective="test fixture",
        contract=OutputContract(universe=("SPY", "TLT", "BIL"), rebalance_frequency="5B"),
        params={"lookback": Param(value=252, unit="trading_days")},
        rules=(Rule(id="min_history", enforced_by="params.lookback", expected=">= 252"),),
        pipeline=("estimate", "allocate"),
        scoring_terms={"momentum": 0.5, "value": 0.5},
        grounded_in=(Citation(identifier="doi:10.0/x", title="A paper"),),
    )
    defaults.update(overrides)
    return Methodology(**defaults)


class TestIdentity:
    def test_concept_and_version_ids(self):
        m = _base()
        assert m.concept_id == "methodology/demo"
        assert m.version_id == "methodology/demo@1"

    def test_hash_is_stable_across_instances(self):
        assert _base().content_hash == _base().content_hash

    def test_hash_ignores_documentation_fields(self):
        """Editing prose must not mint a new version."""
        a = _base(title="Demo")
        b = _base(title="Completely Different Title")
        assert a.content_hash == b.content_hash

    def test_hash_detects_parameter_change(self):
        """Changing a threshold must mint a new version."""
        a = _base()
        b = _base(params={"lookback": Param(value=504, unit="trading_days")})
        assert a.content_hash != b.content_hash

    def test_hash_is_order_independent_for_unordered_fields(self):
        a = _base(excluded_assets=("SPY", "TLT"))
        b = _base(excluded_assets=("TLT", "SPY"))
        assert a.content_hash == b.content_hash

    def test_hash_is_order_dependent_for_pipeline(self):
        """Pipeline order carries meaning, so it must affect identity."""
        a = _base(pipeline=("estimate", "allocate"))
        b = _base(pipeline=("allocate", "estimate"))
        assert a.content_hash != b.content_hash

    def test_roundtrip_through_dict(self):
        m = _base()
        assert from_dict(m.to_json()).content_hash == m.content_hash


class TestRevision:
    def test_revise_records_lineage(self):
        v1 = _base()
        v2 = v1.revise(change_rationale="widen lookback",
                       params={"lookback": Param(value=504)})
        assert v2.version == 2
        assert v2.derived_from == "methodology/demo@1"
        assert v2.content_hash != v1.content_hash

    def test_revise_requires_a_rationale(self):
        with pytest.raises(ValueError, match="change_rationale"):
            _base().revise(change_rationale="   ")


class TestContract:
    def test_identical_contracts_are_compatible(self):
        c = OutputContract(universe=("SPY",), rebalance_frequency="5B")
        assert c.breaks_compatibility_with(c) == []

    def test_rebalance_change_breaks_compatibility(self):
        a = OutputContract(universe=("SPY",), rebalance_frequency="5B")
        b = OutputContract(universe=("SPY",), rebalance_frequency="21B")
        assert any("rebalance" in r for r in b.breaks_compatibility_with(a))

    def test_universe_removal_breaks_compatibility(self):
        a = OutputContract(universe=("SPY", "TLT"), rebalance_frequency="5B")
        b = OutputContract(universe=("SPY",), rebalance_frequency="5B")
        assert any("removed" in r for r in b.breaks_compatibility_with(a))

    def test_dropping_cost_model_breaks_compatibility(self):
        a = OutputContract(universe=("SPY",), rebalance_frequency="5B", requires_cost_model=True)
        b = OutputContract(universe=("SPY",), rebalance_frequency="5B", requires_cost_model=False)
        assert any("cost model" in r for r in b.breaks_compatibility_with(a))


class TestSemanticsTable:
    def test_every_mergeable_field_declares_semantics(self):
        """The merge layer refuses fields absent from the table, so adding a node
        type must force an explicit decision about how it combines."""
        for name in ("rules", "constraints", "pipeline", "fallback_chain",
                     "weights", "universe_filters", "scoring_terms", "params"):
            assert name in FIELD_SEMANTICS

    def test_pipeline_and_fallback_are_ordered(self):
        assert FIELD_SEMANTICS["pipeline"].value == "ordered_sequence"
        assert FIELD_SEMANTICS["fallback_chain"].value == "ordered_sequence"

    def test_weights_are_normalized_map_not_plain_set(self):
        """The distinction that prevents a clean-but-invalid portfolio merge."""
        assert FIELD_SEMANTICS["weights"].value == "normalized_map"


class TestDisclosure:
    def test_every_performance_class_has_a_disclosure(self):
        for pc in PerformanceClass:
            assert REQUIRED_DISCLOSURE[pc].strip()

    def test_backtest_disclosure_names_the_rule(self):
        text = REQUIRED_DISCLOSURE[PerformanceClass.BACKTEST_HYPOTHETICAL]
        assert "206(4)-1" in text
        assert "not a track record" in text.lower()


class TestMerge:
    def test_independent_rule_additions_merge_silently(self):
        base = _base()
        ours = _base(rules=base.rules + (Rule(id="vol_filter", enforced_by="params.lookback", expected=">= 1"),))
        theirs = _base(rules=base.rules + (Rule(id="liq_filter", enforced_by="params.lookback", expected=">= 2"),))

        result = merge(base, ours, theirs)

        assert result.structural_status is StructuralStatus.CLEAN
        assert result.merged is not None
        ids = {r.id for r in result.merged.rules}
        assert ids == {"min_history", "vol_filter", "liq_filter"}

    def test_conflicting_pipeline_reorder_is_a_conflict(self):
        """Ordered sequences must not be silently interleaved."""
        base = _base()
        ours = _base(pipeline=("allocate", "estimate"))
        theirs = _base(pipeline=("estimate", "validate", "allocate"))

        result = merge(base, ours, theirs)

        assert result.structural_status is StructuralStatus.CONFLICTED
        assert any(c.field == "pipeline" for c in result.unresolved_conflicts)
        assert not result.publishable

    def test_conflicting_param_edit_is_a_conflict(self):
        base = _base()
        ours = _base(params={"lookback": Param(value=126)})
        theirs = _base(params={"lookback": Param(value=504)})

        result = merge(base, ours, theirs)

        assert any(c.field == "params" for c in result.unresolved_conflicts)

    def test_one_sided_param_edit_is_taken(self):
        base = _base()
        ours = _base(params={"lookback": Param(value=126)})
        theirs = _base()

        result = merge(base, ours, theirs)

        assert result.structural_status is StructuralStatus.CLEAN
        assert result.merged.params["lookback"].value == 126

    def test_merged_weights_breaking_normalization_are_invalid(self):
        """The normalized_map hazard: both edits reasonable, sum is not 1."""
        base = _base(params={"weight_a": Param(value=0.5), "weight_b": Param(value=0.5)})
        ours = _base(params={"weight_a": Param(value=0.8), "weight_b": Param(value=0.5)})
        theirs = _base(params={"weight_a": Param(value=0.5), "weight_b": Param(value=0.9)})

        result = merge(base, ours, theirs)

        assert result.economic_status is EconomicStatus.INVALID
        assert any("normalization" in n for n in result.notes)
        assert not result.publishable

    def test_merged_exclusions_emptying_universe_are_invalid(self):
        """The conjunction_set hazard: each exclusion fine, union selects nothing."""
        base = _base()
        ours = _base(excluded_assets=("SPY", "TLT"))
        theirs = _base(excluded_assets=("BIL",))

        result = merge(base, ours, theirs)

        assert result.economic_status is EconomicStatus.INVALID
        assert any("empty" in n for n in result.notes)

    def test_contract_change_breaks_comparability(self):
        base = _base()
        ours = _base(
            contract=OutputContract(universe=("SPY", "TLT", "BIL"), rebalance_frequency="21B")
        )
        theirs = _base()

        result = merge(base, ours, theirs)

        assert result.comparability_status is ComparabilityStatus.BROKEN
        assert any("re-run all published results" in r for r in result.required_retests)

    def test_divergent_contract_edits_violate(self):
        base = _base()
        ours = _base(contract=OutputContract(universe=("SPY",), rebalance_frequency="5B"))
        theirs = _base(contract=OutputContract(universe=("TLT",), rebalance_frequency="5B"))

        result = merge(base, ours, theirs)

        assert result.contract_status is ContractStatus.VIOLATED

    def test_merging_different_concepts_fails(self):
        result = merge(_base(), _base(), _base(concept="other"))
        assert result.structural_status is StructuralStatus.FAILED
        assert result.merged is None


class TestRegistry:
    def test_loads_shipped_methodologies(self):
        registry = MethodologyRegistry()
        concepts = registry.concepts()
        assert "hrp" in concepts
        assert concepts["hrp"] == sorted(concepts["hrp"])
        assert concepts["hrp"][0] == 1

    def test_latest_resolves_to_highest_version(self):
        registry = MethodologyRegistry()
        assert registry.get("hrp").version == max(registry.concepts()["hrp"])

    def test_pinning_resolves_exactly(self):
        assert MethodologyRegistry().get("hrp", 1).version == 1

    def test_resolve_accepts_version_id(self):
        m = MethodologyRegistry().resolve("methodology/hrp@1")
        assert m.version_id == "methodology/hrp@1"

    def test_unknown_version_raises_with_available_list(self):
        with pytest.raises(KeyError, match="available"):
            MethodologyRegistry().get("hrp", 99)

    def test_hrp_v2_breaks_comparability_with_v1(self):
        """The shipped example must actually demonstrate supersession."""
        registry = MethodologyRegistry()
        v1, v2 = registry.get("hrp", 1), registry.get("hrp", 2)
        breaks = v2.contract.breaks_compatibility_with(v1.contract)
        assert breaks, "hrp@2 should break comparability — it changes rebalance cadence"

    def test_shipped_methodologies_declare_limitations(self):
        """Derived, not solicited — but a published methodology with an empty
        limitations list is not publishable regardless."""
        for m in MethodologyRegistry().load_all():
            assert m.limitations, f"{m.version_id} declares no limitations"
            assert m.grounded_in, f"{m.version_id} cites no source"
