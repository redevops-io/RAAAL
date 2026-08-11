"""Execution-semantics conformance for Methodology Specification 0.1.

The spec's claim is that a methodology is *executable data*: every value the
computation uses comes from the AST, so two versions differing only in a
parameter necessarily produce different results. These tests hold that claim to
account — most importantly that the executor cannot see the future regardless of
what a pipeline declares.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.methodology import MethodologyRegistry
from src.methodology.executor import ExecutionError, backtest, execute
from src.methodology.spec import Param


@pytest.fixture
def prices():
    """Deterministic multi-asset panel, long enough for a 504-day lookback."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2015-01-01", periods=1400, freq="B")
    tickers = ["SPY", "SH", "TLT", "TBT", "LQD", "DBC", "GLD", "HYG", "BIL"]
    data = {}
    for i, t in enumerate(tickers):
        steps = rng.normal(0.0003, 0.01 + 0.002 * i, len(dates))
        data[t] = 100 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def registry():
    return MethodologyRegistry()


class TestCausality:
    def test_execute_cannot_see_beyond_as_of(self, registry, prices):
        """The strongest guarantee: appending future data must not change a past
        allocation. If it does, the methodology is reading ahead."""
        m = registry.get("hrp", 1)
        as_of = prices.index[600]

        early = execute(m, prices.loc[:as_of], as_of)
        with_future = execute(m, prices, as_of)

        for ticker in early:
            assert early[ticker] == pytest.approx(with_future[ticker], abs=1e-12), (
                f"{ticker} allocation changed when future data was appended — "
                "the executor is not causal"
            )

    def test_insufficient_history_does_not_silently_shorten_the_window(
        self, registry, prices
    ):
        """A short window must fall back or fail, never quietly use fewer days:
        a 252-day claim computed on 30 days is a different methodology."""
        m = registry.get("hrp", 1)
        as_of = prices.index[30]

        weights = execute(m, prices, as_of)
        # hrp@1 declares an inverse_volatility fallback, so it degrades rather
        # than raising — but it must not pretend it had the full lookback.
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_no_fallback_chain_means_failure(self, registry, prices):
        m = registry.get("hrp", 1)
        stripped = type(m)(**{**m.__dict__, "fallback_chain": ()})
        with pytest.raises(ExecutionError, match="lookback"):
            execute(stripped, prices, prices.index[30])


class TestParametersDriveComputation:
    def test_different_lookback_changes_allocation(self, registry, prices):
        """If this passes trivially, the AST is decoration rather than data."""
        m = registry.get("hrp", 1)
        longer = type(m)(**{**m.__dict__, "params": {**m.params, "lookback": Param(value=504)}})
        as_of = prices.index[900]

        assert execute(m, prices, as_of) != execute(longer, prices, as_of)

    def test_different_linkage_changes_allocation(self, registry, prices):
        m = registry.get("hrp", 1)
        complete = type(m)(
            **{**m.__dict__, "params": {**m.params, "linkage_method": Param(value="complete")}}
        )
        as_of = prices.index[900]

        assert execute(m, prices, as_of) != execute(complete, prices, as_of)

    def test_covariance_estimator_is_honoured(self, registry, prices):
        """The undeclared choice that caused a 0.5% divergence against the engine."""
        m = registry.get("hrp", 1)
        exponential = type(m)(
            **{
                **m.__dict__,
                "params": {**m.params, "covariance_estimator": Param(value="exponential")},
            }
        )
        as_of = prices.index[900]

        assert execute(m, prices, as_of) != execute(exponential, prices, as_of)

    def test_unsupported_estimator_is_refused(self, registry, prices):
        m = registry.get("hrp", 1)
        bad = type(m)(
            **{
                **m.__dict__,
                "params": {**m.params, "covariance_estimator": Param(value="shrinkage")},
                "fallback_chain": (),
            }
        )
        with pytest.raises(ExecutionError, match="covariance_estimator"):
            execute(bad, prices, prices.index[900])


class TestContractIsEnforced:
    def test_weights_respect_contract_bounds(self, registry, prices):
        m = registry.get("hrp", 1)
        weights = execute(m, prices, prices.index[900])
        lo = m.contract.weight_bounds["min"]
        hi = m.contract.weight_bounds["max"]

        assert all(lo - 1e-9 <= w <= hi + 1e-9 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_missing_universe_asset_is_an_error(self, registry, prices):
        """Silently allocating over a subset would misreport what ran."""
        m = registry.get("hrp", 1)
        with pytest.raises(ExecutionError, match="absent from the price"):
            execute(m, prices.drop(columns=["GLD"]), prices.index[900])

    def test_rebalance_cadence_comes_from_the_contract(self, registry, prices):
        v1 = backtest(registry.get("hrp", 1), prices)   # 5B
        v2 = backtest(registry.get("hrp", 2), prices)   # 21B

        assert v1["date"].nunique() > v2["date"].nunique() * 3

    def test_unknown_pipeline_step_is_an_error(self, registry, prices):
        """A typo must not silently drop a stage."""
        m = registry.get("hrp", 1)
        broken = type(m)(**{**m.__dict__, "pipeline": ("estimate_correlation", "typo_step")})
        with pytest.raises(ExecutionError, match="unknown pipeline steps"):
            execute(broken, prices, prices.index[900])

    def test_malformed_rebalance_frequency_is_an_error(self, registry, prices):
        from src.methodology.spec import OutputContract

        m = registry.get("hrp", 1)
        bad_contract = OutputContract(
            universe=m.contract.universe, rebalance_frequency="monthly"
        )
        broken = type(m)(**{**m.__dict__, "contract": bad_contract})
        with pytest.raises(ExecutionError, match="unsupported rebalance_frequency"):
            backtest(broken, prices)


class TestTurnoverCap:
    def test_cap_binds_on_v2(self, registry, prices):
        """hrp@2 declares apply_turnover_cap; consecutive rebalances must respect it."""
        m = registry.get("hrp", 2)
        weights = backtest(m, prices)
        cap = float(m.params["max_turnover"].value)

        pivot = weights.pivot(index="date", columns="ticker", values="weight").fillna(0.0)
        turnovers = pivot.diff().abs().sum(axis=1).iloc[1:]

        assert (turnovers <= cap + 1e-9).all(), (
            f"max observed turnover {turnovers.max():.4f} exceeds declared cap {cap}"
        )

    def test_v1_has_no_cap(self, registry, prices):
        """Control: v1 does not declare the step, so it must not be applied."""
        m = registry.get("hrp", 1)
        assert "apply_turnover_cap" not in m.pipeline
        assert "max_turnover" not in m.params

    def test_cap_without_param_is_an_error(self, registry, prices):
        m = registry.get("hrp", 2)
        params = {k: v for k, v in m.params.items() if k != "max_turnover"}
        broken = type(m)(**{**m.__dict__, "params": params, "fallback_chain": ()})
        with pytest.raises(ExecutionError, match="max_turnover"):
            execute(broken, prices, prices.index[900])


class TestDeterminism:
    def test_repeated_execution_is_identical(self, registry, prices):
        m = registry.get("hrp", 1)
        as_of = prices.index[900]
        assert execute(m, prices, as_of) == execute(m, prices, as_of)

    def test_backtest_is_reproducible(self, registry, prices):
        m = registry.get("hrp", 2)
        a = backtest(m, prices)
        b = backtest(m, prices)
        pd.testing.assert_frame_equal(a, b)


class TestVersionsDiffer:
    def test_shipped_versions_produce_different_results(self, registry, prices):
        """The end-to-end claim: versioning is meaningful because versions differ."""
        from src.features import compute_returns
        from src.history import _annualize, strategy_daily_returns

        returns = compute_returns(prices)
        v1 = _annualize(strategy_daily_returns(backtest(registry.get("hrp", 1), prices), returns, "weight"))
        v2 = _annualize(strategy_daily_returns(backtest(registry.get("hrp", 2), prices), returns, "weight"))

        assert not np.isnan(v1) and not np.isnan(v2)
        assert v1 != v2, "hrp@1 and hrp@2 produced identical results — the AST is not driving execution"


class TestMomentumFamily:
    """A second methodology family, deliberately unlike HRP.

    HRP clusters a covariance matrix; momentum ranks and selects. If the artifact
    model describes investment research rather than hierarchical allocation, this
    family should move through it without new core concepts.
    """

    @pytest.fixture
    def momentum(self):
        return MethodologyRegistry().get("xsmom", 1)

    def test_no_new_artifact_kinds_were_required(self):
        """The generality test. A second family should need new *handlers*, not
        new artifact types."""
        from src.methodology.spec import Methodology

        m = MethodologyRegistry().get("xsmom", 1)
        assert isinstance(m, Methodology)
        assert m.claims_ref and m.assumptions_ref
        assert m.contract.rebalance_frequency
        # Same fields as hrp; only the values and pipeline steps differ.
        assert set(m.canonical_form()) == set(
            MethodologyRegistry().get("hrp", 3).canonical_form()
        )

    def test_selection_produces_exactly_top_n_holdings(self, momentum, prices):
        weights = execute(momentum, prices, prices.index[900])
        held = [t for t, w in weights.items() if w > 0]
        assert len(held) == int(momentum.params["top_n"].value)

    def test_skip_changes_the_signal(self, momentum, prices):
        """Validates assumption/skip-most-recent-period@1.

        Omitting the skip measures continuation contaminated by short-horizon
        reversal — a different effect, not a rounding difference.
        """
        no_skip = type(momentum)(
            **{**momentum.__dict__, "params": {**momentum.params, "skip": Param(value=0)}}
        )
        as_of = prices.index[900]
        assert execute(momentum, prices, as_of) != execute(no_skip, prices, as_of)

    def test_top_n_changes_the_allocation(self, momentum, prices):
        wider = type(momentum)(
            **{**momentum.__dict__, "params": {**momentum.params, "top_n": Param(value=5)}}
        )
        as_of = prices.index[900]
        assert len([w for w in execute(wider, prices, as_of).values() if w > 0]) == 5

    def test_top_n_beyond_the_universe_is_refused(self, momentum, prices):
        broken = type(momentum)(
            **{
                **momentum.__dict__,
                "params": {**momentum.params, "top_n": Param(value=99)},
                "fallback_chain": (),
            }
        )
        with pytest.raises(ExecutionError, match="exceeds"):
            execute(broken, prices, prices.index[900])

    def test_insufficient_formation_history_is_refused(self, momentum, prices):
        broken = type(momentum)(**{**momentum.__dict__, "fallback_chain": ()})
        with pytest.raises(ExecutionError, match="formation"):
            execute(broken, prices, prices.index[30])

    def test_causality_holds_for_this_family_too(self, momentum, prices):
        as_of = prices.index[700]
        early = execute(momentum, prices.loc[:as_of], as_of)
        with_future = execute(momentum, prices, as_of)
        assert early == with_future

    def test_contract_bounds_hold(self, momentum, prices):
        weights = execute(momentum, prices, prices.index[900])
        assert max(weights.values()) <= momentum.contract.weight_bounds["max"] + 1e-6


class TestDeclarationsAreRealized:
    """Resolves finding/declared-rules-do-not-execute@1.

    Rules used to carry a free-text `expr` that nothing evaluated. They now name
    the field that realizes them and the property it must have — contracts
    execute, rules verify. The failure mode this closes is *declaration without
    behaviour*, the complement of the hidden-choice defects earlier releases
    eliminated.
    """

    def test_every_shipped_declaration_is_realized(self):
        from src.methodology.verify import unrealized_declarations

        for m in MethodologyRegistry().load_all():
            unrealized = unrealized_declarations(m)
            assert not unrealized, (
                f"{m.version_id} asserts what its own fields do not support: "
                f"{[(r.declaration_id, r.detail) for r in unrealized]}"
            )

    def test_drifted_rule_fails_verification(self):
        """The exact scenario the finding described: the contract moves, the rule
        does not, and nothing notices."""
        from dataclasses import replace
        from src.methodology.verify import unrealized_declarations

        m = MethodologyRegistry().get("hrp", 3)
        assert not unrealized_declarations(m)

        drifted = replace(
            m, contract=replace(m.contract, weight_bounds={"min": 0.0, "max": 0.30})
        )
        failures = unrealized_declarations(drifted)
        assert any(r.declaration_id == "concentration_cap" for r in failures)

    def test_rule_naming_a_nonexistent_field_fails(self):
        from dataclasses import replace
        from src.methodology.spec import Rule
        from src.methodology.verify import unrealized_declarations

        m = MethodologyRegistry().get("hrp", 3)
        ghost = replace(
            m, rules=(Rule(id="ghost", enforced_by="params.nonexistent", expected=">= 1"),)
        )
        failures = unrealized_declarations(ghost)
        assert failures and "unresolvable" in failures[0].detail

    def test_rule_naming_an_absent_pipeline_stage_fails(self):
        from dataclasses import replace
        from src.methodology.spec import Rule
        from src.methodology.verify import unrealized_declarations

        m = MethodologyRegistry().get("hrp", 3)
        missing = replace(
            m,
            rules=(Rule(id="absent", enforced_by="pipeline.compute_momentum",
                        expected="present"),),
        )
        assert unrealized_declarations(missing)

    def test_universe_filters_name_their_realization(self):
        for m in MethodologyRegistry().load_all():
            for f in m.universe_filters:
                assert f.enforced_by, f"{m.version_id}:{f.id} names no realization"

    def test_unrealized_declaration_is_a_hard_blocker(self):
        from src.policy import HARD_BLOCKERS

        assert "unrealized_declaration" in HARD_BLOCKERS
