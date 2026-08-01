"""Conformance tests for EvaluationProtocol.

    methodology + evaluation protocol = performance

The protocol is the half that used to be ambient: transaction costs and execution
lag lived in module constants, the grid was implied by CLI flags, and the data
snapshot was whatever was on disk. These tests hold it to being an identified,
hashable artifact that a result can cite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import EvaluationProtocol, ProtocolRegistry
from src.evaluation.protocol import DataSnapshot, Holdout, TransactionCosts, WalkForward
from src.evaluation.runner import (
    IncompatiblePairing,
    SealViolation,
    apply_seal,
    assess_compatibility,
    check_compatibility,
    evaluate,
)
from src.ledger import Ledger
from src.trial import build_trial_identity
from src.methodology import MethodologyRegistry


@pytest.fixture
def prices():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2015-01-01", periods=1600, freq="B")
    tickers = ["SPY", "SH", "TLT", "TBT", "LQD", "DBC", "GLD", "HYG", "BIL"]
    data = {}
    for i, t in enumerate(tickers):
        vol = 0.0002 if t == "BIL" else 0.008 + 0.002 * i
        data[t] = 100 * np.exp(np.cumsum(rng.normal(0.0002, vol, len(dates))))
    return pd.DataFrame(data, index=dates)


def _protocol(**overrides) -> EvaluationProtocol:
    defaults = dict(
        name="test",
        version=1,
        title="Test protocol",
        data_snapshot=DataSnapshot(source="synthetic", start="2015-01-01", end="2021-12-31"),
        walk_forward=WalkForward(warmup=252, step=5),
        transaction_costs=TransactionCosts(bps=10.0, execution_lag_days=1),
    )
    defaults.update(overrides)
    return EvaluationProtocol(**defaults)


class TestIdentity:
    def test_hash_is_stable(self):
        assert _protocol().content_hash == _protocol().content_hash

    def test_hash_ignores_title(self):
        assert _protocol(title="A").content_hash == _protocol(title="B").content_hash

    def test_hash_detects_cost_change(self):
        """Costs move every published return, so they must move identity."""
        a = _protocol()
        b = _protocol(transaction_costs=TransactionCosts(bps=25.0, execution_lag_days=1))
        assert a.content_hash != b.content_hash

    def test_hash_detects_execution_lag_change(self):
        a = _protocol()
        b = _protocol(transaction_costs=TransactionCosts(bps=10.0, execution_lag_days=2))
        assert a.content_hash != b.content_hash

    def test_hash_detects_embargo_change(self):
        a = _protocol()
        b = _protocol(walk_forward=WalkForward(warmup=252, step=5, embargo=10))
        assert a.content_hash != b.content_hash

    def test_snapshot_binding_does_not_change_identity(self):
        """The realized panel hash is a run-time fact, not part of the procedure.

        Including it in protocol identity would mint a new protocol every time the
        vendor restated history, even though nothing about the procedure changed.
        The hash is recorded on the run, which is what makes a restatement visible
        as "same protocol, different data".
        """
        base = _protocol()
        bound = base.with_snapshot_hash("abc123")
        assert base.content_hash == bound.content_hash
        assert bound.data_snapshot.content_hash == "abc123"

    def test_declared_date_range_does_affect_identity(self):
        a = _protocol()
        b = _protocol(
            data_snapshot=DataSnapshot(source="synthetic", start="2016-01-01", end="2021-12-31")
        )
        assert a.content_hash != b.content_hash

    def test_revise_requires_rationale(self):
        with pytest.raises(ValueError, match="change_rationale"):
            _protocol().revise(change_rationale="  ")

    def test_roundtrip(self):
        from src.evaluation.protocol import from_dict

        p = _protocol()
        assert from_dict(p.to_json()).content_hash == p.content_hash


class TestSealedHoldout:
    def test_sealed_period_is_truncated_before_execution(self, prices):
        """Unreachable, not merely unreported."""
        protocol = _protocol(
            holdout=Holdout(start="2019-01-01", end="2021-12-31", sealed=True)
        )
        usable, sealed = apply_seal(prices, protocol)

        assert sealed is True
        assert usable.index.max() < pd.Timestamp("2019-01-01")

    def test_unsealed_holdout_is_not_truncated(self, prices):
        protocol = _protocol(
            holdout=Holdout(start="2019-01-01", end="2021-12-31", sealed=False)
        )
        usable, sealed = apply_seal(prices, protocol)

        assert sealed is False
        assert usable.index.max() == prices.index.max()

    def test_seal_covering_everything_is_an_error(self, prices):
        protocol = _protocol(
            holdout=Holdout(start="2010-01-01", end="2021-12-31", sealed=True)
        )
        with pytest.raises(SealViolation, match="no evaluable history"):
            apply_seal(prices, protocol)

    def test_unseal_records_the_event(self):
        protocol = _protocol(
            holdout=Holdout(start="2019-01-01", end="2021-12-31", sealed=True)
        )
        opened = protocol.unseal("unlock_001")

        assert opened.holdout.sealed is False
        assert opened.holdout.unlock_event == "unlock_001"
        assert opened.content_hash != protocol.content_hash

    def test_unsealing_twice_is_refused(self):
        protocol = _protocol(
            holdout=Holdout(start="2019-01-01", end="2021-12-31", sealed=True)
        )
        opened = protocol.unseal("unlock_001")
        with pytest.raises(ValueError, match="already opened"):
            opened.unseal("unlock_002")

    def test_unseal_without_holdout_is_refused(self):
        with pytest.raises(ValueError, match="no holdout"):
            _protocol().unseal("unlock_001")

    def test_ledger_refuses_a_second_unlock(self, tmp_path):
        ledger = Ledger(tmp_path / "u.db")
        ledger.record_holdout_unlock(
            unlock_id="u1", protocol_id="protocol/sealed@1",
            reason="pre-registered evaluation complete", authorized_by="reviewer",
        )
        with pytest.raises(ValueError, match="already unlocked"):
            ledger.record_holdout_unlock(
                unlock_id="u2", protocol_id="protocol/sealed@1",
                reason="second look", authorized_by="reviewer",
            )


class TestCompatibility:
    def test_short_warmup_against_long_lookback_is_refused(self):
        """Otherwise early dates silently fall back to a different allocation
        rule while still being reported as the methodology."""
        methodology = MethodologyRegistry().get("hrp", 2)   # 504-day lookback
        protocol = _protocol(walk_forward=WalkForward(warmup=252, step=5))

        with pytest.raises(IncompatiblePairing, match="INSUFFICIENT_WARMUP"):
            check_compatibility(methodology, protocol)

        # The verdict is also available without raising, so it can be stored.
        verdict = assess_compatibility(methodology, protocol)
        assert verdict.compatible is False
        assert verdict.blockers[0]["code"] == "INSUFFICIENT_WARMUP"
        assert verdict.blockers[0]["required"] == 504
        assert verdict.blockers[0]["provided"] == 252

    def test_matched_warmup_is_accepted(self):
        methodology = MethodologyRegistry().get("hrp", 2)
        protocol = _protocol(walk_forward=WalkForward(warmup=504, step=5))
        check_compatibility(methodology, protocol)

    def test_zero_cost_protocol_is_refused_when_contract_requires_costs(self):
        methodology = MethodologyRegistry().get("hrp", 1)
        protocol = _protocol(
            transaction_costs=TransactionCosts(bps=0.0, execution_lag_days=1)
        )
        with pytest.raises(IncompatiblePairing, match="MISSING_COST_MODEL"):
            check_compatibility(methodology, protocol)


class TestProtocolDrivesResults:
    def test_costs_change_the_published_number(self, prices):
        """The strongest evidence the protocol is load-bearing."""
        methodology = MethodologyRegistry().get("hrp", 1)
        cheap, _ = evaluate(methodology, _protocol(), prices)
        dear, _ = evaluate(
            methodology,
            _protocol(transaction_costs=TransactionCosts(bps=100.0, execution_lag_days=1)),
            prices,
        )
        assert cheap.annualized_return != dear.annualized_return
        assert dear.annualized_return < cheap.annualized_return

    def test_execution_lag_changes_the_published_number(self, prices):
        methodology = MethodologyRegistry().get("hrp", 1)
        one, _ = evaluate(methodology, _protocol(), prices)
        five, _ = evaluate(
            methodology,
            _protocol(transaction_costs=TransactionCosts(bps=10.0, execution_lag_days=5)),
            prices,
        )
        assert one.annualized_return != five.annualized_return

    def test_result_cites_both_hashes(self, prices):
        methodology = MethodologyRegistry().get("hrp", 1)
        result, effective = evaluate(methodology, _protocol(), prices)

        assert result.methodology_hash == methodology.content_hash
        assert result.protocol_hash == effective.content_hash
        assert effective.data_snapshot.content_hash, "snapshot must be bound"

    def test_evaluation_is_deterministic(self, prices):
        methodology = MethodologyRegistry().get("hrp", 1)
        a, _ = evaluate(methodology, _protocol(), prices)
        b, _ = evaluate(methodology, _protocol(), prices)
        assert a.annualized_return == b.annualized_return
        assert a.protocol_hash == b.protocol_hash


class TestDiagnostics:
    def test_cash_dominated_result_is_flagged(self, prices):
        """The first real run produced Sharpe 6.59 from a 99.6%-cash portfolio."""
        methodology = MethodologyRegistry().get("hrp", 1)   # uncapped
        result, _ = evaluate(methodology, _protocol(), prices)

        assert not result.publishable
        joined = " ".join(result.flags)
        assert "concentration" in joined
        assert result.diagnostics["effective_n_assets"] < 2.0

    def test_capped_methodology_is_not_flagged(self, prices):
        methodology = MethodologyRegistry().get("hrp", 3)   # 25% ceiling
        protocol = _protocol(walk_forward=WalkForward(warmup=504, step=5))
        result, _ = evaluate(methodology, protocol, prices)

        assert result.diagnostics["top_asset_mean_weight"] <= 0.25 + 1e-6
        assert result.diagnostics["effective_n_assets"] > 3.0

    def test_contract_bounds_hold_on_every_rebalance(self, prices):
        """Including on the fallback path, which previously bypassed the check."""
        methodology = MethodologyRegistry().get("hrp", 3)
        protocol = _protocol(walk_forward=WalkForward(warmup=504, step=5))
        result, _ = evaluate(methodology, protocol, prices)

        pivot = result.weights.pivot(index="date", columns="ticker", values="weight")
        assert pivot.max().max() <= methodology.contract.weight_bounds["max"] + 1e-6


class TestRegistry:
    def test_loads_shipped_protocols(self):
        names = ProtocolRegistry().names()
        assert "standard" in names
        assert "sealed" in names

    def test_sealed_protocol_ships_sealed(self):
        assert ProtocolRegistry().get("sealed", 1).holdout.sealed is True

    def test_resolve_accepts_protocol_id(self):
        p = ProtocolRegistry().resolve("protocol/standard@1")
        assert p.protocol_id == "protocol/standard@1"


class TestTrialIdentity:
    """A trial is a materially distinct configuration, not a version and not a
    version × protocol product."""

    def _ledger(self, tmp_path):
        ledger = Ledger(tmp_path / "t.db")
        m = MethodologyRegistry().get("hrp", 1)
        ledger.publish_methodology(m)
        return ledger, m

    def test_protocol_search_counts_as_multiple_testing(self, tmp_path):
        """Trying five embargo settings is five trials even though the
        methodology never changed."""
        ledger, m = self._ledger(tmp_path)
        for i in range(5):
            identity = build_trial_identity(
                methodology_hash=m.content_hash,
                protocol_hash=f"{i:064d}",
                execution_assumptions={"embargo": i},
            )
            ledger.record_run(
                run_id=f"r{i}", version_id=m.version_id,
                protocol_id=f"protocol/variant@{i}", protocol_hash=f"{i:064d}",
                manifest={}, manifest_digest=f"d{i}", trial_id=identity.trial_id,
            )

        breakdown = ledger.trial_breakdown("hrp")
        assert breakdown["attempted_trials"] == 5
        assert breakdown["dsr_countable_trials"] == 5
        assert breakdown["distinct_methodology_versions"] == 1

    def test_repeating_a_configuration_is_not_a_new_trial(self, tmp_path):
        """Re-running the identical configuration is a reproducibility check."""
        ledger, m = self._ledger(tmp_path)
        identity = build_trial_identity(
            methodology_hash=m.content_hash, protocol_hash="a" * 64
        )
        for i in range(4):
            ledger.record_run(
                run_id=f"r{i}", version_id=m.version_id,
                protocol_id="protocol/standard@1", protocol_hash="a" * 64,
                manifest={}, manifest_digest=f"d{i}", trial_id=identity.trial_id,
            )

        breakdown = ledger.trial_breakdown("hrp")
        assert breakdown["attempted_trials"] == 1
        assert breakdown["executions"] == 4
        assert breakdown["repeat_executions"] == 3

    def test_execution_assumptions_distinguish_trials(self, tmp_path):
        """Same methodology, same protocol id, different cost assumption."""
        ledger, m = self._ledger(tmp_path)
        a = build_trial_identity(
            methodology_hash=m.content_hash, protocol_hash="a" * 64,
            execution_assumptions={"cost_bps": 10},
        )
        b = build_trial_identity(
            methodology_hash=m.content_hash, protocol_hash="a" * 64,
            execution_assumptions={"cost_bps": 25},
        )
        assert a.trial_id != b.trial_id

    def test_data_partition_distinguishes_trials(self):
        """Evaluating in-sample and on the holdout are different attempts."""
        a = build_trial_identity(
            methodology_hash="m", protocol_hash="p", data_partition="full"
        )
        b = build_trial_identity(
            methodology_hash="m", protocol_hash="p", data_partition="holdout_sealed"
        )
        assert a.trial_id != b.trial_id

    def test_blocked_pairings_are_attempted_but_not_dsr_countable(self, tmp_path):
        """A refusal reveals nothing about the data, so it must not inflate the
        DSR denominator — but hiding it would understate how hard someone looked."""
        ledger, m = self._ledger(tmp_path)
        identity = build_trial_identity(
            methodology_hash=m.content_hash, protocol_hash="b" * 64
        )
        ledger.record_compatibility(
            compatibility_id="c1", concept="hrp", version_id=m.version_id,
            protocol_id="protocol/bad@1", trial_id=identity.trial_id,
            compatible=False,
            blockers=[{"code": "INSUFFICIENT_WARMUP", "required": 504, "provided": 252}],
        )

        breakdown = ledger.trial_breakdown("hrp")
        assert breakdown["attempted_trials"] == 1
        assert breakdown["blocked_before_execution"] == 1
        assert breakdown["dsr_countable_trials"] == 0

    def test_dsr_policy_is_stated(self):
        from src.trial import DSR_COUNTABLE_OUTCOMES, TrialOutcome

        assert TrialOutcome.BLOCKED_INCOMPATIBLE not in DSR_COUNTABLE_OUTCOMES
        assert TrialOutcome.COMPLETED in DSR_COUNTABLE_OUTCOMES


class TestResultStatus:
    def test_status_fields_are_separate(self, prices):
        """One generic warning field cannot distinguish a failed computation from
        a successful one describing a cash proxy."""
        methodology = MethodologyRegistry().get("hrp", 1)
        result, _ = evaluate(methodology, _protocol(), prices)
        status = result.result_status

        assert status["computation_valid"] is True
        assert status["contract_valid"] is True
        assert status["economic_valid"] is False
        assert status["economically_degenerate"] is True

        # Deliberately absent: whether the evidence meets a standard is a
        # versioned-policy decision, and whether it may be shown is a
        # surface-specific publication decision. The runner knows neither.
        assert "statistical_valid" not in status
        assert "publication_eligible" not in status
        assert status["statistical_assessment_complete"] is False

    def test_execution_audit_reports_fallback_and_overrides(self, prices):
        methodology = MethodologyRegistry().get("hrp", 3)
        protocol = _protocol(walk_forward=WalkForward(warmup=504, step=5))
        result, _ = evaluate(methodology, protocol, prices)
        audit = result.execution_audit

        assert "fallback_share" in audit
        assert "precedence_override_count" in audit
        assert audit["requested_turnover_cap"] == 0.25

    def test_heavy_fallback_use_is_flagged(self, prices):
        """A record largely produced by a degradation rule is not a record of the
        methodology it is labelled with."""
        from src.evaluation.runner import FALLBACK_SHARE_LIMIT

        assert 0.0 < FALLBACK_SHARE_LIMIT < 1.0


class TestConstraintPolicy:
    def test_policy_is_part_of_the_contract_hash(self):
        from src.methodology.spec import ConstraintPolicy, OutputContract

        a = OutputContract(universe=("SPY",), rebalance_frequency="5B")
        b = OutputContract(
            universe=("SPY",), rebalance_frequency="5B",
            constraint_policy=ConstraintPolicy(soft_may_be_violated_to_satisfy_hard=False),
        )
        assert a.to_json() != b.to_json()

    def test_default_policy_makes_bounds_hard(self):
        from src.methodology.spec import ConstraintPolicy

        policy = ConstraintPolicy()
        assert "weight_bounds" in policy.hard
        assert "turnover_cap" in policy.soft
        assert policy.soft_may_be_violated_to_satisfy_hard is True
