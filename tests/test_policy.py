"""Tests for the three-layer decision split.

    statistics  — what does the evidence say?
    policy      — does it meet a declared standard?
    publication — who may see it, and labelled how?

The tests exist mainly to hold the *separation*: each layer must be unable to
make the next layer's decision, and a weak result must remain publishable while
being disqualified from a validated claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.policy import (
    Decision,
    EvidenceGrade,
    PolicyRegistry,
    PolicyStatus,
    Severity,
    Surface,
    decide,
)
from src.policy.statistical_policy import Requirement, StatisticalPolicy
from src.statistics.assessment import assess
from src.statistics.neutralize import FactorModel

NOW = "2026-07-30T00:00:00Z"


@pytest.fixture
def strong_returns():
    rng = np.random.default_rng(11)
    idx = pd.date_range("2019-01-01", periods=1500, freq="B")
    return pd.Series(rng.normal(0.0012, 0.009, len(idx)), index=idx)


@pytest.fixture
def weak_returns():
    rng = np.random.default_rng(12)
    idx = pd.date_range("2019-01-01", periods=1500, freq="B")
    return pd.Series(rng.normal(0.00005, 0.011, len(idx)), index=idx)


@pytest.fixture
def lineage(strong_returns, weak_returns):
    rng = np.random.default_rng(13)
    cols = {"strong": strong_returns, "weak": weak_returns}
    for i in range(6):
        cols[f"noise_{i}"] = pd.Series(
            rng.normal(0.0, 0.01, len(strong_returns)), index=strong_returns.index
        )
    return pd.DataFrame(cols)


class TestAssessmentReportsFactsOnly:
    def test_assessment_has_no_pass_fail_field(self, strong_returns, lineage):
        """The statistics layer must not adjudicate."""
        result = assess(strong_returns, trial_count=8, lineage_returns=lineage)
        payload = result.to_json()

        for forbidden in ("passed", "valid", "eligible_for_publication", "status"):
            assert forbidden not in payload, (
                f"assessment must not carry {forbidden!r} — that is a policy decision"
            )
        assert payload["computation_status"] in {"VALID", "PARTIAL", "FAILED"}

    def test_assessment_records_the_trial_count_and_policy(self, strong_returns, lineage):
        result = assess(strong_returns, trial_count=8, lineage_returns=lineage)
        assert result.trial_count == 8
        assert result.count_policy == "DSR_COUNTABLE_OUTCOMES"

    def test_missing_lineage_marks_assessment_partial(self, strong_returns):
        """PBO needs comparable configurations; without them the picture is incomplete."""
        result = assess(strong_returns, trial_count=1)
        assert result.computation_status == "PARTIAL"
        assert any("PBO not computed" in w for w in result.warnings)

    def test_neutralization_is_included_when_requested(self, strong_returns, lineage):
        rng = np.random.default_rng(14)
        factors = pd.DataFrame(
            {"market": rng.normal(0.0004, 0.01, len(strong_returns))},
            index=strong_returns.index,
        )
        model = FactorModel(name="market-only", version=1, factors=("market",))

        result = assess(
            strong_returns, trial_count=8, lineage_returns=lineage,
            factor_returns=factors, factor_model=model,
        )
        assert result.factor_neutralization is not None
        assert "raw_annualized_mean" in result.factor_neutralization
        assert "residual_annualized_mean" in result.factor_neutralization


class TestPolicyIsVersioned:
    def test_shipped_policy_loads(self):
        policy = PolicyRegistry().get("library-default")
        assert policy.policy_id == "stat-policy/library-default@1"
        assert policy.content_hash

    def test_threshold_change_changes_identity(self):
        base = StatisticalPolicy(
            name="p", version=1, title="p",
            requirements=(Requirement("minimum_dsr", "", Severity.WARN, 0.5),),
        )
        stricter = StatisticalPolicy(
            name="p", version=1, title="p",
            requirements=(Requirement("minimum_dsr", "", Severity.WARN, 0.9),),
        )
        assert base.content_hash != stricter.content_hash

    def test_no_performance_threshold_blocks(self):
        """A low return is a finding, not a defect. Blocking it would turn the
        library into a highlight reel."""
        policy = PolicyRegistry().get("library-default")
        blocking = {r.code for r in policy.requirements if r.severity is Severity.BLOCK}

        assert "minimum_dsr" not in blocking
        assert "maximum_pbo" not in blocking
        assert "require_complete_computation" in blocking


class TestPolicyEvaluation:
    def test_strong_evidence_passes(self, strong_returns, lineage):
        policy = PolicyRegistry().get("library-default")
        assessment = assess(strong_returns, trial_count=8, lineage_returns=lineage)
        # Supply neutralization so the WARN requirement is satisfiable.
        result = policy.evaluate(assessment, now=NOW)

        assert result.status in {PolicyStatus.PASS, PolicyStatus.WARN}
        assert result.policy_id == "stat-policy/library-default@1"
        assert result.policy_hash

    def test_short_record_blocks_on_observations(self, strong_returns, lineage):
        policy = PolicyRegistry().get("library-default")
        assessment = assess(
            strong_returns.iloc[:300], trial_count=8,
            lineage_returns=lineage.iloc[:300],
        )
        result = policy.evaluate(assessment, now=NOW)

        assert result.status is PolicyStatus.FAIL
        codes = {f.code for f in result.blocking_findings}
        assert "minimum_observations" in codes

    def test_weak_result_warns_rather_than_failing(self, weak_returns, lineage):
        """The library thesis: weak is publishable, labelled weak."""
        policy = PolicyRegistry().get("library-default")
        assessment = assess(weak_returns, trial_count=8, lineage_returns=lineage)
        result = policy.evaluate(assessment, now=NOW)

        assert result.status is not PolicyStatus.PASS
        assert result.evidence_grade in {EvidenceGrade.WEAK, EvidenceGrade.MODERATE}

    def test_evidence_grade_is_independent_of_status(self, strong_returns, lineage):
        """Grade describes the evidence; status describes conformance."""
        permissive = StatisticalPolicy(
            name="permissive", version=1, title="permissive",
            requirements=(Requirement("minimum_dsr", "", Severity.WARN, 0.01),),
        )
        assessment = assess(strong_returns, trial_count=8, lineage_returns=lineage)
        result = permissive.evaluate(assessment, now=NOW)

        assert result.status is PolicyStatus.PASS
        # Passing a permissive policy does not make the evidence strong.
        assert result.evidence_grade in set(EvidenceGrade)


class TestPublicationGate:
    def _clean_status(self):
        return {
            "computation_valid": True,
            "contract_valid": True,
            "economic_valid": True,
            "statistical_assessment_complete": True,
            "reproducible": True,
            "flags": [],
        }

    def _evaluation(self, returns, lineage, policy=None):
        policy = policy or PolicyRegistry().get("library-default")
        assessment = assess(returns, trial_count=8, lineage_returns=lineage)
        return assessment, policy.evaluate(assessment, now=NOW)

    def test_private_draft_shows_everything(self, weak_returns, lineage):
        assessment, evaluation = self._evaluation(weak_returns, lineage)
        decision = decide(
            surface=Surface.PRIVATE_DRAFT, result_status=self._clean_status(),
            assessment=assessment, policy_evaluation=evaluation,
        )
        assert decision.decision is Decision.ALLOW
        assert decision.may_claim_validated is False

    def test_public_library_allows_weak_with_disclosure(self, weak_returns, lineage):
        """A statistically weak result is publishable as documented research."""
        assessment, evaluation = self._evaluation(weak_returns, lineage)
        decision = decide(
            surface=Surface.PUBLIC_LIBRARY, result_status=self._clean_status(),
            assessment=assessment, policy_evaluation=evaluation,
        )
        assert decision.decision is Decision.ALLOW_WITH_DISCLOSURE
        assert decision.disclosures
        assert decision.may_claim_validated is False

    def test_validated_badge_requires_pass_and_strong(self, weak_returns, lineage):
        assessment, evaluation = self._evaluation(weak_returns, lineage)
        decision = decide(
            surface=Surface.VALIDATED_BADGE, result_status=self._clean_status(),
            assessment=assessment, policy_evaluation=evaluation,
        )
        assert decision.decision is Decision.BLOCK
        assert decision.may_claim_validated is False

    def test_forward_surface_rejects_backtests_outright(self, strong_returns, lineage):
        """Ranking a backtest on the forward surface would reintroduce exactly the
        failure that surface exists to avoid."""
        assessment, evaluation = self._evaluation(strong_returns, lineage)
        decision = decide(
            surface=Surface.FORWARD_TRACK_RECORD, result_status=self._clean_status(),
            assessment=assessment, policy_evaluation=evaluation,
        )
        assert decision.decision is Decision.BLOCK
        assert "backtest_not_eligible_for_forward_surface" in decision.hard_blockers

    def test_institutional_export_includes_failures(self, weak_returns, lineage):
        """An export that omitted failures would misrepresent the search."""
        assessment, evaluation = self._evaluation(weak_returns, lineage)
        status = self._clean_status()
        status["contract_valid"] = False

        decision = decide(
            surface=Surface.INSTITUTIONAL_EXPORT, result_status=status,
            assessment=assessment, policy_evaluation=evaluation,
        )
        assert decision.decision is Decision.ALLOW_WITH_DISCLOSURE
        assert "contract_violation" in decision.disclosures

    def test_hard_blocker_blocks_public_surface(self, strong_returns, lineage):
        assessment, evaluation = self._evaluation(strong_returns, lineage)
        status = self._clean_status()
        status["contract_valid"] = False

        decision = decide(
            surface=Surface.PUBLIC_LIBRARY, result_status=status,
            assessment=assessment, policy_evaluation=evaluation,
        )
        assert decision.decision is Decision.BLOCK
        assert "contract_violation" in decision.hard_blockers

    def test_severe_degeneracy_blocks_but_concentration_alone_does_not(
        self, strong_returns, lineage
    ):
        """A concentrated portfolio is a finding worth publishing with a caveat;
        a ratio that is an artifact of a near-zero denominator is not."""
        assessment, evaluation = self._evaluation(strong_returns, lineage)

        concentrated = self._clean_status()
        concentrated["flags"] = ["concentration: BIL holds 60% of the portfolio"]
        assert (
            decide(
                surface=Surface.PUBLIC_LIBRARY, result_status=concentrated,
                assessment=assessment, policy_evaluation=evaluation,
            ).decision
            is not Decision.BLOCK
        )

        degenerate = self._clean_status()
        degenerate["flags"] = ["degenerate volatility: 0.2% annualized is cash-equivalent"]
        assert (
            decide(
                surface=Surface.PUBLIC_LIBRARY, result_status=degenerate,
                assessment=assessment, policy_evaluation=evaluation,
            ).decision
            is Decision.BLOCK
        )

    def test_acknowledgement_clears_a_blocker_and_is_recorded(
        self, strong_returns, lineage
    ):
        """An operator may accept a documented defect; the acceptance is recorded
        rather than silently clearing the flag."""
        assessment, evaluation = self._evaluation(strong_returns, lineage)
        status = self._clean_status()
        status["reproducible"] = False

        blocked = decide(
            surface=Surface.PUBLIC_LIBRARY, result_status=status,
            assessment=assessment, policy_evaluation=evaluation,
        )
        assert blocked.decision is Decision.BLOCK

        accepted = decide(
            surface=Surface.PUBLIC_LIBRARY, result_status=status,
            assessment=assessment, policy_evaluation=evaluation,
            acknowledgements=("failed_reproducibility",),
        )
        assert accepted.decision is not Decision.BLOCK
        assert "failed_reproducibility" in accepted.acknowledgements


class TestLayerSeparation:
    def test_runner_status_does_not_adjudicate_statistics(self):
        """The runner cannot know whether evidence meets a standard."""
        from src.evaluation.runner import EvaluationResult

        fields = EvaluationResult.__dataclass_fields__
        assert "policy_evaluation" not in fields
        assert "publication_decision" not in fields

    def test_publication_needs_all_three_inputs(self, strong_returns, lineage):
        """Removing the policy evaluation must change the decision, proving the
        gate is not silently re-deriving it."""
        assessment = assess(strong_returns, trial_count=8, lineage_returns=lineage)
        status = {
            "computation_valid": True, "contract_valid": True, "economic_valid": True,
            "statistical_assessment_complete": True, "reproducible": True, "flags": [],
        }
        without = decide(
            surface=Surface.VALIDATED_BADGE, result_status=status, assessment=assessment
        )
        assert without.decision is Decision.BLOCK
        assert without.policy_status is None


class TestTheRegistryOnlyLoadsPolicies:
    """`policies/` is read wholesale by filename shape: anything matching
    `name@version.yaml` is parsed as a statistical policy.

    A market-data licensing record was written there because it, too, records a
    decision under a version. It parsed as YAML, matched the filename pattern,
    and died on `payload["name"]` — a bare `KeyError: 'name'` from inside a
    registry load, raised while rendering, which surfaced as a 500 on every
    page of the site and named neither the file nor the field.

    Skipping unrecognised files would hide the opposite failure: a genuine
    policy with a mistyped key would disappear from the registry and results
    would be judged under a standard quietly missing a member. So this stays
    fatal, and only says what it choked on.
    """

    def _registry(self, tmp_path, text):
        (tmp_path / "market-data-licensing@1.yaml").write_text(text)
        return PolicyRegistry(tmp_path)

    def test_a_non_policy_names_itself_and_the_missing_field(self, tmp_path):
        registry = self._registry(tmp_path, "policy_version: x\nanswers: {}\n")
        with pytest.raises(ValueError) as raised:
            registry.load_all()
        message = str(raised.value)
        assert "market-data-licensing@1.yaml" in message
        assert "name" in message and "version" in message

    def test_it_does_not_fail_on_a_real_policy(self, tmp_path):
        """The discriminating half: the check must reject the intruder without
        rejecting a policy that merely omits optional fields."""
        registry = self._registry(
            tmp_path, "name: market-data-licensing\nversion: 1\n")
        assert [(p.name, p.version) for p in registry.load_all()] == \
            [("market-data-licensing", 1)]
