"""Statistical estimator conformance, proved on a synthetic lineage.

The live HRP lineage is far too shallow to demonstrate DSR meaningfully — with
one or two trials the correction is nearly inert. So the proof is built on a
controlled lineage where ground truth is known by construction:

* many variants attempted, only a few attractive by raw Sharpe;
* trial count fully known;
* returns deliberately non-normal;
* several variants highly correlated with each other;
* exactly one genuinely stronger signal;
* the best raw Sharpe is *not* the best DSR.

That last property is the point. If the estimators cannot distinguish a lucky
maximum from a real edge on data where we know which is which, they cannot be
trusted on data where we do not.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.statistics import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    purged_walk_forward_splits,
    sharpe_ratio,
)
from src.statistics.neutralize import FactorModel, neutralize_returns
from src.statistics.purged_cv import leakage_score

SEED = 20260730
N_DAYS = 1260          # five years of daily observations
N_NOISE_VARIANTS = 40  # attempted configurations that are pure noise


@pytest.fixture(scope="module")
def synthetic_lineage():
    """A lineage with known ground truth.

    Returns (frame, truth) where `truth` names the one variant with real edge.
    """
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2019-01-01", periods=N_DAYS, freq="B")

    columns = {}

    # 40 pure-noise variants. Zero expected return; any Sharpe they show is luck.
    for i in range(N_NOISE_VARIANTS):
        columns[f"noise_{i:02d}"] = rng.normal(0.0, 0.01, N_DAYS)

    # A correlated cluster: the same idea reparameterized. Searching over these
    # is not independent search, and treating them as distinct trials overstates
    # breadth — which is exactly why trial identity is configuration-based.
    base = rng.normal(0.0, 0.01, N_DAYS)
    for i in range(5):
        columns[f"corr_{i}"] = base + rng.normal(0.0, 0.002, N_DAYS)

    # One genuine edge: strong enough to clear the maximum a 47-way search
    # produces from noise alone. Sharpe ~1.6 annualized.
    columns["real_edge"] = rng.normal(0.0013, 0.009, N_DAYS)

    # A deliberately non-normal variant engineered to a *higher* raw Sharpe than
    # the real edge, built from steady small gains punctuated by rare large
    # losses. This is the shape that flatters raw Sharpe, and separating it from
    # the real edge requires the moment corrections rather than the trial count.
    steady = rng.normal(0.0016, 0.005, N_DAYS)
    shock_idx = rng.choice(N_DAYS, size=8, replace=False)
    steady[shock_idx] -= rng.uniform(0.04, 0.07, size=8)
    columns["fat_tailed"] = steady

    frame = pd.DataFrame(columns, index=dates)
    return frame, "real_edge"


class TestSharpeAndPSR:
    def test_psr_rises_with_sample_length(self, synthetic_lineage):
        """The same Sharpe is stronger evidence over a longer record."""
        frame, truth = synthetic_lineage
        series = frame[truth]

        short = probabilistic_sharpe_ratio(series.iloc[:252])
        long = probabilistic_sharpe_ratio(series)

        assert long.value > short.value
        assert long.inputs.observations > short.inputs.observations

    def test_psr_penalizes_negative_skew_and_fat_tails(self, synthetic_lineage):
        """Two series with similar raw Sharpe are not equally credible."""
        frame, _ = synthetic_lineage
        fat = frame["fat_tailed"]

        result = probabilistic_sharpe_ratio(fat)

        assert result.inputs.skewness < 0, "fixture should be negatively skewed"
        assert result.inputs.kurtosis > 3, "fixture should be leptokurtic"

        # Against a clean series engineered to the same observed Sharpe, the
        # fat-tailed one must score lower.
        rng = np.random.default_rng(1)
        clean_vals = rng.normal(fat.mean(), fat.std(), len(fat))
        clean = pd.Series(clean_vals, index=fat.index)
        clean = (clean - clean.mean()) / clean.std() * fat.std() + fat.mean()

        clean_result = probabilistic_sharpe_ratio(clean)
        assert abs(sharpe_ratio(clean) - sharpe_ratio(fat)) < 0.05
        assert result.value < clean_result.value

    def test_psr_reports_zero_variance_as_ineligible(self):
        flat = pd.Series([0.0] * 300)
        result = probabilistic_sharpe_ratio(flat)
        assert result.eligible is False
        assert "zero return variance" in " ".join(result.warnings)


class TestDeflatedSharpe:
    def test_expected_max_sharpe_grows_with_trials(self):
        """With more attempts, a higher maximum arises from luck alone."""
        assert expected_max_sharpe(1, 1.0) == 0.0
        assert expected_max_sharpe(10, 1.0) < expected_max_sharpe(100, 1.0)
        assert expected_max_sharpe(100, 1.0) < expected_max_sharpe(1000, 1.0)

    def test_dsr_declines_as_search_depth_increases(self, synthetic_lineage):
        """The central claim: the same track record is weaker evidence when more
        configurations were attempted to find it."""
        frame, truth = synthetic_lineage
        series = frame[truth]
        variance = float(np.var([sharpe_ratio(frame[c]) for c in frame.columns], ddof=1))

        values = [
            deflated_sharpe_ratio(series, trials_observed=n, variance_of_sharpes=variance).value
            for n in (1, 10, 50, 200)
        ]
        assert values == sorted(values, reverse=True), (
            f"DSR must be non-increasing in trial count, got {values}"
        )

    def test_single_trial_warns_that_no_deflation_applied(self, synthetic_lineage):
        frame, truth = synthetic_lineage
        result = deflated_sharpe_ratio(frame[truth], trials_observed=1)
        assert "no deflation" in " ".join(result.warnings)

    def test_luckiest_noise_variant_fails_the_deflated_bar(self, synthetic_lineage):
        """The proof the module exists for.

        Among 40 pure-noise configurations, the luckiest shows a respectable raw
        Sharpe. Undeflated it looks like a finding. Deflated by the true trial
        count it is exposed as the maximum of a search, which is what it is.

        Note DSR cannot *reorder* variants by trial count alone — with a common N
        it is monotone in Sharpe. What it changes is the bar, and the bar is the
        whole point.
        """
        frame, _ = synthetic_lineage
        noise = frame[[c for c in frame.columns if c.startswith("noise_")]]
        sharpes = {c: sharpe_ratio(noise[c]) for c in noise.columns}
        variance = float(np.var(list(sharpes.values()), ddof=1))
        luckiest = max(sharpes, key=sharpes.get)

        assert sharpes[luckiest] > 0.9, "fixture should produce a flattering winner"

        undeflated = deflated_sharpe_ratio(
            noise[luckiest], trials_observed=1, variance_of_sharpes=variance
        )
        deflated = deflated_sharpe_ratio(
            noise[luckiest], trials_observed=noise.shape[1], variance_of_sharpes=variance
        )

        assert undeflated.value > 0.95, "undeflated, the lucky winner looks significant"
        assert deflated.value < 0.95, (
            f"deflated by {noise.shape[1]} trials it must fail the bar, "
            f"got {deflated.value:.4f}"
        )

    def test_real_edge_survives_deflation_far_better_than_luck(self, synthetic_lineage):
        """A genuine edge must not be destroyed by an honest trial count.

        Compared against the luckiest noise variant under *identical* deflation,
        rather than against an absolute threshold — where the bar sits is a
        publication-policy decision, not a property of the estimator, and baking
        one into the estimator's test would confuse the two.
        """
        frame, truth = synthetic_lineage
        sharpes = {c: sharpe_ratio(frame[c]) for c in frame.columns}
        variance = float(np.var(list(sharpes.values()), ddof=1))
        n_trials = frame.shape[1]

        noise_cols = [c for c in frame.columns if c.startswith("noise_")]
        luckiest = max(noise_cols, key=lambda c: sharpes[c])

        real = deflated_sharpe_ratio(
            frame[truth], trials_observed=n_trials, variance_of_sharpes=variance
        )
        lucky = deflated_sharpe_ratio(
            frame[luckiest], trials_observed=n_trials, variance_of_sharpes=variance
        )

        assert real.value > 0.75, f"real edge should survive, got {real.value:.4f}"
        assert lucky.value < 0.25, f"lucky noise should not, got {lucky.value:.4f}"
        assert real.value - lucky.value > 0.5, (
            "the estimator must separate a genuine edge from the maximum of a "
            f"search: {real.value:.4f} vs {lucky.value:.4f}"
        )

    def test_equal_raw_sharpe_is_not_equal_evidence(self, synthetic_lineage):
        """Where the moment corrections bite.

        Two series with the *same* raw Sharpe are not equally credible: steady
        gains punctuated by rare large losses is a shape that flatters the ratio.
        Holding Sharpe fixed isolates the skewness and kurtosis penalty from every
        other effect, which is a more robust demonstration than tuning two series
        to a near-tie and hoping the penalty happens to exceed the gap.
        """
        frame, _ = synthetic_lineage
        fat = frame["fat_tailed"]

        # A Gaussian series matched to the same mean and standard deviation, and
        # therefore to the same raw Sharpe, but with clean moments.
        rng = np.random.default_rng(4242)
        draws = rng.normal(0.0, 1.0, len(fat))
        clean = pd.Series(
            (draws - draws.mean()) / draws.std() * fat.std() + fat.mean(),
            index=fat.index,
        )

        assert sharpe_ratio(clean) == pytest.approx(sharpe_ratio(fat), abs=0.01)

        dsr_fat = deflated_sharpe_ratio(fat, trials_observed=47, variance_of_sharpes=1.0)
        dsr_clean = deflated_sharpe_ratio(clean, trials_observed=47, variance_of_sharpes=1.0)

        assert dsr_fat.inputs.skewness < -1.0, "fixture should be strongly left-skewed"
        assert dsr_fat.inputs.kurtosis > 10.0, "fixture should be strongly leptokurtic"
        assert dsr_fat.value < dsr_clean.value, (
            "identical raw Sharpe must not yield identical evidence: "
            f"fat-tailed {dsr_fat.value:.4f} vs clean {dsr_clean.value:.4f}"
        )

    def test_dsr_records_the_trial_count_it_used(self, synthetic_lineage):
        """A DSR without its N is not interpretable."""
        frame, truth = synthetic_lineage
        result = deflated_sharpe_ratio(frame[truth], trials_observed=47)
        assert result.inputs.trials_observed == 47
        assert result.estimator_version
        assert any("platform-observed" in a for a in result.assumptions)


class TestMinimumTrackRecord:
    def test_required_length_falls_as_sharpe_rises(self, synthetic_lineage):
        frame, truth = synthetic_lineage
        strong = frame[truth]
        weak = frame[truth] * 0.3 + frame["noise_00"] * 0.7

        assert (
            minimum_track_record_length(strong).value
            < minimum_track_record_length(weak).value
        )

    def test_sharpe_below_benchmark_is_never_sufficient(self, synthetic_lineage):
        frame, _ = synthetic_lineage
        result = minimum_track_record_length(frame["noise_00"], benchmark_sharpe=2.0)
        assert result.value == float("inf")
        assert result.eligible is False


class TestPBO:
    def test_pbo_is_high_when_selecting_among_noise(self, synthetic_lineage):
        """Choosing the in-sample best among pure noise carries no out-of-sample
        information, so the winner lands below median about half the time."""
        frame, _ = synthetic_lineage
        noise_only = frame[[c for c in frame.columns if c.startswith("noise_")]]

        result = probability_of_backtest_overfitting(noise_only, n_splits=8)

        assert result.eligible
        assert result.value > 0.35, f"expected substantial PBO on noise, got {result.value}"

    def test_pbo_is_lower_with_a_real_edge_present(self, synthetic_lineage):
        """When one configuration genuinely dominates, in-sample selection starts
        to transfer out of sample."""
        frame, truth = synthetic_lineage
        noise_only = frame[[c for c in frame.columns if c.startswith("noise_")]]
        with_edge = frame[[truth] + [c for c in noise_only.columns]]

        pbo_noise = probability_of_backtest_overfitting(noise_only, n_splits=8).value
        pbo_edge = probability_of_backtest_overfitting(with_edge, n_splits=8).value

        assert pbo_edge < pbo_noise

    def test_single_configuration_is_ineligible(self, synthetic_lineage):
        frame, truth = synthetic_lineage
        result = probability_of_backtest_overfitting(frame[[truth]])
        assert result.eligible is False


class TestPurgedCV:
    def test_no_observation_appears_in_train_and_test(self):
        index = pd.date_range("2020-01-01", periods=1000, freq="B")
        for split in purged_walk_forward_splits(index, n_splits=5, purge=10, embargo=10):
            assert leakage_score(split.train_index, split.test_index) == 0

    def test_training_always_precedes_testing(self):
        index = pd.date_range("2020-01-01", periods=1000, freq="B")
        for split in purged_walk_forward_splits(index, n_splits=5):
            if len(split.train_index):
                assert split.train_index.max() < split.test_index.min()

    def test_purge_removes_the_boundary_observations(self):
        """A larger purge must shrink the training set."""
        index = pd.date_range("2020-01-01", periods=1000, freq="B")
        none = purged_walk_forward_splits(index, n_splits=5, purge=0)
        heavy = purged_walk_forward_splits(index, n_splits=5, purge=50)

        assert sum(len(s.train_index) for s in heavy) < sum(
            len(s.train_index) for s in none
        )

    def test_purge_removes_known_label_overlap(self):
        """Construct explicit overlap and verify purging eliminates it.

        A label formed at position *t* using a horizon of *h* is still resolving
        at *t+h*. Counting training positions whose label window reaches into the
        test block gives a direct, countable measure of leakage — which must be
        positive without a purge and exactly zero with a purge of `h`.
        """
        horizon = 20
        index = pd.date_range("2020-01-01", periods=600, freq="B")

        def leaking_positions(split, index):
            if not len(split.train_index):
                return 0
            positions = index.get_indexer(split.train_index)
            test_start = index.get_indexer(split.test_index).min()
            # A training position leaks if its label window reaches test_start.
            return int(np.sum(positions + horizon >= test_start))

        unpurged = purged_walk_forward_splits(index, n_splits=4, purge=0)
        purged = purged_walk_forward_splits(index, n_splits=4, purge=horizon)

        leaked_before = sum(leaking_positions(s, index) for s in unpurged)
        leaked_after = sum(leaking_positions(s, index) for s in purged)

        assert leaked_before > 0, "fixture must actually contain leakage to remove"
        assert leaked_after == 0, (
            f"a purge of {horizon} must remove all label overlap, {leaked_after} remain"
        )

    def test_split_records_its_own_parameters(self):
        index = pd.date_range("2020-01-01", periods=600, freq="B")
        split = purged_walk_forward_splits(index, n_splits=3, purge=5, embargo=7)[0]
        payload = split.to_json()
        assert payload["purged"] == 5
        assert payload["train_start"] and payload["test_end"]


class TestFactorNeutralization:
    @pytest.fixture
    def factor_data(self):
        """A return stream with a deliberately embedded common-factor component."""
        rng = np.random.default_rng(99)
        dates = pd.date_range("2020-01-01", periods=800, freq="B")
        market = rng.normal(0.0004, 0.011, len(dates))
        size = rng.normal(0.0001, 0.006, len(dates))

        # 0.8 beta to market, 0.3 to size, plus a small genuine alpha.
        alpha = 0.00012
        idiosyncratic = rng.normal(0.0, 0.004, len(dates))
        strategy = alpha + 0.8 * market + 0.3 * size + idiosyncratic

        factors = pd.DataFrame({"market": market, "size": size}, index=dates)
        return pd.Series(strategy, index=dates), factors

    def test_neutralization_recovers_known_betas(self, factor_data):
        strategy, factors = factor_data
        model = FactorModel(name="test", version=1, factors=("market", "size"))

        result = neutralize_returns(strategy, factors, model)

        assert result.betas["market"] == pytest.approx(0.8, abs=0.05)
        assert result.betas["size"] == pytest.approx(0.3, abs=0.05)

    def test_neutralization_removes_the_embedded_factor_return(self, factor_data):
        """The residual must retain the alpha and shed the factor contribution."""
        strategy, factors = factor_data
        model = FactorModel(name="test", version=1, factors=("market", "size"))

        result = neutralize_returns(strategy, factors, model)

        assert result.residual_returns.std() < result.raw_returns.std()
        assert result.r_squared > 0.5
        # Residual mean should approximate the embedded alpha, not the raw mean.
        assert result.residual_returns.mean() == pytest.approx(0.00012, abs=0.0002)

    def test_raw_returns_are_preserved_alongside_residual(self, factor_data):
        """Neither substitutes for the other."""
        strategy, factors = factor_data
        model = FactorModel(name="test", version=1, factors=("market", "size"))

        result = neutralize_returns(strategy, factors, model)

        assert len(result.raw_returns) == len(result.residual_returns)
        assert not result.raw_returns.equals(result.residual_returns)
        assert "raw_annualized_mean" in result.to_json()
        assert "residual_annualized_mean" in result.to_json()

    def test_factor_model_is_a_versioned_hashable_artifact(self):
        """Otherwise 'factor-neutralized' hides its own implementation choices."""
        a = FactorModel(name="us4", version=1, factors=("market", "size"))
        b = FactorModel(name="us4", version=1, factors=("market", "size"), estimation_window=504)

        assert a.model_id == "factor-model/us4@1"
        assert a.content_hash != b.content_hash

    def test_missing_factor_is_refused(self, factor_data):
        strategy, factors = factor_data
        model = FactorModel(name="test", version=1, factors=("market", "momentum"))
        with pytest.raises(ValueError, match="absent from the data"):
            neutralize_returns(strategy, factors, model)


class TestReproducibility:
    def test_estimators_are_deterministic(self, synthetic_lineage):
        frame, truth = synthetic_lineage
        a = deflated_sharpe_ratio(frame[truth], trials_observed=47)
        b = deflated_sharpe_ratio(frame[truth], trials_observed=47)
        assert a.value == b.value

    def test_pbo_is_deterministic(self, synthetic_lineage):
        frame, _ = synthetic_lineage
        subset = frame[[c for c in frame.columns if c.startswith("noise_")][:10]]
        assert (
            probability_of_backtest_overfitting(subset, n_splits=6).value
            == probability_of_backtest_overfitting(subset, n_splits=6).value
        )

    def test_results_carry_estimator_version(self, synthetic_lineage):
        frame, truth = synthetic_lineage
        for result in (
            probabilistic_sharpe_ratio(frame[truth]),
            deflated_sharpe_ratio(frame[truth], trials_observed=10),
            minimum_track_record_length(frame[truth]),
        ):
            assert result.estimator_version
