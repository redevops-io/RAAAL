"""The market-data tier split, and the integrity checks that make it safe.

Two claims are under test:

    the default suite runs on committed, invented data and reaches no network
    a pinned snapshot that has moved is refused, never silently substituted

The second is the one that matters. An integrity check that falls back to the
newest object is not a check; it is a log line. Every figure computed after that
point describes data the run never names.
"""
from __future__ import annotations

import shutil

import pandas as pd
import pytest

from src.market_data import (
    IntegrityError,
    Snapshot,
    SnapshotUnavailable,
    content_digest,
    describe_for_run,
    file_sha256,
    load_manifest,
    load_prices,
    production_snapshot,
    synthetic_snapshot,
)
from tests.conftest import SYNTHETIC


class TestTheCommittedFixture:

    def test_it_is_in_the_repository(self):
        assert SYNTHETIC.exists(), (
            "the synthetic fixture is committed on purpose; without it a fresh "
            "clone cannot run the evaluation tests at all")

    def test_it_matches_its_pinned_content_digest(self, synthetic_prices):
        """A fixture edited by hand is caught here, not three layers up."""
        assert content_digest(synthetic_prices) == \
            synthetic_snapshot().content_digest

    def test_it_is_marked_redistributable(self):
        snapshot = synthetic_snapshot()
        assert snapshot.redistributable is True
        assert snapshot.license_class == "redistributable"
        assert snapshot.may_be_exported is True

    def test_it_covers_the_whole_protocol_window(self, synthetic_prices):
        """Protocols run 2016-2025 with 504-day lookbacks. A shorter fixture
        leaves the long-warmup protocol nothing to measure after warmup."""
        assert str(synthetic_prices.index.min().date()) <= "2016-01-05"
        assert str(synthetic_prices.index.max().date()) >= "2025-11-19"
        assert len(synthetic_prices) > 2400

    def test_the_methodology_universe_is_complete(self, synthetic_prices):
        universe = ["SPY", "SH", "TLT", "TBT", "LQD", "DBC", "GLD", "HYG", "BIL"]
        assert set(universe) <= set(synthetic_prices.columns)
        assert not synthetic_prices[universe].isna().any().any(), (
            "a gap in the evaluated universe changes results rather than "
            "exercising the missing-value path; the gap belongs elsewhere")

    def test_it_carries_a_missing_value_case(self, synthetic_prices):
        """Code that assumes a complete rectangle should break in a test."""
        assert synthetic_prices["BRK-B"].isna().any()
        assert synthetic_prices["BRK-B"].loc["2016":"2017-06-14"].isna().all()
        assert synthetic_prices["BRK-B"].loc["2017-07":].notna().all()

    def test_it_carries_a_split_like_discontinuity(self, synthetic_prices):
        """A price jump with no matching return — the corporate-action shape."""
        before = synthetic_prices["MGK"].loc[:"2021-03-12"].iloc[-1]
        after = synthetic_prices["MGK"].loc["2021-03-15":].iloc[0]
        assert after < before / 3, "expected a 4:1-style discontinuity"

    def test_the_covariance_structure_is_real(self, synthetic_prices):
        """Hierarchical clustering needs correlations to cluster.

        Independent assets give an identity correlation matrix, and every
        allocation result becomes equal-weight for reasons the test would not
        reveal.
        """
        returns = synthetic_prices[["SPY", "SH", "QQQ", "BIL"]].pct_change().dropna()
        correlation = returns.corr()
        assert correlation.loc["SPY", "QQQ"] > 0.5, "equities should co-move"
        assert correlation.loc["SPY", "SH"] < -0.5, "the inverse pair should oppose"
        assert abs(correlation.loc["SPY", "BIL"]) < 0.3, "cash should be near-flat"

    def test_it_is_not_calibrated_to_anything_real(self, synthetic_prices):
        """A guard against someone quietly swapping vendor data in here.

        Real SPY did not start at 200.00 on the first session of 2016, and a
        fixture that happened to match a real series would be redistributing
        licensed data under a redistributable label.
        """
        assert synthetic_prices["SPY"].iloc[0] == pytest.approx(200.0, abs=1.0)


class TestIntegrity:

    def test_the_digest_survives_reserialization(self, synthetic_prices, tmp_path):
        """Parquet embeds its writer version; the content digest must not care.

        Otherwise an Arrow upgrade fails the suite with "the data moved" when
        nothing moved, and people learn to regenerate the expected value.
        """
        other = tmp_path / "rewritten.parquet"
        synthetic_prices.to_parquet(other, compression="gzip")
        assert content_digest(pd.read_parquet(other)) == \
            content_digest(synthetic_prices)

    def test_a_changed_value_changes_the_digest(self, synthetic_prices):
        tampered = synthetic_prices.copy()
        tampered.iloc[100, 0] += 0.01
        assert content_digest(tampered) != content_digest(synthetic_prices)

    def test_missing_and_zero_are_distinguished(self, synthetic_prices):
        """"Not listed yet" is not "worth nothing"."""
        a = synthetic_prices.copy(); a.iloc[5, 3] = float("nan")
        b = synthetic_prices.copy(); b.iloc[5, 3] = 0.0
        assert content_digest(a) != content_digest(b)

    def test_column_order_does_not_change_the_digest(self, synthetic_prices):
        shuffled = synthetic_prices[list(reversed(synthetic_prices.columns))]
        assert content_digest(shuffled) == content_digest(synthetic_prices)

    def test_a_mismatched_digest_is_refused(self, tmp_path):
        """The load path must raise, not warn and continue."""
        copy = tmp_path / "prices.parquet"
        shutil.copy(SYNTHETIC, copy)
        snapshot = Snapshot(
            dataset_id="market-data/prices", snapshot_id="pinned-elsewhere",
            kind="synthetic", uri=str(copy), schema_version="1",
            content_digest="mdv1:" + "0" * 64)

        with pytest.raises(IntegrityError) as exc:
            load_prices(snapshot)
        assert "does not match its pinned content digest" in str(exc.value)
        assert "data the run never saw" in str(exc.value)


class TestTheLicensedSnapshotIsNotReachableByAccident:

    def test_the_default_suite_cannot_reach_object_storage(self):
        """`allow_network` defaults to False.

        A test that silently acquires a network dependency passes locally, fails
        in CI, and gets diagnosed as flaky for a week.
        """
        snapshot = Snapshot(
            dataset_id="market-data/prices", snapshot_id="prices-2026-07-30",
            kind="licensed", uri="s3://example/prices.parquet",
            schema_version="1", object_version_id="abc", sha256="0" * 64)

        with pytest.raises(SnapshotUnavailable) as exc:
            load_prices(snapshot, allow_network=False)
        assert "network access was not requested" in str(exc.value)

    def test_an_unversioned_object_is_refused(self):
        """Same commit + same URI must not be able to mean two things."""
        snapshot = Snapshot(
            dataset_id="market-data/prices", snapshot_id="unpinned",
            kind="licensed", uri="s3://example/prices.parquet",
            schema_version="1", object_version_id=None)

        with pytest.raises(SnapshotUnavailable) as exc:
            load_prices(snapshot, allow_network=True)
        assert "no object version id" in str(exc.value)

    def test_the_licensed_manifest_is_marked_restricted(self):
        snapshot = production_snapshot()
        assert snapshot.license_class == "restricted"
        assert snapshot.redistributable is False
        assert snapshot.may_be_exported is False, (
            "restricted data must never reach a public export or case bundle")

    def test_the_licensed_manifest_records_an_unfinished_review(self):
        """The licensing questions are open, and the manifest says so.

        This test exists to fail the day someone populates the URI without
        answering them — private storage does not cure a licence that forbids
        server-side redistribution or multi-user serving.
        """
        raw = production_snapshot().raw
        review = raw["licensing_review"]
        unconfirmed = [k for k, v in review.items() if v == "UNCONFIRMED"]
        if not unconfirmed:
            assert raw.get("uri") and not str(raw["uri"]).startswith("${"), (
                "the licensing review is complete but no snapshot is pinned")
        else:
            assert str(raw["uri"]).startswith("${"), (
                f"a snapshot is pinned while {unconfirmed} remain unconfirmed; "
                "uploading licensed data before the review is the thing this "
                "manifest exists to prevent")

    def test_the_manifest_demands_immutability(self):
        requirements = production_snapshot().raw["storage_requirements"]
        assert requirements["bucket_versioning"] == "required"
        assert requirements["immutable_object_keys"] == "required"
        assert requirements["sha256_verified_on_load"] == "required"
        assert requirements["refuse_on_mismatch"] == "required"


class TestWhatARunRecords:

    def test_a_run_records_the_realized_snapshot(self, synthetic_prices):
        """The realized digest, not the manifest's copy of it.

        What was actually loaded is the only version of this fact worth storing.
        """
        record = describe_for_run(synthetic_snapshot(), synthetic_prices)
        assert record["snapshot_id"] == "prices-synthetic-1"
        assert record["realized_content_digest"] == content_digest(synthetic_prices)
        assert record["sessions"] == len(synthetic_prices)
        assert record["calendar"] == "calendar/nyse@1"

    def test_the_record_carries_no_prices(self, synthetic_prices):
        """Identity, never bytes. A run record is not a copy of the corpus."""
        record = describe_for_run(synthetic_snapshot(), synthetic_prices)
        assert all(not isinstance(v, (list, dict)) for v in record.values())


@pytest.mark.market_data_integration
class TestTheLicensedTier:
    """Explicitly enabled: `pytest -m market_data_integration`.

    Fails rather than skips when requested. A silent skip in the tier that
    exists to check the real data is indistinguishable from a pass.
    """

    def test_the_pinned_snapshot_loads_and_verifies(self, licensed_snapshot):
        frame, snapshot = licensed_snapshot
        assert not frame.empty
        assert snapshot.object_version_id, "the tier requires a pinned version"
        assert file_sha256 is not None

    def test_the_realized_snapshot_is_recorded(self, licensed_snapshot):
        frame, snapshot = licensed_snapshot
        record = describe_for_run(snapshot, frame)
        assert record["license_class"] == "restricted"
        assert record["realized_content_digest"]
