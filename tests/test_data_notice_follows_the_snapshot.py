"""The disclosure describes the data the run actually used.

`_data_notice` was a hardcoded synthetic sentence and a `None` for everything
else:

    "The series are invented ... calibrated to no real security."

The moment a vendor snapshot was authorised, that sentence became a false
statement printed beside every figure — and the other branch said nothing at
all, where the licensing record requires an attribution.

A disclosure written next to the data it describes goes stale when the data
changes. One derived from the snapshot cannot. So the notice is read from the
manifest of whatever snapshot the policy resolved, and the licensing record's
attribution decision reaches the page without anyone remembering to copy it.

The failure this guards is worse than a missing notice: a missing one leaves a
user uninformed, and a wrong one tells them the number in front of them came
from invented data when it did not, or from real data when it did not.
"""
from __future__ import annotations

import pytest

VENDOR = "market-data-egress/pilot-vendor-approved@1"


@pytest.fixture
def under():
    from src.deploy.context import bind, resolve, unbind

    def _bind(policy):
        bind(resolve({"PILOT_DATA_POLICY": policy}))
        from src.workspace.routes import _data_notice
        return _data_notice()

    try:
        yield _bind
    finally:
        unbind()


class TestSynthetic:
    def test_it_says_the_series_are_invented(self, under):
        notice = under("SYNTHETIC_ONLY")
        assert "synthetic" in notice["headline"].lower()
        assert "invented" in notice["detail"]

    def test_it_does_not_name_a_vendor(self, under):
        """Naming a source for invented data would be the same lie facing the
        other way."""
        notice = under("SYNTHETIC_ONLY")
        assert "yahoo" not in (notice["headline"] + notice["detail"]).lower()


class TestVendor:
    def test_it_names_the_source_and_the_acknowledgement(self, under):
        notice = under(VENDOR)
        both = notice["headline"] + " " + notice["detail"]
        assert "Yahoo Finance" in both
        assert "non-commercial" in both.lower()

    def test_it_no_longer_claims_the_data_is_invented(self, under):
        """The specific false sentence this file exists for."""
        both = (under(VENDOR)["headline"] + " " + under(VENDOR)["detail"]).lower()
        for phrase in ("invented", "no real security", "synthetic"):
            assert phrase not in both, phrase

    def test_it_states_the_coverage_the_snapshot_actually_has(self, under):
        """Read from the manifest, not typed here — a hand-written range is
        the thing that goes stale on the next snapshot."""
        from src.market_data.access import approved_snapshot

        coverage = (approved_snapshot().raw or {}).get("coverage") or {}
        detail = under(VENDOR)["detail"]
        assert str(coverage["start"]) in detail
        assert str(coverage["end"]) in detail

    def test_the_attribution_comes_from_the_licensing_decision(self, under):
        """The manifest's attribution block is the decision recorded in
        `data/licensing/market-data-licensing@1.yaml`. If someone changes the
        decision and not the manifest, this fails rather than the page quietly
        disagreeing with the record."""
        import yaml
        from pathlib import Path

        from src.market_data.access import approved_snapshot
        from src.market_data.loader import REPO_ROOT

        snapshot = approved_snapshot()
        record = yaml.safe_load(
            (REPO_ROOT / snapshot.raw["license_record"]).read_text())
        decision = record["answers"]["what attribution or notice is required"]
        # Substance, not string equality: the record says "Yahoo" and the
        # manifest says "Yahoo Finance", which is the same decision. What must
        # not happen is the two naming different sources, or the manifest
        # dropping the non-commercial acknowledgement the decision requires.
        answer = decision["answer"].lower()
        source = snapshot.raw["attribution"]["source"].lower()
        assert "yahoo" in answer and "yahoo" in source
        assert source.split()[0] in answer, (source, answer)
        assert "non-commercial" in answer
        assert "non-commercial" in \
            snapshot.raw["attribution"]["acknowledgement"].lower()


class TestNoApprovedSnapshot:
    def test_it_says_so_rather_than_nothing(self, under, monkeypatch):
        """The old code returned None here, so a deployment with no authorised
        snapshot showed no disclosure at all. It also shows no figures, and
        saying why is better than silence."""
        import src.market_data.access as access

        monkeypatch.setattr(access, "approved_snapshot", lambda: None)
        notice = under(VENDOR)
        assert notice is not None
        assert "no approved market data" in notice["headline"].lower()


class TestTheGateIsChecked:
    """`approved_snapshot` must verify the licensing record, not merely find a
    manifest. A file anyone can add is not an authorisation."""

    def test_an_incomplete_record_yields_no_snapshot(self, monkeypatch):
        from src.market_data import access

        monkeypatch.setattr(access, "licensing_resolved", lambda record: False,
                            raising=False)
        import src.market_data.pilot_policy as policy

        monkeypatch.setattr(policy, "licensing_resolved", lambda record: False)
        assert access.approved_snapshot() is None

    def test_a_missing_record_yields_no_snapshot(self, monkeypatch):
        from src.market_data import access
        from src.market_data.loader import load_manifest

        def without_record(path):
            snapshot = load_manifest(path)
            snapshot.raw.pop("license_record", None)
            return snapshot

        monkeypatch.setattr(access, "load_manifest", without_record,
                            raising=False)
        import src.market_data.loader as loader

        monkeypatch.setattr(loader, "load_manifest", without_record)
        assert access.approved_snapshot() is None
