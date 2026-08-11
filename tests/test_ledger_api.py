"""Tests for the ledgers and the investment-agent API surface.

The invariants under test are the ones that must hold in the schema and the
transport rather than in a UI template, because a convention enforced only at
render time is bypassed by the first new consumer.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.ledger import Ledger
from src.methodology import MethodologyRegistry
from src.methodology.spec import PerformanceClass

#: Every run and performance row must name the evaluation protocol that produced
#: it — `methodology + protocol = performance`. Citing only the methodology would
#: make the figure irreproducible, since costs, lag, grid and data snapshot all
#: live in the protocol.
PROTO = {"protocol_id": "protocol/standard@1", "protocol_hash": "0" * 64}


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "test.db")


@pytest.fixture
def published(ledger):
    registry = MethodologyRegistry()
    v1 = registry.get("hrp", 1)
    ledger.publish_methodology(v1)
    return ledger, v1


class TestMethodologyLedger:
    def test_publish_is_idempotent(self, published):
        ledger, v1 = published
        assert ledger.publish_methodology(v1) == v1.version_id
        assert len(ledger.list_methodologies()) == 1

    def test_republishing_changed_content_is_refused(self, published):
        """A version id names an immutable artifact; results already cite it."""
        ledger, v1 = published
        tampered = v1.revise(change_rationale="x", params={})
        object.__setattr__(tampered, "version", v1.version)  # same id, new body

        with pytest.raises(ValueError, match="immutable"):
            ledger.publish_methodology(tampered)


class TestTrialCounting:
    def test_ordinal_is_assigned_by_the_ledger(self, published):
        """The caller never supplies it — that is the entire mechanism."""
        ledger, v1 = published
        first = ledger.record_run(
            run_id="run_1", version_id=v1.version_id, manifest={}, manifest_digest="a", **PROTO
        )
        second = ledger.record_run(
            run_id="run_2", version_id=v1.version_id, manifest={}, manifest_digest="b", **PROTO
        )
        assert (first, second) == (1, 2)

    def test_trials_count_across_versions_of_a_lineage(self, ledger):
        """Twenty variants published under different version numbers is still
        twenty trials — the multiple-testing problem is per-lineage."""
        registry = MethodologyRegistry()
        v1, v2 = registry.get("hrp", 1), registry.get("hrp", 2)
        ledger.publish_methodology(v1)
        ledger.publish_methodology(v2)

        ledger.record_run(
            run_id="r1", version_id=v1.version_id, manifest={}, manifest_digest="a", **PROTO
        )
        ledger.record_run(
            run_id="r2", version_id=v2.version_id, manifest={}, manifest_digest="b", **PROTO
        )

        assert ledger.trial_count("hrp") == 2

    def test_run_against_unknown_version_is_refused(self, ledger):
        with pytest.raises(ValueError, match="unknown methodology"):
            ledger.record_run(
                run_id="r", version_id="methodology/nope@1", manifest={}, manifest_digest="x", **PROTO)


class TestPerformanceRecords:
    def test_disclosure_is_attached_to_the_row(self, published):
        """So the number cannot be served away from its caveat."""
        ledger, v1 = published
        ledger.record_run(
            run_id="r1", version_id=v1.version_id, manifest={}, manifest_digest="a", **PROTO
        )

        record = ledger.record_performance(
            **PROTO,
            performance_id="p1",
            run_id="r1",
            version_id=v1.version_id,
            performance_class=PerformanceClass.BACKTEST_HYPOTHETICAL,
            metric="annualized_return",
            value=0.0217,
            cost_model="flat_bps",
        )
        assert "206(4)-1" in record.disclosure
        assert record.trials_at_publication == 1

    def test_unclassified_performance_is_refused(self, published):
        ledger, v1 = published
        ledger.record_run(
            run_id="r1", version_id=v1.version_id, manifest={}, manifest_digest="a", **PROTO
        )

        with pytest.raises(TypeError, match="performance_class"):
            ledger.record_performance(
            **PROTO,
            performance_id="p1",
                run_id="r1",
                version_id=v1.version_id,
                performance_class="backtest",  # a bare string
                metric="annualized_return",
                value=0.0217,
                cost_model="flat_bps",
            )


class TestErrata:
    def test_erratum_supersedes_without_deleting(self, published):
        ledger, v1 = published
        ledger.record_run(
            run_id="r1", version_id=v1.version_id, manifest={}, manifest_digest="a", **PROTO
        )
        ledger.record_performance(
            **PROTO,
            performance_id="p1", run_id="r1", version_id=v1.version_id,
            performance_class=PerformanceClass.BACKTEST_HYPOTHETICAL,
            metric="annualized_return", value=0.13, cost_model="none",
        )
        ledger.publish_erratum(
            erratum_id="e1", title="Execution lag and costs",
            correction_type="NUMERICAL", cause_type="EXECUTION", severity="material",
            summary="look-ahead removed", supersedes=["p1"],
        )

        assert ledger.list_performance(v1.version_id) == []
        retained = ledger.list_performance(v1.version_id, include_superseded=True)
        assert len(retained) == 1
        assert retained[0]["superseded_by"] == "e1"
        assert retained[0]["value"] == 0.13  # the old figure is preserved verbatim


@pytest.fixture
def client(tmp_path, monkeypatch):
    import src.api as api

    monkeypatch.setattr(api, "_ledger", Ledger(tmp_path / "api.db"))
    api._bootstrap()
    return TestClient(api.app)


class TestAPI:
    def test_health_declares_paper_only(self, client):
        body = client.get("/health").json()
        assert body["paper_only"] is True
        assert body["external_execution_path"] is False

    def test_info_states_the_personalization_boundary(self, client):
        """The publisher's-exclusion posture is a served fact, not a doc comment."""
        body = client.get("/info").json()
        assert body["personalization"]["enabled"] is False
        assert "Lowe" in body["personalization"]["reason"]

    def test_agpl_source_offer_is_served(self, client):
        """AGPL §13 obliges offering source to network users."""
        body = client.get("/info").json()
        assert body["license"]["license"] == \
            "AGPL-3.0-or-later WITH Commons-Clause"
        # The entitlement itself, not just the label. The Commons Clause
        # withholds the right to sell and leaves §13 untouched, so a change
        # that dropped the source offer while renaming the licence would be
        # the one this test exists to catch.
        assert "corresponding source" in body["license"]["notice"]
        assert body["license"]["source"].startswith("https://")

    def test_latest_and_pinned_resolution(self, client):
        latest = client.get("/methodologies/hrp").json()
        assert latest["resolved"] == "latest"
        assert latest["methodology"]["version"] == max(latest["available_versions"])

        pinned = client.get("/methodologies/hrp?version=1").json()
        assert pinned["resolved"] == "pinned"
        assert pinned["methodology"]["version"] == 1

    def test_unknown_concept_is_404(self, client):
        assert client.get("/methodologies/nonexistent").status_code == 404

    def test_diff_reports_broken_comparability(self, client):
        body = client.get("/methodologies/hrp/diff?base=1&target=2").json()
        assert body["comparability"] == "broken"
        assert body["contract_breaks"]
        assert "lookback" in str(body["changed_fields"])

    def test_merge_endpoint_returns_layered_verdict(self, client):
        body = client.post("/methodologies/hrp/merge?base=1&ours=2&theirs=1").json()
        for key in (
            "structural_status", "contract_status",
            "economic_status", "comparability_status", "publishable",
        ):
            assert key in body

    def test_trials_endpoint_reports_platform_count(self, client):
        body = client.get("/trials/hrp").json()
        assert body["trials_observed"] == 0
        assert "self-reported" in body["note"]

    def test_current_strategies_exposes_provenance(self, client):
        strategies = client.get("/current-strategies").json()["strategies"]
        hrp = next(s for s in strategies if s["concept_id"] == "methodology/hrp")
        assert hrp["grounded_in"], "a published methodology must cite its source"
        assert hrp["limitations"]
        assert hrp["content_hash"]

    def test_mixed_class_series_is_refused(self, client, tmp_path):
        """The GIPS rule enforced in transport: the endpoint will not serve the
        data a backtest-into-live chart would need."""
        import src.api as api

        registry = MethodologyRegistry()
        v1 = registry.get("hrp", 1)
        api._ledger.record_run(
            run_id="r1", version_id=v1.version_id, manifest={}, manifest_digest="a", **PROTO
        )
        for pid, klass in (
            ("p1", PerformanceClass.BACKTEST_HYPOTHETICAL),
            ("p2", PerformanceClass.PAPER_LIVE_OOS),
        ):
            api._ledger.record_performance(
            **PROTO,
            performance_id=pid, run_id="r1", version_id=v1.version_id,
                performance_class=klass, metric="annualized_return",
                value=0.02, cost_model="flat_bps",
            )

        response = client.get(
            f"/performance/series?version_id={v1.version_id}&metric=annualized_return"
        )
        assert response.status_code == 409
        assert "mixed-class" in response.json()["detail"]

    def test_compare_refuses_naive_ranking_across_periods(self, client):
        """A longer lookback needs more warmup, so it starts later. Ranking the
        two returns would compare strategies measured on different data."""
        import src.api as api

        registry = MethodologyRegistry()
        for version, start in ((1, "2016-09-13"), (2, "2017-05-23")):
            m = registry.get("hrp", version)
            run_id = f"r{version}"
            api._ledger.record_run(
                run_id=run_id, version_id=m.version_id, manifest={}, manifest_digest=f"d{version}"
            , **PROTO)
            api._ledger.record_performance(
            **PROTO,
            performance_id=f"p{version}", run_id=run_id, version_id=m.version_id,
                performance_class=PerformanceClass.BACKTEST_HYPOTHETICAL,
                metric="annualized_return", value=0.01 * version,
                cost_model="flat_bps", period_start=start, period_end="2025-11-19",
            )

        body = client.get("/methodologies/hrp/compare").json()

        assert body["directly_comparable"] is False
        blockers = " ".join(body["comparability_blockers"])
        assert "evaluation periods differ" in blockers
        assert "rebalance frequency" in blockers
        assert body["note"], "must explain why the figures are not ranked"

    def test_compare_reports_trials_across_lineage(self, client):
        import src.api as api

        m = MethodologyRegistry().get("hrp", 1)
        api._ledger.record_run(
            run_id="rx", version_id=m.version_id, manifest={}, manifest_digest="dx", **PROTO
        )
        assert client.get("/methodologies/hrp/compare").json()["trials_observed"] == 1

    def test_discoveries_surface_broken_comparability(self, client):
        discoveries = client.get("/project/discoveries").json()["discoveries"]
        kinds = {d["kind"] for d in discoveries}
        assert "COMPARABILITY_BROKEN" in kinds

        proposal = next(d for d in discoveries if d["kind"] == "COMPARABILITY_BROKEN")
        assert proposal["hypothesis"]["verification_plan"]
        assert proposal["approval_policy"] == "human_required"

    def test_discovery_never_mutates(self, client):
        """Proposals are inputs to review, not changes."""
        before = client.get("/methodologies/hrp").json()["methodology"]["content_hash"]
        client.get("/project/discoveries")
        after = client.get("/methodologies/hrp").json()["methodology"]["content_hash"]
        assert before == after
