"""Tests for the UI layer.

The success criterion is legibility, not rendering: a reader should be able to
work out *why* a methodology is trustworthy — or why it is not — by navigating
the pages. So these assert that the provenance is actually present and that the
platform refuses to present misleading comparisons, rather than merely that a
page returns 200.
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from src.ledger import Ledger
from src.web.chain import CHAIN_ORDER


@pytest.fixture
def client(tmp_path, monkeypatch, prices_on_disk):
    import src.api as api
    import src.web.routes as routes

    ledger = Ledger(tmp_path / "ui.db")
    monkeypatch.setattr(api, "_ledger", ledger)
    monkeypatch.setattr(routes, "Ledger", lambda *a, **k: ledger)
    api._bootstrap()
    return TestClient(api.app)


def visible_text(html: str) -> str:
    body = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    body = re.sub(r"<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", body)


class TestPagesRender:
    @pytest.mark.parametrize(
        "path",
        ["/ui/", "/ui/protocols", "/ui/errata", "/ui/m/hrp", "/ui/m/hrp/1", "/ui/m/hrp/3"],
    )
    def test_page_is_served(self, client, path):
        assert client.get(path).status_code == 200

    def test_unknown_concept_is_404(self, client):
        assert client.get("/ui/m/nonexistent").status_code == 404

    def test_unknown_version_is_404(self, client):
        assert client.get("/ui/m/hrp/99").status_code == 404


class TestArtifactChain:
    """The panel that explains the architecture without documentation."""

    def test_chain_names_every_layer(self, client):
        html = client.get("/ui/m/hrp/3").text
        chain = html.split("Artifact chain")[1]

        for step, _domain in CHAIN_ORDER:
            assert step in chain, f"artifact chain is missing the {step} step"

    def test_chain_shows_concrete_artifact_ids(self, client):
        """Not labels — the actual versioned identifiers a result cites."""
        chain = client.get("/ui/m/hrp/3").text.split("Artifact chain")[1]

        assert "methodology/hrp@3" in chain
        assert "calendar/nyse@1" in chain
        assert "protocol/" in chain
        assert "stat-policy/library-default@1" in chain

    def test_chain_appears_on_every_methodology_page(self, client):
        for version in (1, 2, 3):
            assert "Artifact chain" in client.get(f"/ui/m/hrp/{version}").text


class TestProvenanceIsVisible:
    def test_methodology_page_shows_its_parameters(self, client):
        """A methodology is executable data; the page must show the data."""
        text = visible_text(client.get("/ui/m/hrp/3").text)

        assert "covariance estimator" in text
        assert "lookback" in text
        assert "weight bounds" in text
        assert "content hash" in text

    def test_methodology_page_shows_assumptions_and_limitations(self, client):
        text = visible_text(client.get("/ui/m/hrp/3").text)
        assert "Assumptions" in text
        assert "Limitations" in text
        assert "not a replication of that paper" in text

    def test_result_names_the_protocol_that_produced_it(self, client):
        """`methodology + protocol = performance` — a figure citing only the
        methodology would be irreproducible."""
        text = visible_text(client.get("/ui/m/hrp/3").text)
        assert "Measured under" in text
        assert "protocol/" in text
        assert "execution lag" in text

    def test_trial_count_is_shown_on_the_library(self, client):
        text = visible_text(client.get("/ui/").text)
        assert "Configurations attempted" in text
        assert "multiple-testing correction" in text

    def test_statistics_explain_what_they_account_for(self, client):
        text = visible_text(client.get("/ui/m/hrp/3").text)
        assert "DSR" in text
        assert "attempted configuration" in text

    def test_change_rationale_is_shown(self, client):
        """Why a version exists is a first-class question."""
        text = visible_text(client.get("/ui/m/hrp/3").text)
        assert "Why it exists" in text or "change" in text.lower()


class TestComparabilityRefusal:
    """The strongest differentiator: refusing a misleading comparison."""

    def test_incomparable_versions_render_a_verdict_not_a_chart(self, client):
        html = client.get("/ui/m/hrp/compare?a=2&b=3").text
        text = visible_text(html)

        assert "Not comparable" in text
        assert "weight bound" in text, "the specific reason must be named"
        # A *performance* chart would invite the comparison the page refuses.
        # The lineage timeline is a provenance visual and is deliberately drawn:
        # it shows where the boundary falls, which is the page's subject.
        assert "<canvas" not in html
        assert "Performance comparison unavailable" in text

    def test_verdict_precedes_the_eligibility_checklist(self, client):
        """Order is the argument: a reader must meet the verdict first."""
        text = visible_text(client.get("/ui/m/hrp/compare?a=2&b=3").text)
        assert text.index("Not comparable") < text.index("Performance comparison")

    def test_each_blocking_difference_states_why_it_matters(self, client):
        text = visible_text(client.get("/ui/m/hrp/compare?a=2&b=3").text)
        assert "different strategies" in text, (
            "naming the field that changed is not the same as saying what it means"
        )

    def test_compare_page_explains_what_would_fix_it(self, client):
        text = visible_text(client.get("/ui/m/hrp/compare?a=2&b=3").text)
        assert "How comparability could be restored" in text
        assert "Re-run" in text

    def test_methodology_page_links_incomparable_siblings_to_the_reason(self, client):
        html = client.get("/ui/m/hrp/3").text
        assert "compare?a=" in html, (
            "an incomparable sibling must link to the explanation, not just say no"
        )


class TestNoCharts:
    """The differentiator is provenance, not plotting."""

    @pytest.mark.parametrize("path", ["/ui/", "/ui/m/hrp/3", "/ui/protocols", "/ui/errata"])
    def test_no_chart_libraries_or_canvases(self, client, path):
        html = client.get(path).text
        for marker in ("<canvas", "chart.js", "plotly", "d3.min.js", "bokeh"):
            assert marker not in html.lower(), f"{path} pulls in {marker}"


class TestErrataAreFirstClass:
    def test_errata_page_lists_published_corrections(self, client):
        import src.web.routes as routes

        routes.Ledger().publish_erratum(
            erratum_id="test-01", title="Test correction",
            correction_type="NUMERICAL", cause_type="DATA", severity="material",
            summary="A correction.", supersedes=[],
        )
        text = visible_text(client.get("/ui/errata").text)
        assert "test-01" in text
        assert "Test correction" in text

    def test_errata_page_states_the_retention_principle(self, client):
        text = visible_text(client.get("/ui/errata").text)
        assert "never deleted" in text or "retained" in text


class TestDisclosureTravels:
    def test_hypothetical_disclosure_on_every_page(self, client):
        """The disclosure is bound to the data object, so it renders everywhere."""
        for path in ("/ui/", "/ui/m/hrp/3", "/ui/protocols", "/ui/errata"):
            text = visible_text(client.get(path).text)
            assert "Hypothetical backtested performance" in text
            assert "not a track record" in text.lower()

    def test_agpl_source_offer_is_present(self, client):
        html = client.get("/ui/").text
        assert "AGPL-3.0" in html
        assert "github.com/redevops-io/RAAAL" in html


class TestProtocolPage:
    def test_shows_what_a_protocol_carries(self, client):
        text = visible_text(client.get("/ui/protocols").text)
        for field in ("Calendar", "Transaction costs", "Execution lag", "Walk-forward"):
            assert field in text

    def test_sealed_holdout_is_labelled_and_explained(self, client):
        text = visible_text(client.get("/ui/protocols").text)
        assert "sealed" in text.lower()
        assert "unreachable" in text

    def test_calendars_are_shown_as_artifacts(self, client):
        text = visible_text(client.get("/ui/protocols").text)
        assert "calendar/nyse@1" in text
        assert "Sessions per year" in text


class TestRunIsARecordNotARendering:
    """A run page must read persisted state, not recompute it.

    Recomputing would mean the page shows today's code applied to yesterday's
    execution — which is exactly the confusion the artifact model exists to
    prevent.
    """

    def _recorded_run(self, client):
        import src.web.routes as routes
        from src.methodology import MethodologyRegistry

        ledger = routes.Ledger()
        m = MethodologyRegistry().get("hrp", 3)
        ledger.publish_methodology(m)
        ledger.record_run(
            run_id="run_test_1",
            version_id=m.version_id,
            protocol_id="protocol/long-warmup@1",
            protocol_hash="a" * 64,
            manifest={}, manifest_digest="d1",
            result_status={"computation_valid": True, "contract_valid": True,
                           "economic_valid": True, "reproducible": True,
                           "statistical_assessment_complete": True, "flags": []},
            diagnostics={"flags": [], "top_asset": "BIL",
                         "top_asset_mean_weight": 0.25, "effective_n_assets": 6.37},
            execution_audit={"n_rebalances": 95, "fallback_share": 0.0,
                             "requested_turnover_cap": 0.25,
                             "realized_turnover_mean": 0.067,
                             "realized_turnover_max": 0.25,
                             "precedence_override_count": 7,
                             "precedence_overrides": []},
            assessment={"computation_status": "VALID", "trial_count": 1,
                        "observations": 1982, "estimator_version": "0.1.0",
                        "count_policy": "DSR_COUNTABLE_OUTCOMES",
                        "dsr": {"value": 0.93}, "psr": {"value": 0.93},
                        "pbo": {"value": 0.0}},
            policy_evaluation={"policy_id": "stat-policy/library-default@1",
                               "status": "WARN", "evidence_grade": "MODERATE",
                               "findings": []},
            publication_decision={"surface": "PUBLIC_LIBRARY",
                                  "decision": "ALLOW_WITH_DISCLOSURE",
                                  "may_claim_validated": False,
                                  "disclosures": [], "hard_blockers": []},
        )
        return "run_test_1"

    def test_run_page_shows_persisted_verdicts(self, client):
        run_id = self._recorded_run(client)
        text = visible_text(client.get(f"/ui/runs/{run_id}").text)

        assert "WARN" in text
        assert "ALLOW_WITH_DISCLOSURE" in text
        assert "MODERATE" in text

    def test_run_page_states_nothing_is_recomputed(self, client):
        run_id = self._recorded_run(client)
        text = visible_text(client.get(f"/ui/runs/{run_id}").text)
        assert "re-derived a verdict" in text
        assert "Recorded at execution" in text and "State now" in text

    def test_run_page_shows_precedence_overrides(self, client):
        run_id = self._recorded_run(client)
        text = visible_text(client.get(f"/ui/runs/{run_id}").text)
        assert "Precedence overrides" in text
        assert "7" in text

    def test_run_page_shows_all_five_status_conditions(self, client):
        run_id = self._recorded_run(client)
        text = visible_text(client.get(f"/ui/runs/{run_id}").text)
        for condition in ("computation valid", "contract valid", "economic valid",
                          "reproducible"):
            assert condition in text

    def test_unknown_run_is_404(self, client):
        assert client.get("/ui/runs/nope").status_code == 404

    def test_run_survives_a_policy_change(self, client):
        """The point of persisting: a run keeps the verdict it received."""
        import src.web.routes as routes

        run_id = self._recorded_run(client)
        stored = routes.Ledger().get_run(run_id)
        assert stored["policy_evaluation"]["policy_id"] == "stat-policy/library-default@1"
        assert stored["policy_evaluation"]["status"] == "WARN"


class TestFindingsPage:
    def test_findings_page_lists_conclusions(self, client):
        text = visible_text(client.get("/ui/findings").text)
        assert "hrp-degenerates-to-cash-proxy" in text
        assert "Synthesised from" in text
        assert "What it changed" in text

    def test_findings_explain_why_they_are_separate(self, client):
        text = visible_text(client.get("/ui/findings").text)
        assert "one conclusion" in text

    def test_methodology_page_shows_what_was_concluded(self, client):
        text = visible_text(client.get("/ui/m/hrp/3").text)
        assert "What has been concluded about this" in text


class TestErrataTaxonomy:
    """Three questions, three stored columns, none inferred from prose."""

    def _publish(self, client, **kw):
        import src.web.routes as routes

        defaults = dict(
            erratum_id="e-tax", title="T", correction_type="NUMERICAL",
            cause_type="DATA", severity="material", summary="S", supersedes=[],
        )
        defaults.update(kw)
        routes.Ledger().publish_erratum(**defaults)

    def test_both_dimensions_are_stored_and_shown(self, client):
        self._publish(client, correction_type="INTERPRETIVE", cause_type="STATISTICAL")
        text = visible_text(client.get("/ui/errata").text)
        assert "INTERPRETIVE" in text
        assert "STATISTICAL" in text

    def test_taxonomy_is_read_verbatim_not_derived(self, client):
        """The UI must not infer a cause from the summary text."""
        import src.web.routes as routes

        self._publish(
            client, correction_type="METHODOLOGICAL", cause_type="PUBLICATION",
            summary="this prose mentions data and execution and statistics",
        )
        stored = next(
            e for e in routes.Ledger().list_errata() if e["erratum_id"] == "e-tax"
        )
        assert stored["correction_type"] == "METHODOLOGICAL"
        assert stored["cause_type"] == "PUBLICATION"

    def test_invalid_correction_type_is_refused(self, client):
        with pytest.raises(ValueError, match="correction_type"):
            self._publish(client, correction_type="SOMETHING_ELSE")

    def test_invalid_cause_type_is_refused(self, client):
        with pytest.raises(ValueError, match="cause_type"):
            self._publish(client, cause_type="VIBES")

    def test_no_unknown_member_exists(self):
        """A system insisting on declared meaning must not ship an undeclared value."""
        from src.ledger import CAUSE_TYPES, CORRECTION_TYPES

        assert "UNKNOWN" not in CORRECTION_TYPES
        assert "UNKNOWN" not in CAUSE_TYPES

    def test_shipped_errata_declare_both_dimensions(self, client):
        """Backfilled explicitly, not defaulted."""
        import scripts.seed_errata as seed

        for erratum in seed.ERRATA:
            assert erratum["correction_type"]
            assert erratum["cause_type"]
            assert erratum["correction_type"] != erratum["cause_type"]
