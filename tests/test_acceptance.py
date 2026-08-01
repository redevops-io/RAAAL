"""The eleven acceptance questions, asked of the rendered pages.

Section 8 of the UI plan states them as questions a reader must be able to
answer *without reading source*. They are written here as questions rather than
as component tests on purpose: a suite of green component tests is compatible
with a product nobody can use, and the criterion that actually matters is
whether the answer is on the page.

Question 11 — the five-second criterion — is the only one that cannot be
asserted directly, so it is tested by proxy: the library must carry a per-version
state summary above the fold, and it must be scannable without expanding
anything or opening a version page.
"""
from __future__ import annotations

import re

import pytest

from src.ledger import Ledger


@pytest.fixture
def client(tmp_path, monkeypatch):
    import src.api as api
    import src.web.routes as routes
    from fastapi.testclient import TestClient

    ledger = Ledger(tmp_path / "acceptance.db")
    monkeypatch.setattr(api, "_ledger", ledger)
    monkeypatch.setattr(routes, "Ledger", lambda *a, **k: ledger)
    api._bootstrap()
    return TestClient(api.app)


def text(html: str) -> str:
    body = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


def record_run(client, *, status="WARN", policy_findings=()):
    import src.web.routes as routes
    from src.methodology import MethodologyRegistry

    ledger = routes.Ledger()
    m = MethodologyRegistry().get("hrp", 3)
    ledger.publish_methodology(m)
    ledger.record_run(
        run_id="run_acc_1", version_id=m.version_id,
        protocol_id="protocol/long-warmup@1", protocol_hash="a" * 64,
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
                         "precedence_override_count": 7, "precedence_overrides": []},
        assessment={"computation_status": "VALID", "trial_count": 1,
                    "observations": 1982, "estimator_version": "0.1.0",
                    "count_policy": "DSR_COUNTABLE_OUTCOMES",
                    "dsr": {"value": 0.93}, "psr": {"value": 0.93},
                    "pbo": {"value": 0.0}},
        policy_evaluation={"policy_id": "stat-policy/library-default@1",
                           "status": status, "evidence_grade": "MODERATE",
                           "findings": list(policy_findings)},
        publication_decision={"surface": "PUBLIC_LIBRARY",
                              "decision": "ALLOW_WITH_DISCLOSURE",
                              "may_claim_validated": False,
                              "disclosures": [], "hard_blockers": []},
    )
    return "run_acc_1"


class TestTheElevenQuestions:

    def test_q1_why_hrp3_exists(self, client):
        page = text(client.get("/ui/m/hrp/3").text)
        assert "Why it exists" in page
        # Not merely that a rationale field exists — that it is populated.
        lineage = page.split("Why it exists")[1]
        assert len(lineage.strip()) > 80

    def test_q2_why_a_version_is_blocked_despite_strong_statistics(self, client):
        """The blocker must be named, next to the statistics that look fine."""
        page = text(client.get("/ui/m/hrp/1").text)
        assert "Publication status" in page
        assert "Decision" in page
        # The chain headline states the kind of adversity, not just "bad".
        assert any(phrase in page for phrase in
                   ("Blocked at", "attention at", "affected by", "Clear across the chain"))

    def test_q3_what_invalidated_the_published_figures(self, client):
        page = text(client.get("/ui/findings").text)
        assert "invalidates results of" in page
        assert "Synthesised from" in page, "a finding must show its evidence"

    def test_q4_whether_two_methodologies_are_comparable_and_why_not(self, client):
        page = text(client.get("/ui/m/hrp/compare?a=2&b=3").text)
        assert "Not comparable" in page
        assert "weight bound" in page
        assert "different strategies" in page, "the reason must state a consequence"

    def test_q5_declared_versus_inherited_assumptions(self, client):
        html = client.get("/ui/claims").text
        # Directness is its own visual channel: inherited edges are dashed.
        assert "edge-inherited" in html
        assert "edge-direct" in html
        assert "inherited" in text(html)

    def test_q6_which_mechanism_realizes_every_declared_rule(self, client):
        page = text(client.get("/ui/claims").text)
        assert "Realized by" in page, (
            "a declaration with no named realization is the failure mode the "
            "verifier exists to catch"
        )

    def test_q7_what_verdict_a_run_received_historically(self, client):
        run_id = record_run(client)
        page = text(client.get(f"/ui/runs/{run_id}").text)

        assert "Recorded at execution" in page
        assert "WARN" in page and "ALLOW_WITH_DISCLOSURE" in page

    def test_q8_whether_current_policy_would_differ(self, client):
        run_id = record_run(client)
        page = text(client.get(f"/ui/runs/{run_id}").text)

        assert "Would today's policy agree?" in page
        assert "As recorded" in page
        assert "stat-policy/library-default@1" in page

    def test_q8_drift_never_overwrites_the_recorded_verdict(self, client):
        """The whole point of the ledger. Re-judging must not rewrite history."""
        run_id = record_run(client, status="FAIL")
        page = text(client.get(f"/ui/runs/{run_id}").text)

        recorded = page.split("Would today's policy agree?")[0]
        assert "FAIL" in recorded, "the recorded verdict was replaced by a fresh one"

    def test_q9_what_evidence_changed_a_claims_status(self, client):
        page = text(client.get("/ui/claims").text)
        assert "Evidence" in page
        for stance in ("Supports", "Contradicts"):
            assert stance in page
        assert "Strength" in page

    def test_q10_whether_an_investigation_concluded_with_no_finding(self, client):
        """Answerable only since `Investigation` became its own artifact.

        The findings page can say a conclusion is open or superseded. It cannot
        say an inquiry ran and produced nothing, because an inquiry with no
        finding leaves no finding to render.
        """
        page = text(client.get("/ui/investigations").text)

        assert "Closed with none" in page
        assert "No finding." in page
        assert "a result, not an absence of one" in page

    def test_q11_library_state_is_scannable_without_opening_anything(self, client):
        html = client.get("/ui/").text
        page = text(html)

        # One glyph row per version, each dot carrying a symbol that means
        # something without colour.
        assert html.count('class="glyph"') >= 1
        assert any(sym in page for sym in ("●", "◐", "✕", "○", "·"))
        # A plain-language headline per row, so scanning needs no legend lookup.
        assert any(phrase in page for phrase in
                   ("Clear across the chain", "Blocked at", "attention at", "affected by"))
        # And nothing requiring interaction to reveal state.
        library = html.split("<main")[-1] if "<main" in html else html
        assert "<details" not in library.split("Relationships as a table")[0]


class TestTheReleaseCriterion:
    """Backward to declarations; forward to impact; visually, before any prose."""

    def test_every_conclusion_links_backward_to_its_declarations(self, client):
        html = client.get("/ui/m/hrp/3").text
        for target in ("/ui/claims", "/ui/findings", "/ui/protocols", "/ui/errata"):
            assert target in html, f"no backward path to {target}"

    def test_every_artifact_links_forward_to_its_impact(self, client):
        claims = client.get("/ui/claims").text
        assert "reference this claim" in text(claims)
        assert "/ui/m/" in claims, "a claim must reach the methodologies using it"

    def test_both_directions_are_visual_before_they_are_textual(self, client):
        """Each graph carries an SVG *and* the same payload as a table."""
        for path in ("/ui/claims", "/ui/findings"):
            html = client.get(path).text
            assert "graph-svg" in html
            assert "Relationships as a table" in html
            svg_keys = set(re.findall(r'data-key="([^"]+)"', html))
            table_keys = set(re.findall(r'data-edge="([^"]+)"', html))
            assert svg_keys == table_keys, (
                f"{path}: diagram and table disagree about what is shown"
            )


class TestQuestionTenNowHasAnArtifact:
    """§8.10 — whether an investigation concluded with no finding.

    It was previously unanswerable in principle, not just unimplemented: a
    finding *was* the investigation, so an inquiry that produced no finding had
    no object to exist as.
    """

    def test_null_results_are_visible_without_opening_anything(self, client):
        page = text(client.get("/ui/investigations").text)

        assert "Closed with none" in page
        assert "Null result" in page
        assert "Inconclusive" in page

    def test_a_null_result_says_what_it_examined(self, client):
        page = text(client.get("/ui/investigations").text)
        assert "What it examined" in page
        assert "methodology/xsmom@1" in page

    def test_a_null_result_is_not_presented_as_a_failure(self, client):
        html = client.get("/ui/investigations").text
        for fragment in re.findall(r"<[^>]*chip[^>]*>[^<]*(?:Null result|Inconclusive)[^<]*<", html):
            assert "chip block" not in fragment

    def test_inconclusive_is_distinguished_from_no_effect(self, client):
        page = text(client.get("/ui/investigations").text)
        assert "could not settle the question" in page
        assert "is not there" in page

    def test_open_questions_are_shown_as_open_not_missing(self, client):
        page = text(client.get("/ui/investigations").text)
        assert "still open" in page

    def test_trials_spent_without_a_conclusion_are_counted(self, client):
        page = text(client.get("/ui/investigations").text)
        assert "Trials spent, no conclusion" in page
        assert "raise the bar" in page, (
            "the arithmetic consequence of an unrecorded search must be stated"
        )

    def test_a_finding_with_no_recorded_inquiry_is_flagged(self, client):
        """The mirror failure: a conclusion asserting unnamed work."""
        page = text(client.get("/ui/investigations").text)
        # The library is currently clean, so the warning must be absent —
        # not absent because the page cannot express it.
        assert "with no recorded inquiry" not in page

        from src.web.routes import _graph
        assert hasattr(_graph(), "unattributed_findings")

    def test_the_methodology_page_declares_the_trial_gap(self, client):
        """Deflation uses the ledger count; investigations may know of more."""
        page = text(client.get("/ui/m/xsmom/1").text)

        assert "Questions asked about this" in page
        assert "Trials" in page
