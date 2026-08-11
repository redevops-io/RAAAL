"""Reasoning as versioned artifacts.

The tests hold three properties the design depends on:

* claims are **referenced**, not embedded, so one claim can serve two
  methodologies and be contradicted without editing either;
* claim status is **derived** from evidence, never stored, so it cannot drift
  from the evidence that justifies it;
* assumption dependency is a **graph query**, not a string search, including the
  dependencies a methodology inherits through a claim rather than declaring.
"""
from __future__ import annotations

import pytest

from src.knowledge import (
    Assumption,
    AssumptionKind,
    AssumptionRegistry,
    Claim,
    ClaimRegistry,
    ClaimStatus,
    Evidence,
    EvidenceKind,
    EvidenceRegistry,
    KnowledgeGraph,
    Realization,
    Stance,
    assess_claim,
)
from src.methodology import MethodologyRegistry


@pytest.fixture
def graph():
    return KnowledgeGraph(
        MethodologyRegistry().load_all(),
        ClaimRegistry().load_all(),
        AssumptionRegistry().load_all(),
        EvidenceRegistry().load_all(),
    )


class TestAddressability:
    def test_claims_have_stable_and_versioned_ids(self):
        claim = ClaimRegistry().get("hrp-outperforms-mvo-out-of-sample", 1)
        assert claim.concept_id == "claim/hrp-outperforms-mvo-out-of-sample"
        assert claim.artifact_id == "claim/hrp-outperforms-mvo-out-of-sample@1"
        assert claim.content_hash

    def test_assumptions_have_ids_and_hashes(self):
        a = AssumptionRegistry().get("sample-covariance", 1)
        assert a.artifact_id == "assumption/sample-covariance@1"
        assert a.content_hash

    def test_changing_a_statement_changes_claim_identity(self):
        a = Claim(name="x", version=1, statement="A")
        b = Claim(name="x", version=1, statement="B")
        assert a.content_hash != b.content_hash

    def test_methodologies_reference_rather_than_embed(self):
        """The whole point: a claim can be shared, contradicted and superseded."""
        m = MethodologyRegistry().get("hrp", 3)
        assert m.claims_ref, "hrp@3 should reference claims"
        assert all(r.startswith("claim/") for r in m.claims_ref)
        assert all(r.startswith("assumption/") for r in m.assumptions_ref)

    def test_references_are_part_of_methodology_identity(self):
        """Changing what a methodology rests on changes what it is."""
        from src.methodology.spec import Methodology, OutputContract

        base = Methodology(
            concept="x", version=1, title="x", objective="",
            contract=OutputContract(universe=("SPY",), rebalance_frequency="5B"),
        )
        with_claim = Methodology(
            concept="x", version=1, title="x", objective="",
            contract=OutputContract(universe=("SPY",), rebalance_frequency="5B"),
            claims_ref=("claim/something@1",),
        )
        assert base.content_hash != with_claim.content_hash


class TestClaimStatusIsDerived:
    def _claim(self, **kw):
        return Claim(name="c", version=1, statement="s", **kw)

    def _evidence(self, stance, strength="moderate", valid_to=None):
        return Evidence(
            name="e", version=1, kind=EvidenceKind.PAPER, about="claim/c@1",
            stance=stance, summary="", strength=strength, valid_to=valid_to,
        )

    def test_no_evidence_is_unassessed(self):
        assert assess_claim(self._claim(), []).status is ClaimStatus.UNASSESSED

    def test_support_alone_is_supported(self):
        result = assess_claim(self._claim(), [self._evidence(Stance.SUPPORTS)])
        assert result.status is ClaimStatus.SUPPORTED

    def test_qualification_narrows_rather_than_refutes(self):
        result = assess_claim(self._claim(), [
            self._evidence(Stance.SUPPORTS),
            self._evidence(Stance.QUALIFIES),
        ])
        assert result.status is ClaimStatus.QUALIFIED

    def test_weak_dissent_against_strong_support_is_contested_not_refuted(self):
        """A single weak objection should not overturn well-supported work."""
        result = assess_claim(self._claim(), [
            self._evidence(Stance.SUPPORTS, strength="strong"),
            self._evidence(Stance.CONTRADICTS, strength="weak"),
        ])
        assert result.status is ClaimStatus.CONTESTED

    def test_strong_contradiction_without_support_refutes(self):
        result = assess_claim(
            self._claim(), [self._evidence(Stance.CONTRADICTS, strength="strong")]
        )
        assert result.status is ClaimStatus.REFUTED

    def test_invalidated_evidence_is_excluded_not_deleted(self):
        """Bi-temporal: evidence is invalidated, and stops counting, but persists."""
        stale = self._evidence(Stance.CONTRADICTS, strength="strong", valid_to="2020-01-01")
        result = assess_claim(self._claim(), [self._evidence(Stance.SUPPORTS), stale])
        assert result.status is ClaimStatus.SUPPORTED

    def test_superseded_claim_reports_superseded(self):
        claim = self._claim(superseded_by="claim/c@2")
        result = assess_claim(claim, [self._evidence(Stance.SUPPORTS)])
        assert result.status is ClaimStatus.SUPERSEDED

    def test_status_is_not_a_stored_field(self):
        """Storing it would let it drift from the evidence that justifies it."""
        assert "status" not in Claim(name="c", version=1, statement="s").to_json()


class TestShippedKnowledge:
    def test_platform_result_contradicts_its_own_claim(self, graph):
        """The demonstration: the platform refuted a claim its own methodology
        rested on, using its own evaluation as the contradicting evidence."""
        claim = ClaimRegistry().get("hrp-diversifies-without-constraints", 1)
        result = assess_claim(claim, graph.evidence)

        assert result.status is ClaimStatus.REFUTED
        assert result.contradicting
        assert result.contradicting[0].kind is EvidenceKind.PLATFORM_RESULT

    def test_source_claim_is_qualified_not_refuted(self, graph):
        """The capped result narrows where the source claim holds; it does not
        overturn the paper."""
        claim = ClaimRegistry().get("hrp-outperforms-mvo-out-of-sample", 1)
        result = assess_claim(claim, graph.evidence)

        assert result.status is ClaimStatus.QUALIFIED
        assert result.supporting and result.qualifying

    def test_every_assumption_records_what_goes_wrong(self):
        for a in AssumptionRegistry().load_all():
            assert a.risk.strip(), f"{a.artifact_id} does not say what breaks if false"

    def test_every_assumption_names_where_it_takes_effect(self):
        for a in AssumptionRegistry().load_all():
            assert a.realized_by, f"{a.artifact_id} is not linked to any artifact field"

    def test_shipped_assumptions_are_all_validated(self):
        """An assumption nothing checks is an assertion. Historically that is
        exactly where each erratum came from."""
        unvalidated = [a.artifact_id for a in AssumptionRegistry().load_all() if not a.is_validated]
        assert not unvalidated, f"unvalidated assumptions: {unvalidated}"


class TestGraphQueries:
    def test_direct_assumption_dependency(self, graph):
        affected = graph.methodologies_depending_on_assumption(
            "assumption/sample-covariance@1"
        )
        assert {m.version_id for m in affected} == {
            "methodology/hrp@1", "methodology/hrp@2", "methodology/hrp@3"
        }

    def test_dependency_inherited_through_a_claim(self, graph):
        """hrp@1 and hrp@2 do not declare nyse-sessions; they inherit it via the
        claim they reference. A search of their own lists would miss them."""
        m1 = MethodologyRegistry().get("hrp", 1)
        assert "assumption/nyse-sessions@1" not in m1.assumptions_ref

        affected = graph.methodologies_depending_on_assumption(
            "assumption/nyse-sessions@1"
        )
        assert "methodology/hrp@1" in {m.version_id for m in affected}

    def test_unknown_assumption_returns_empty(self, graph):
        assert graph.methodologies_depending_on_assumption("assumption/nope@1") == []

    def test_impact_of_a_claim_change(self, graph):
        """The query Discovery needs when new evidence arrives."""
        impact = graph.impact_of_claim_change(
            "claim/hrp-diversifies-without-constraints@1"
        )
        assert impact["found"] is True
        assert impact["status"] == "REFUTED"
        assert impact["affected_count"] >= 2

    def test_assumptions_for_methodology_include_inherited(self, graph):
        m = MethodologyRegistry().get("hrp", 1)
        ids = {a.artifact_id for a in graph.assumptions_for_methodology(m)}
        assert "assumption/sample-covariance@1" in ids     # declared
        assert "assumption/nyse-sessions@1" in ids          # inherited via claim

    def test_contested_claims_are_listed(self, graph):
        contested = graph.contested_claims()
        assert any(c.status is ClaimStatus.REFUTED for c in contested)

    def test_unvalidated_assumption_register(self):
        """The standing register of where the next defect probably is."""
        graph = KnowledgeGraph(
            [], [],
            [Assumption(name="x", version=1, statement="s", kind=AssumptionKind.DATA)],
            [],
        )
        assert len(graph.unvalidated_assumptions()) == 1


class TestEvidenceDeclaresStance:
    def test_stance_lives_on_evidence_not_on_the_claim(self):
        """So recording disagreement never requires editing the thing disagreed
        with — otherwise dissent is gated on the claim's owner."""
        claim = Claim(name="c", version=1, statement="s")
        assert not hasattr(claim, "supported_by")
        assert not hasattr(claim, "contradicted_by")

        e = EvidenceRegistry().get("raaal-cash-degeneracy-2026", 1)
        assert e.about.startswith("claim/")
        assert e.stance is Stance.CONTRADICTS

    def test_evidence_records_what_produced_it(self):
        e = EvidenceRegistry().get("raaal-cash-degeneracy-2026", 1)
        assert "methodology/hrp@1" in e.produced_by
        assert "protocol/standard@1" in e.produced_by


class TestFindings:
    """A finding is the conclusion of an investigation — not a claim, not evidence."""

    @pytest.fixture
    def graph(self):
        from src.knowledge import FindingRegistry

        return KnowledgeGraph(
            MethodologyRegistry().load_all(),
            ClaimRegistry().load_all(),
            AssumptionRegistry().load_all(),
            EvidenceRegistry().load_all(),
            FindingRegistry().load_all(),
        )

    def test_findings_are_addressable_and_hashed(self):
        from src.knowledge import FindingRegistry

        f = FindingRegistry().get("hrp-degenerates-to-cash-proxy", 1)
        assert f.artifact_id == "finding/hrp-degenerates-to-cash-proxy@1"
        assert f.content_hash

    def test_a_finding_spans_multiple_artifact_kinds(self, graph):
        """The reason it cannot be modelled as a claim or a piece of evidence:
        one conclusion, many things changed."""
        from src.knowledge import FindingRegistry

        f = FindingRegistry().get("hrp-degenerates-to-cash-proxy", 1)
        assert f.targets("claim"), "should touch claims"
        assert f.targets("methodology"), "should touch methodologies"
        assert f.targets("assumption"), "should touch assumptions"
        assert len(f.impacts) >= 5

    def test_impacts_are_typed_not_prose(self):
        from src.knowledge import FindingRegistry, ImpactRelation

        f = FindingRegistry().get("hrp-degenerates-to-cash-proxy", 1)
        relations = {i.relation for i in f.impacts}
        assert ImpactRelation.REFUTES in relations
        assert ImpactRelation.MOTIVATED in relations
        assert ImpactRelation.INVALIDATES_RESULTS_OF in relations

    def test_why_a_methodology_exists_is_a_query(self, graph):
        """Previously answerable only by reading change_rationale prose."""
        motivating = [
            f for f in graph.findings_affecting("methodology/hrp@3")
            if any(
                i.target == "methodology/hrp@3" and i.relation.value == "MOTIVATED"
                for i in f.impacts
            )
        ]
        assert len(motivating) == 1
        assert motivating[0].name == "hrp-degenerates-to-cash-proxy"

    def test_what_was_concluded_about_a_version(self, graph):
        affecting = graph.findings_affecting("methodology/hrp@1")
        assert affecting
        assert any(
            i.relation.value == "INVALIDATES_RESULTS_OF"
            for f in affecting for i in f.impacts if i.target == "methodology/hrp@1"
        )

    def test_provenance_resolves_evidence_and_targets(self, graph):
        from src.knowledge import FindingRegistry

        f = FindingRegistry().get("hrp-degenerates-to-cash-proxy", 1)
        p = graph.finding_provenance(f)

        assert len(p["evidence"]) == 3
        assert "claim/hrp-diversifies-without-constraints@1" in p["impacted_claims"]
        assert "methodology/hrp@3" in p["impacted_methodologies"]

    def test_every_shipped_finding_cites_evidence(self, graph):
        """A finding with no evidence is an opinion."""
        assert graph.unevidenced_findings() == []

    def test_every_concluded_finding_records_a_resolution(self):
        from src.knowledge import FindingRegistry, FindingStatus

        for f in FindingRegistry().load_all():
            if f.status is FindingStatus.CONCLUDED:
                assert f.resolution.strip(), (
                    f"{f.artifact_id} concluded without recording what was done"
                )

    def test_finding_is_not_a_claim_or_evidence(self):
        """Distinct type, distinct namespace."""
        from src.knowledge import Claim, Evidence, Finding

        assert Finding is not Claim and Finding is not Evidence
        f = Finding(
            name="x", version=1, statement="s",
            status=__import__("src.knowledge", fromlist=["FindingStatus"]).FindingStatus.OPEN,
        )
        assert f.artifact_id.startswith("finding/")

    def test_provisional_findings_are_listed(self):
        from src.knowledge import Finding, FindingStatus

        graph = KnowledgeGraph([], [], [], [], [
            Finding(name="o", version=1, statement="s", status=FindingStatus.OPEN)
        ])
        assert len(graph.provisional_findings()) == 1
