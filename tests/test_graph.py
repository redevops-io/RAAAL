"""Graph projections — bidirectional equivalence and edge-level semantics.

Milestone 1 proved artifact *state* can be scanned honestly. These tests hold the
harder property: artifact *relationships* rendered visually without collapsing
typed impacts into generic good/bad edges.
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from src.knowledge import (
    AssumptionRegistry, ClaimRegistry, EvidenceRegistry, FindingRegistry,
    KnowledgeGraph, assess_claim,
)
from src.ledger import Ledger
from src.methodology import MethodologyRegistry
from src.web.chain import Adversity, State
from src.web.graph import (
    Directness, GraphEdge, GraphNode, GraphViewModel,
    assumption_dependency_projection, claim_stance_projection,
    finding_impact_projection,
)


@pytest.fixture
def graph():
    return KnowledgeGraph(
        MethodologyRegistry().load_all(), ClaimRegistry().load_all(),
        AssumptionRegistry().load_all(), EvidenceRegistry().load_all(),
        FindingRegistry().load_all(),
    )


@pytest.fixture
def impact(graph):
    f = FindingRegistry().get("hrp-degenerates-to-cash-proxy", 1)
    return finding_impact_projection(f, graph.evidence)


@pytest.fixture
def client(tmp_path, monkeypatch):
    import src.api as api
    import src.web.routes as routes

    ledger = Ledger(tmp_path / "g.db")
    monkeypatch.setattr(api, "_ledger", ledger)
    monkeypatch.setattr(routes, "Ledger", lambda *a, **k: ledger)
    api._bootstrap()
    return TestClient(api.app)


class TestEdgesOwnImpact:
    """A finding is not globally adverse. Its outgoing edges are, or are not."""

    def test_one_finding_carries_opposite_effects(self, impact):
        by_relation = {e.relation_type: e.effect for e in impact.edges}

        assert by_relation["INVALIDATES_RESULTS_OF"] is Adversity.BLOCKING
        assert by_relation["REFUTES"] is Adversity.BLOCKING
        assert by_relation["QUALIFIES"] is Adversity.ADVISORY
        assert by_relation["MOTIVATED"] is Adversity.NONE, (
            "a finding that motivated a methodology is not adverse to it"
        )
        assert by_relation["INTRODUCED"] is Adversity.NONE

    def test_node_adversity_does_not_override_edges(self, impact):
        """The relation-level truth must survive the node-level summary."""
        root = next(n for n in impact.nodes if n.artifact_type == "finding")
        effects = {e.effect for e in impact.edges if e.source == root.id}
        assert len(effects) > 1, "one finding should carry several distinct effects"

    def test_undeclared_relation_raises_rather_than_defaulting(self):
        """A silent default is how an adverse edge comes to render as benign."""
        from src.web.semantics import UndeclaredRelation

        with pytest.raises(UndeclaredRelation, match="no visual semantics"):
            GraphEdge(source="a", target="b", relation_type="INVENTED").effect

    def test_every_declared_relation_has_an_effect(self):
        from src.web.semantics import RELATION_SEMANTICS

        for relation, semantics in RELATION_SEMANTICS.items():
            assert isinstance(semantics.effect, Adversity)


class TestBidirectionalEquivalence:
    """Neither rendering may silently omit an awkward relationship."""

    def _edge_key(self, row) -> tuple:
        return (row["source"], row["relation_type"], row["target"])

    def test_every_edge_appears_in_the_fallback(self, impact):
        rows = {self._edge_key(r) for r in impact.fallback_rows()}
        edges = {(e.source, e.relation_type, e.target) for e in impact.edges}
        assert rows == edges

    def test_fallback_preserves_effect_and_directness(self, impact):
        for row, edge in zip(impact.fallback_rows(), impact.edges):
            assert row["effect"] == edge.effect.value
            assert row["directness"] == edge.directness.value

    def test_layout_places_every_declared_edge(self, impact):
        """A layout bug must not silently drop an edge the table still lists."""
        placed = impact.layout()["edges"]
        assert len(placed) == len(impact.edges)

        placed_keys = {(e["source"], e["relation_type"], e["target"]) for e in placed}
        row_keys = {self._edge_key(r) for r in impact.fallback_rows()}
        assert placed_keys == row_keys

    def test_rendered_svg_and_table_agree_both_ways(self, client):
        """End-to-end, in both directions: every drawn line has a table row and
        every table row has a drawn line."""
        html = client.get("/ui/findings").text

        drawn = len(re.findall(r'<line [^>]*class="edge ', html))
        rows = len(re.findall(r'data-edge="', html))

        assert drawn > 0, "no edges drawn"
        assert rows > 0, "no fallback rows rendered"
        assert drawn == rows, (
            f"diagram drew {drawn} relationships, table listed {rows} — "
            "one rendering is omitting relationships the other shows"
        )

    def test_accessibility_summary_counts_every_edge(self, impact):
        summary = impact.accessibility_summary()
        total = sum(
            int(n) for n in re.findall(r"(\d+) [a-z]", summary.split("linked by")[1])
        )
        assert total == len(impact.edges)


class TestOmissionIsDeclared:
    def test_omitted_count_is_reported(self):
        """A view showing 12 of 40 dependents must say so."""
        assumption = AssumptionRegistry().get("sample-covariance", 1)
        methodologies = MethodologyRegistry().load_all()

        projection = assumption_dependency_projection(
            assumption, direct=methodologies, inherited=[], limit=2
        )
        assert projection.omitted_count == len(methodologies) - 2

    def test_nothing_omitted_reports_zero(self, impact):
        assert impact.omitted_count == 0

    def test_omission_surfaces_in_the_summary(self):
        assumption = AssumptionRegistry().get("sample-covariance", 1)
        projection = assumption_dependency_projection(
            assumption, direct=MethodologyRegistry().load_all(), inherited=[], limit=1
        )
        assert "omitted" in projection.accessibility_summary()


class TestDirectnessIsItsOwnChannel:
    def test_inherited_dependency_is_marked(self, graph):
        """hrp@1 inherits nyse-sessions via a claim; it does not declare it."""
        assumption = AssumptionRegistry().get("nyse-sessions", 1)
        direct = [
            m for m in graph.methodologies
            if assumption.artifact_id in getattr(m, "assumptions_ref", ())
        ]
        inherited = [
            m for m in graph.methodologies_depending_on_assumption(assumption.artifact_id)
            if assumption.artifact_id not in getattr(m, "assumptions_ref", ())
        ]
        projection = assumption_dependency_projection(assumption, direct, inherited)

        kinds = {e.directness for e in projection.edges}
        assert Directness.INHERITED in kinds
        assert Directness.DIRECT in kinds

    def test_directness_is_not_conflated_with_effect(self, graph):
        """Line style means directness; it must not also mean advisory."""
        assumption = AssumptionRegistry().get("nyse-sessions", 1)
        projection = assumption_dependency_projection(
            assumption,
            direct=[],
            inherited=graph.methodologies_depending_on_assumption(assumption.artifact_id),
        )
        for edge in projection.edges:
            assert edge.directness is Directness.INHERITED
            assert edge.effect is Adversity.NONE, (
                "inheriting a dependency is not itself adverse"
            )


class TestClaimStanceKeepsItsOwnSemantics:
    def test_stance_not_traffic_lights(self, graph):
        """Refuting and narrowing are different acts; both must stay visible."""
        claim = ClaimRegistry().get("hrp-outperforms-mvo-out-of-sample", 1)
        projection = claim_stance_projection(assess_claim(claim, graph.evidence))

        relations = {e.relation_type for e in projection.edges}
        assert "SUPPORTS" in relations
        assert "QUALIFIES" in relations

    def test_refuted_claim_shows_the_contradiction(self, graph):
        claim = ClaimRegistry().get("hrp-diversifies-without-constraints", 1)
        projection = claim_stance_projection(assess_claim(claim, graph.evidence))

        assert any(e.relation_type == "CONTRADICTS" for e in projection.edges)
        root = next(n for n in projection.nodes if n.artifact_type == "claim")
        assert root.state is State.BLOCK


class TestProjectionIsNotTheGraph:
    def test_projection_records_what_it_is(self, impact):
        payload = impact.to_json()
        assert payload["projection_type"] == "finding_impact"
        assert payload["root_ids"]
        assert payload["grouping_rules"]
        assert payload["source_graph_version"]

    def test_projection_carries_all_four_renderings(self, impact):
        payload = impact.to_json()
        for key in ("nodes", "edges", "fallback_rows",
                    "accessibility_summary", "compact_summary"):
            assert key in payload


class TestPagesRenderGraphs:
    def test_findings_page_draws_the_impact_graph(self, client):
        html = client.get("/ui/findings").text
        assert 'data-projection="finding_impact"' in html
        assert "<svg" in html

    def test_claims_page_draws_stance_and_dependency(self, client):
        html = client.get("/ui/claims").text
        assert 'data-projection="claim_stance"' in html
        assert 'data-projection="assumption_dependency"' in html

    def test_graphs_are_not_performance_charts(self, client):
        """The Phase 1-3 guard still stands: graph visuals only."""
        for path in ("/ui/findings", "/ui/claims"):
            html = client.get(path).text.lower()
            for marker in ("<canvas", "chart.js", "plotly", "d3.min.js"):
                assert marker not in html


class TestSemanticsRegistry:
    """One declaration per relation, read by every component."""

    def test_every_graph_relation_has_semantics(self):
        from src.web.graph import version_timeline  # noqa: F401
        from src.web.semantics import RELATION_SEMANTICS, resolve

        for relation in RELATION_SEMANTICS:
            s = resolve(relation)
            assert s.label and s.summary_template

    def test_undeclared_relation_refuses_to_render(self):
        from src.web.semantics import UndeclaredRelation, resolve

        with pytest.raises(UndeclaredRelation, match="no visual semantics"):
            resolve("MADE_UP_RELATION")

    def test_effect_has_exactly_one_source(self, impact):
        """The graph must not keep its own copy of the effect mapping."""
        from src.web.semantics import effect_of

        for edge in impact.edges:
            assert edge.effect is effect_of(edge.relation_type)

    def test_language_is_consistent_across_components(self):
        """The same relation cannot read 'invalidates' here and 'affects' there."""
        from src.web.semantics import label_of

        assert label_of("INVALIDATES_RESULTS_OF") == "invalidates results of"
        assert label_of("MOTIVATED") == "motivated"

    def test_constructive_relations_are_documented_as_such(self):
        from src.web.semantics import resolve

        assert resolve("MOTIVATED").note, "why MOTIVATED is not adverse must be stated"
        assert resolve("CONTRADICTS").note


class TestSemanticEquivalence:
    """Counts agreeing while meanings differ would pass a count test and lie."""

    def test_canonical_key_covers_meaning_not_just_endpoints(self, impact):
        for edge in impact.edges:
            key = edge.semantic_key
            assert len(key) == 5
            assert key[3] == edge.effect.value
            assert key[4] == edge.directness.value

    def test_fallback_rows_expose_the_same_key(self, impact):
        row_keys = {tuple(r["semantic_key"].split("|")) for r in impact.fallback_rows()}
        assert row_keys == impact.semantic_keys()

    def test_layout_preserves_the_key(self, impact):
        placed = {tuple(e["semantic_key"].split("|")) for e in impact.layout()["edges"]}
        assert placed == impact.semantic_keys()

    def test_rendered_svg_and_table_share_semantic_keys(self, client):
        """End-to-end semantic equivalence, not merely equal counts."""
        html = client.get("/ui/findings").text

        drawn = set(re.findall(r'<line [^>]*data-key="([^"]+)"', html))
        listed = set(re.findall(r'data-edge="([^"]+)"', html))

        assert drawn, "diagram exposed no semantic keys"
        assert drawn == listed, (
            "diagram and table disagree about what the relationships mean"
        )


class TestOmissionReason:
    def test_reason_is_typed_not_implied(self):
        from src.web.graph import OmissionReason

        assumption = AssumptionRegistry().get("sample-covariance", 1)
        projection = assumption_dependency_projection(
            assumption, direct=MethodologyRegistry().load_all(), inherited=[], limit=1
        )
        assert projection.omission_reason is OmissionReason.DISPLAY_LIMIT

    def test_reasons_are_not_interchangeable(self):
        """A reader must know whether nodes are hidden for readability or access."""
        from src.web.graph import OmissionReason

        assert OmissionReason.DISPLAY_LIMIT is not OmissionReason.PERMISSION_FILTER
        assert len(set(OmissionReason)) == 4

    def test_reason_surfaces_in_the_accessibility_summary(self):
        assumption = AssumptionRegistry().get("sample-covariance", 1)
        projection = assumption_dependency_projection(
            assumption, direct=MethodologyRegistry().load_all(), inherited=[], limit=1
        )
        assert "display limit" in projection.accessibility_summary()


class TestVersionTimeline:
    """Walls are derived from the comparability engine, never authored."""

    def _timeline(self, concept="hrp"):
        from src.web.graph import version_timeline

        return version_timeline(concept, MethodologyRegistry().versions(concept))

    def test_walls_match_the_comparability_engine(self):
        timeline = self._timeline()
        versions = MethodologyRegistry().versions("hrp")

        for older, newer in zip(versions, versions[1:]):
            engine = newer.contract.breaks_compatibility_with(older.contract)
            segment = next(
                s for s in timeline.segments
                if s.from_version == older.version_id and s.to_version == newer.version_id
            )
            assert segment.comparable is (not engine)
            assert list(segment.blockers) == engine

    def test_hrp_has_two_walls(self):
        timeline = self._timeline()
        assert len(timeline.walls) == 2
        assert any("rebalance frequency" in b for w in timeline.walls for b in w.blockers)
        assert any("weight bound" in b for w in timeline.walls for b in w.blockers)

    def test_single_version_lineage_has_no_walls(self):
        assert self._timeline("xsmom").walls == []

    def test_layout_places_every_version_and_segment(self):
        timeline = self._timeline()
        layout = timeline.layout()
        assert len(layout["stops"]) == len(timeline.versions)
        assert len(layout["spans"]) == len(timeline.segments)

    def test_wall_links_to_the_comparability_verdict(self, client):
        html = client.get("/ui/m/hrp/3").text
        assert "compare?a=" in html
        assert "not comparable" in html.lower()

    def test_timeline_renders_on_the_methodology_page(self, client):
        html = client.get("/ui/m/hrp/3").text
        assert 'class="timeline"' in html
        assert "comparability wall" in html or "comparability walls" in html
