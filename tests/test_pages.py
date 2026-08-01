"""Pages arrange prepared values; they do not derive them.

Every test here injects a view model whose values are deliberately *at odds*
with what a template could plausibly infer from the underlying names — a
relation called REFUTES carrying ADVISORY, a boundary marked comparable that
still lists blockers, a step called Errata sitting clear. If the page renders
what it was given, it is composing. If it renders what "looks right", it has
grown a second opinion about meaning, and two opinions about meaning is the
failure mode this whole architecture exists to prevent.

The point is not that these combinations are correct. It is that the template
must not be the thing that decides they are wrong.
"""
from __future__ import annotations

import re

import pytest

from src.web.chain import Adversity, ChainState, Domain, Link, State
from src.web.eligibility import Gate, PerformanceVisualEligibility
from src.web.graph import TimelineSegment, TimelineViewModel
from src.web.routes import TEMPLATES
from src.web.semantics import RelationSemantics, resolve
from src.web.chain import Adversity as A


def render(template: str, **ctx) -> str:
    return TEMPLATES.env.get_template(template).render(**ctx)


def visible(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


class TestChainRendersItsPayload:
    def test_errata_step_renders_clear_when_the_payload_says_clear(self):
        """The word "errata" must not make the template decide it is adverse."""
        chain = ChainState(subject="x", links=(
            Link(step="Errata", domain=Domain.JUDGMENT, state=State.OK,
                 value="4 affecting lineage", summary="counted, not adverse",
                 adversity=Adversity.NONE),
        ))
        html = render("_chain.html", chain=chain)

        assert 'class="ok"' in html
        assert "4 affecting lineage" in visible(html)
        assert "block" not in html.split("<ol>")[1]

    def test_publication_step_renders_the_state_it_is_given(self):
        """A value reading BLOCK with state OK must render OK."""
        chain = ChainState(subject="x", links=(
            Link(step="Publication", domain=Domain.JUDGMENT, state=State.OK,
                 value="BLOCK", adversity=Adversity.NONE),
        ))
        html = render("_chain.html", chain=chain)

        assert 'class="ok"' in html, "the template inferred state from the value"
        assert "BLOCK" in visible(html)

    def test_headline_is_taken_verbatim(self):
        chain = ChainState(subject="x", links=(
            Link(step="Run", domain=Domain.EXECUTION, state=State.BLOCK,
                 value="—", adversity=Adversity.NONE),
        ))
        # BLOCK state with no adversity yields "Clear across the chain": the
        # two channels are independent by design, and the page must not reconcile.
        assert "Clear across the chain" in visible(render("_chain.html", chain=chain))

    def test_glyph_and_table_render_the_same_payload(self):
        chain = ChainState(subject="hrp@3", links=(
            Link(step="Claims", domain=Domain.REASONING, state=State.WARN, value="2"),
            Link(step="Run", domain=Domain.EXECUTION, state=State.BLOCK, value="—"),
        ))
        glyph, table = render("_glyph.html", chain=chain), render("_chain.html", chain=chain)

        assert glyph.count("◐") == 1 and glyph.count("✕") == 1
        assert 'class="warn"' in table and 'class="block"' in table


class TestRelationBadgeRendersItsPayload:
    def test_a_blocking_sounding_relation_renders_the_declared_effect(self):
        """REFUTES sounds terminal. The payload says advisory; advisory wins."""
        rel = RelationSemantics(
            relation="REFUTES", label="refutes", effect=A.ADVISORY,
            direction=resolve("REFUTES").direction,
            summary_template="{source} refutes {target}.",
        )
        html = render("_relation.html", rel=rel, source="evidence/e1", target="claim/c1")

        assert "relation-advisory" in html
        assert "relation-blocking" not in html

    def test_label_comes_from_the_payload_not_the_relation_name(self):
        rel = RelationSemantics(
            relation="INVALIDATES_RESULTS_OF", label="narrows the scope of",
            effect=A.NONE, direction=resolve("INVALIDATES_RESULTS_OF").direction,
            summary_template="{source} narrows {target}.",
        )
        text = visible(render("_relation.html", rel=rel, source="a", target="b"))

        assert "narrows the scope of" in text
        assert "invalidates" not in text.lower()


class TestTimelineRendersItsPayload:
    def test_a_comparable_segment_with_blockers_still_draws_no_wall(self):
        """The engine owns the verdict; the template owns none of it.

        A template that drew a wall wherever `blockers` was non-empty would be
        authoring comparability decisions in Jinja, which is exactly the split
        `TimelineSegment` documents against.
        """
        t = TimelineViewModel(
            concept="hrp",
            versions=[{"version_id": "methodology/hrp@1", "version": 1, "concept": "hrp",
                       "deprecated": False, "errata": 0, "latest": False},
                      {"version_id": "methodology/hrp@2", "version": 2, "concept": "hrp",
                       "deprecated": False, "errata": 0, "latest": True}],
            segments=(TimelineSegment(
                from_version="methodology/hrp@1", to_version="methodology/hrp@2",
                comparable=True, blockers=("weight bound max 1.0 -> 0.25",),
            ),),
        )
        html = render("_timeline.html", t=t)

        assert "span-ok" in html
        assert 'class="wall"' not in html
        assert "not comparable" not in visible(html)

    def test_an_incomparable_segment_with_no_blockers_still_draws_a_wall(self):
        t = TimelineViewModel(
            concept="hrp",
            versions=[{"version_id": "methodology/hrp@1", "version": 1, "concept": "hrp",
                       "deprecated": False, "errata": 0, "latest": False},
                      {"version_id": "methodology/hrp@2", "version": 2, "concept": "hrp",
                       "deprecated": False, "errata": 0, "latest": True}],
            segments=(TimelineSegment(
                from_version="methodology/hrp@1", to_version="methodology/hrp@2",
                comparable=False, blockers=(),
            ),),
        )
        html = render("_timeline.html", t=t)

        assert 'class="wall"' in html
        assert "span-ok" not in html


class TestEligibilityRendersItsPayload:
    def test_gates_render_the_verdict_they_carry(self):
        el = PerformanceVisualEligibility(gates=(
            Gate("COMPARABLE", "Versions are directly comparable", True,
                 "3 blocking contract differences"),
            Gate("PUBLICATION_SURFACE", "Publication permits this surface", False, "ALLOW"),
        ))
        html = render("_eligibility.html", el=el)

        rows = re.findall(r'class="gate gate-(ok|fail)"', html)
        assert rows == ["ok", "fail"], (
            "the template read the detail text instead of the passed flag"
        )

    def test_headline_is_not_recomputed_from_the_gates(self):
        el = PerformanceVisualEligibility(gates=(
            Gate("COMPARABLE", "Versions are directly comparable", False),
        ))
        assert el.headline() in visible(render("_eligibility.html", el=el))

    def test_all_gates_passing_removes_the_explanatory_note(self):
        el = PerformanceVisualEligibility(gates=(
            Gate("COMPARABLE", "Versions are directly comparable", True),
        ))
        text = visible(render("_eligibility.html", el=el))

        assert "Performance comparison available" in text
        assert "is itself a claim" not in text


class TestGraphRendersItsPayload:
    """The projection decides effect and directness; the SVG only draws them."""

    @staticmethod
    def _vm(relation: str, directness):
        from src.web.graph import GraphEdge, GraphNode, GraphViewModel

        return GraphViewModel(
            projection_type="test",
            root_ids=("a",),
            nodes=(GraphNode(id="a", artifact_type="finding", label="finding/f1"),
                   GraphNode(id="b", artifact_type="methodology",
                             label="methodology/hrp@1", group="methodology")),
            edges=(GraphEdge(source="a", target="b", relation_type=relation,
                             directness=directness),),
        )

    def test_a_constructive_relation_does_not_render_as_adverse(self):
        """MOTIVATED created its target. Drawing it red would assert the
        opposite of what happened."""
        from src.web.graph import Directness

        html = render("_graph.html", g=self._vm("MOTIVATED", Directness.INHERITED))

        assert "edge-none" in html and "edge-inherited" in html
        assert "edge-blocking" not in html
        assert "stroke-dasharray" in html, "directness lost its own visual channel"

    def test_table_and_svg_carry_identical_semantic_keys(self):
        from src.web.graph import Directness

        html = render("_graph.html", g=self._vm("REFUTES", Directness.DIRECT))

        assert set(re.findall(r'data-key="([^"]+)"', html)) == \
               set(re.findall(r'data-edge="([^"]+)"', html))


class TestAdversityStylingMatchesTheDeclaration:
    """A page may not style something as a blocker that the chain calls advisory.

    This is the flattening the `Adversity` enum was introduced to stop. A chip is
    a small thing, but "Not comparable" in blocker red teaches a reader that
    minting an honest new version is a failure, which is the opposite of the
    discipline the platform is trying to create.
    """

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api
        import src.web.routes as routes
        from src.ledger import Ledger

        ledger = Ledger(tmp_path / "styling.db")
        monkeypatch.setattr(api, "_ledger", ledger)
        monkeypatch.setattr(routes, "Ledger", lambda *a, **k: ledger)
        api._bootstrap()
        return TestClient(api.app)

    def test_incomparability_is_not_styled_as_a_blocker(self, client):
        html = client.get("/ui/m/hrp/3").text
        for fragment in re.findall(r'<[^>]*chip[^>]*>[^<]*Not comparable[^<]*<', html):
            assert "chip block" not in fragment

    def test_errata_are_not_styled_as_blockers(self, client):
        """Errata are AFFECTING: an adverse fact about history, not a gate."""
        html = client.get("/ui/").text
        for fragment in re.findall(r'<[^>]*chip[^>]*>[^<]*errata[^<]*<', html):
            assert "chip block" not in fragment

    def test_the_status_chip_has_one_definition(self):
        """Claim status is derived, so its rendering needs one author.

        Pages may still *branch* on a status — claims.html decides whether to
        warn about dependents — but only the macro may decide what a status
        looks like.
        """
        from pathlib import Path

        templates = Path("src/web/templates")
        offenders = [
            p.name for p in templates.glob("*.html")
            if p.name != "_status_chip.html"
            and any(f'"{s}"' in p.read_text() or f"'{s}'" in p.read_text()
                    for s in ("SUPPORTED", "QUALIFIED"))
        ]
        assert offenders == [], (
            f"claim status is styled independently in {offenders}"
        )

    def test_the_macro_maps_every_declared_status(self):
        from src.knowledge.artifacts import ClaimStatus

        macro = (
            __import__("pathlib").Path("src/web/templates/_status_chip.html").read_text()
        )
        for status in ClaimStatus:
            assert status.value in macro or status.value == "UNASSESSED", (
                f"{status.value} would render with no class at all"
            )


class TestTheDesignSystemIsSharedNotCopied:
    """Tokens are one file; layout is not shared at all.

    Two surfaces had independently chosen what "ok" looks like — #1f7a4d in the
    library and #4ba36b in the workspace. That is the visual-semantics defect
    expressed in CSS: a reader who learns a colour means "clear" in one half of
    the product has to learn it again in the other, and will eventually conclude
    they mean different things.
    """

    def test_only_one_file_defines_the_tokens(self):
        from pathlib import Path

        definers = [
            p for p in Path("src").rglob("*.html")
            if "--ok:" in p.read_text() or "--ok :" in p.read_text()
        ]
        assert [p.name for p in definers] == ["_tokens.html"], (
            f"status colour is defined independently in {[str(p) for p in definers]}"
        )

    def test_both_surfaces_load_the_same_tokens(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api
        import src.web.routes as web_routes
        import src.workspace.routes as workspace_routes
        from src.ledger import Ledger
        from src.workspace.store import WorkspaceStore

        ledger = Ledger(tmp_path / "public.db")
        monkeypatch.setattr(api, "_ledger", ledger)
        monkeypatch.setattr(web_routes, "Ledger", lambda *a, **k: ledger)
        monkeypatch.setattr(workspace_routes, "_store",
                            lambda: WorkspaceStore(tmp_path / "w.db"))
        api._bootstrap()
        client = TestClient(api.app)

        library = client.get("/ui/").text
        workspace = client.get("/workspace/").text
        for token in ("--ok:", "--warn:", "--block:", "--accent:", "--step-0:"):
            assert token in library and token in workspace

    def test_layout_is_not_shared(self):
        """Sharing layout would make one surface wrong for its audience."""
        from pathlib import Path

        library = Path("src/web/templates/base.html").read_text()
        workspace = Path("src/workspace/templates/base.html").read_text()

        assert ".chain {" in library
        assert ".private-badge" in workspace
        assert ".private-badge" not in library
