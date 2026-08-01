"""Chain state — one payload, two renderings.

The invariant under test is not aesthetic. If the glyph and the table can
disagree, the visual has become a second source of truth, and by the project's
own principle a second source always drifts.
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from src.ledger import Ledger
from src.web.chain import CHAIN_ORDER, ChainState, Domain, Link, State, build_chain_state


@pytest.fixture
def client(tmp_path, monkeypatch):
    import src.api as api
    import src.web.routes as routes

    ledger = Ledger(tmp_path / "chain.db")
    monkeypatch.setattr(api, "_ledger", ledger)
    monkeypatch.setattr(routes, "Ledger", lambda *a, **k: ledger)
    api._bootstrap()
    return TestClient(api.app)


def _state(**kw) -> ChainState:
    from src.methodology import MethodologyRegistry

    defaults = dict(subject="test", methodology=MethodologyRegistry().get("hrp", 3))
    defaults.update(kw)
    return build_chain_state(**defaults)


class TestEquivalence:
    """The visual must never infer or recompute a status."""

    def test_glyph_and_table_come_from_one_payload(self):
        chain = _state()
        glyph_symbols = [c for c in chain.glyph_text() if c not in " │"]
        table_symbols = [row["symbol"] for row in chain.as_table()]
        assert glyph_symbols == table_symbols

    def test_rendered_page_glyph_matches_rendered_table(self, client):
        """End-to-end: the HTML glyph and the HTML matrix row must agree."""
        html = client.get("/ui/").text

        matrix = re.search(r'<table class="matrix">.*?</table>', html, re.S)
        assert matrix, "status matrix did not render"

        rows = re.findall(
            r'methodology/hrp@3</a></td>(.*?)</tr>', matrix.group(0), re.S
        )
        assert rows, "hrp@3 row missing from the matrix"
        matrix_states = re.findall(r'class="dot dot-(\w+)"', rows[0])

        card = re.search(
            r'<span class="glyph"[^>]*>(.*?)</span>\s*<span class="mono muted">methodology/hrp@3',
            html, re.S,
        )
        assert card, "glyph missing from the hrp card"
        glyph_states = re.findall(r'class="dot dot-(\w+)"', card.group(1))

        assert glyph_states == matrix_states, (
            "glyph and matrix disagree — the visual is computing its own state"
        )

    def test_every_link_has_an_accessible_label(self):
        for link in _state().links:
            assert link.aria
            assert link.state.label in link.aria


class TestStateNotPresence:
    """A dot reports the condition of an artifact class, not that it exists."""

    def test_refuted_claim_blocks_even_though_claims_exist(self):
        from src.knowledge import (
            AssumptionRegistry, ClaimRegistry, EvidenceRegistry,
            FindingRegistry, KnowledgeGraph,
        )
        from src.methodology import MethodologyRegistry

        graph = KnowledgeGraph(
            MethodologyRegistry().load_all(), ClaimRegistry().load_all(),
            AssumptionRegistry().load_all(), EvidenceRegistry().load_all(),
            FindingRegistry().load_all(),
        )
        m = MethodologyRegistry().get("hrp", 1)
        claims = graph.claims_for_methodology(m)

        assert claims, "hrp@1 references claims"
        chain = _state(methodology=m, claims=claims)
        link = next(l for l in chain.links if l.step == "Claims")

        assert link.state is State.BLOCK, (
            "hrp@1 references a REFUTED claim — presence must not read as health"
        )

    def test_absent_is_distinct_from_clear(self):
        chain = _state(claims=(), assumptions=())
        claims = next(l for l in chain.links if l.step == "Claims")
        assert claims.state is State.ABSENT
        assert claims.state.symbol != State.OK.symbol


class TestScannability:
    def test_position_is_stable_across_rows(self):
        a = _state()
        b = _state(errata=[{"erratum_id": "x"}])
        assert [l.step for l in a.links] == [l.step for l in b.links]
        assert [l.step for l in a.links] == [s for s, _ in CHAIN_ORDER]

    def test_symbols_distinguish_without_colour(self):
        symbols = {s.symbol for s in State}
        assert len(symbols) == len(State), "two states share a symbol"

    def test_glyph_groups_the_three_domains(self):
        glyph = _state().glyph_text()
        assert glyph.count("│") == 2, "expected reasoning │ execution │ judgment"

    def test_worst_is_pessimistic(self):
        """One blocker outranks ten clear links."""
        chain = ChainState(
            subject="x",
            links=(
                Link("Findings", Domain.REASONING, State.OK, "0"),
                Link("Claims", Domain.REASONING, State.BLOCK, "1 refuted"),
                Link("Methodology", Domain.EXECUTION, State.OK, "m"),
            ),
        )
        assert chain.worst is State.BLOCK

    def test_headline_is_plain_language(self):
        from src.web.chain import Adversity

        chain = ChainState(
            subject="x",
            links=(
                Link("Claims", Domain.REASONING, State.BLOCK, "1 refuted",
                     adversity=Adversity.BLOCKING),
                Link("Policy", Domain.JUDGMENT, State.WARN, "WARN",
                     adversity=Adversity.ADVISORY),
            ),
        )
        assert chain.headline == "Blocked at claims"

    def test_headline_separates_blocking_from_affecting(self):
        """"Blocked at errata" asserted that errata are failures. An erratum is
        an adverse historical condition; it does not block on its own."""
        from src.web.chain import Adversity

        chain = ChainState(
            subject="x",
            links=(
                Link("Claims", Domain.REASONING, State.BLOCK, "1 refuted",
                     adversity=Adversity.BLOCKING),
                Link("Errata", Domain.JUDGMENT, State.WARN, "2 affecting lineage",
                     adversity=Adversity.AFFECTING),
            ),
        )
        assert chain.headline == "Blocked at claims · affected by 2 errata"

    def test_errata_presence_never_blocks(self):
        """The publication gate decides what may be shown; a published
        correction is a fact about history, not an active blocker."""
        from src.web.chain import Adversity

        chain = _state(errata=[{"erratum_id": "a"}, {"erratum_id": "b"}])
        errata = next(l for l in chain.links if l.step == "Errata")

        assert errata.adversity is Adversity.AFFECTING
        assert errata.state is not State.BLOCK
        assert "Blocked at errata" not in chain.headline

    def test_a_motivating_finding_is_not_adverse_as_a_blocker(self):
        """finding/hrp-degenerates-to-cash-proxy MOTIVATED hrp@3. Presence of a
        finding must not read as failure."""
        from src.web.chain import Adversity

        chain = _state(findings=[object()], invalidating=[])
        link = next(l for l in chain.links if l.step == "Findings")
        assert link.adversity is Adversity.ADVISORY

    def test_singular_noun_when_count_is_one(self):
        from src.web.chain import Adversity

        chain = ChainState(
            subject="x",
            links=(Link("Errata", Domain.JUDGMENT, State.WARN, "1 affecting lineage",
                        adversity=Adversity.AFFECTING),),
        )
        assert "1 erratum" in chain.headline or "1 errata" not in chain.headline

    def test_clear_chain_says_so(self):
        chain = ChainState(
            subject="x",
            links=(Link("Claims", Domain.REASONING, State.OK, "2"),),
        )
        assert "Clear" in chain.headline


class TestLibraryPage:
    def test_legend_is_present_and_sticky(self, client):
        html = client.get("/ui/").text
        assert 'class="legend"' in html
        assert "reasoning" in html and "execution" in html and "judgment" in html

    def test_matrix_lists_every_version(self, client):
        html = client.get("/ui/").text
        for version_id in (
            "methodology/hrp@1", "methodology/hrp@2",
            "methodology/hrp@3", "methodology/xsmom@1",
        ):
            assert version_id in html

    def test_headline_appears_beside_each_lineage(self, client):
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", client.get("/ui/").text))
        assert "Blocked at" in text or "Clear across the chain" in text

    def test_still_no_performance_charts(self, client):
        """The Phase 1–3 guard stands: graph visuals only."""
        html = client.get("/ui/").text.lower()
        for marker in ("<canvas", "chart.js", "plotly", "d3.min.js"):
            assert marker not in html
