"""§6 (evaluation result UX) and §7 (research ↔ evaluator continuity).

The invariant these guard, stated once:

    The result view is RESTRUCTURED into the four §6 decision-grouped sections
    (A interpreted strategy · B outcome · C evidence/reproducibility · D the
    analytical alternative) — but *nothing computed changes*. Every figure the
    page showed before still shows; the evaluation, the save, and the exact-save
    hash are untouched. The benchmark is a labelled alternative, never merged
    into the user's strategy or the saved artifact.

So the assertions are not "a page rendered". They are:

* the four sections are present and each pulls from the context that already fed
  the page — parameters into A, chart/figure into B, provenance into C,
  benchmarks into D;
* each parameter is labelled in §6's own terms (supplied / inferred / defaulted
  / unresolved) from the `state`/`author` the reading already carried;
* a result-changing unresolved dimension blocks the final figure;
* the benchmark renders in D as an alternative and is absent from A and from the
  saved plan; saving keeps the user's confirmed strategy (Gate 2's content hash);
* the evidence panel surfaces what provenance genuinely carries and marks — never
  fabricates — what it does not;
* the research surface offers a public-semantics-only move into `/evaluate`, and
  the evidence panel links back to a `/methodologies/{concept}` page;
* the same evaluation still produces the same figure and intent hash — no second
  parser, no model call added by the restructuring.
"""
from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

SENTENCE = "invest $500 monthly"
ANSWER = {"describe": SENTENCE, "answer_assets": "VTI"}
SUBJECT_HEADER = "x-test-subject"


# --- fixtures ---------------------------------------------------------------

def _runtime_env(monkeypatch, tmp_path):
    """A runtime deployment reading from recordings, with synthetic prices so a
    figure — and therefore benchmarks, a chart and provenance — is reachable."""
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-by-recordings")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/r.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)
    return deploy_context


@pytest.fixture
def client(monkeypatch, tmp_path):
    """No identity provider — evaluation is fully public here."""
    _runtime_env(monkeypatch, tmp_path)
    from src.api import app

    return TestClient(app, follow_redirects=False)


@pytest.fixture
def identity_client(monkeypatch, tmp_path):
    """The same, plus an identity provider — so Save has a boundary to cross and
    the saved-artifact assertions can run through it (Gate 2 reused)."""
    _runtime_env(monkeypatch, tmp_path)

    from src.deploy.context import IdentityTarget

    target = IdentityTarget(issuer="https://auth.example.test",
                            audience="client-1", client_id="client-1")
    monkeypatch.setattr("src.workspace.auth_routes._target", lambda: target)

    from src.deploy.identity import Identity
    import src.workspace.auth_routes as auth_routes

    def signed_in(request):
        subject = request.headers.get(SUBJECT_HEADER)
        return (Identity(subject=subject, email=f"{subject}@x.test")
                if subject else None)

    monkeypatch.setattr(auth_routes, "signed_in", signed_in)

    from src.api import app

    return TestClient(app, follow_redirects=False)


# --- helpers ----------------------------------------------------------------

def _evaluate(client, data=ANSWER):
    """Evaluate a complete strategy and return the rendered result page."""
    posted = client.post("/evaluate", data=data)
    assert posted.status_code == 303, posted.text[:300]
    location = posted.headers["location"]
    assert location.startswith("/pilot/reviews/"), location
    page = client.get(location)
    assert page.status_code == 200
    return page.text


def _section(html, section_id):
    """The markup of one `id="…"` result section, from its opening tag to the
    next section's opening tag — so a claim about 'in section A' is checked
    against A's own bytes, not the whole page."""
    start = html.find(f'id="{section_id}"')
    assert start != -1, f"section {section_id} is not on the page"
    tail = html[start:]
    nxt = re.search(r'class="result-section', tail[10:])
    return tail if nxt is None else tail[: nxt.start() + 10]


# --- §6: the four sections --------------------------------------------------

class TestFourSectionsPresent:
    def test_all_four_decision_sections_render(self, client):
        html = _evaluate(client)
        for section_id in ("interpreted-strategy", "outcome",
                           "evidence", "alternative"):
            assert f'id="{section_id}"' in html, f"missing section {section_id}"
        # And each is a §6 section, marked A/B/C/D.
        for mark in (">A<", ">B<", ">C<", ">D<"):
            assert mark in html

    def test_each_section_pulls_from_the_existing_context(self, client):
        html = _evaluate(client)
        # A — the parameter the reading settled.
        assert "assets" in _section(html, "interpreted-strategy")
        # B — the figure the run produced (the same class the journey pins).
        assert 'class="figure"' in _section(html, "outcome")
        # C — the intent hash / run identity from provenance.
        assert "run id / hash" in _section(html, "evidence")
        # D — a benchmark named as an alternative.
        assert "Hold cash" in _section(html, "alternative")


# --- §6.A: provenance labelling ---------------------------------------------

class TestProvenanceLabelling:
    """Rendered directly with a controlled parameter set, so all four §6 states
    are present at once and each lands under the correct label. Uses a real
    reading (a catalogue pick — no model) for the surrounding context, then
    injects the four rows; nothing here computes a figure."""

    def _render_with(self, parameters, needed, picked=""):
        from src.workspace.catalog_intent import reading_for
        from src.workspace.pilot_routes import page, methodology_concept_for
        from src.workspace.routes import TEMPLATES

        reading = reading_for("cross-sectional-momentum",
                              "Hold the strongest few, rebalanced monthly.")
        ctx = page(reading, text="Hold the strongest few, rebalanced monthly.",
                   run={})
        ctx["parameters"] = parameters
        ctx["needed"] = needed
        ctx["picked"] = picked
        ctx["methodology_concept"] = methodology_concept_for(picked)
        return TEMPLATES.env.get_template("pilot.html").render(**ctx)

    def test_each_state_shows_its_six_a_label(self):
        from src.workspace.parameters import Parameter

        html = self._render_with(
            parameters=[
                Parameter(name="assets", state="SETTLED", value="VTI",
                          author="USER"),
                Parameter(name="amount", state="SETTLED", value="500",
                          author="", provenance="read from your words"),
                Parameter(name="day_rule", state="CHOSEN",
                          value="ModifiedFollowing"),
                Parameter(name="lookback", state="NEEDED", value=""),
            ],
            needed=["lookback"],
            picked="cross-sectional-momentum")

        assert "explicitly supplied" in html   # author == USER
        assert "inferred by witness" in html    # SETTLED, read by the reader
        assert "defaulted by methodology" in html  # CHOSEN engine default
        assert "unresolved" in html             # NEEDED

    def test_a_refused_row_is_not_given_a_six_a_label(self):
        """REFUSED is 'won't be used', not one of the four §6.A categories — so
        it must not be mislabelled 'defaulted by methodology'."""
        from src.workspace.parameters import Parameter

        html = self._render_with(
            parameters=[
                Parameter(name="withdrawal_order", state="REFUSED",
                          value="LIFO", detail="not executed by this build"),
            ],
            needed=[])
        # It appears, under its own 'won't run' treatment…
        assert "withdrawal_order" in html
        assert "won't run" in html
        # …and is not tagged with a §6.A methodology-default label.
        assert "withdrawal_order</strong><span class=\"plabel defaulted\"" \
            not in html.replace("\n", "")


# --- §6.A: unresolved blocks final evaluation -------------------------------

class TestUnresolvedBlocks:
    def test_a_result_changing_unresolved_dimension_blocks_the_figure(self, client):
        # No asset supplied: 'assets' is NEEDED and result-changing.
        page = client.get("/evaluate", params={"describe": SENTENCE})
        assert page.status_code == 200
        html = page.text
        assert "Final evaluation is blocked" in html
        assert "assets" in html
        # And no final figure is presented for the unfinished request.
        assert 'class="figure"' not in html


# --- §6.D: the benchmark is an alternative, never merged ---------------------

class TestBenchmarkIsAnAlternative:
    def test_the_benchmark_renders_in_d_and_not_in_a(self, client):
        html = _evaluate(client)
        alternative = _section(html, "alternative")
        interpreted = _section(html, "interpreted-strategy")
        # The benchmark alternative is in D…
        assert "Hold cash" in alternative
        assert "alternative" in alternative.lower()
        # …and is NOT folded into the user's interpreted strategy in A.
        assert "Hold cash" not in interpreted
        # D states saving keeps the user's strategy, not an alternative.
        assert "Saving keeps your strategy" in alternative

    def test_saving_keeps_the_users_strategy_not_the_benchmark(
            self, identity_client, monkeypatch):
        """Gate 2 reused: the saved plan's content hash is the user's confirmed
        strategy, and no benchmark is folded into the stored artifact."""
        import json

        # Evaluate anonymously, then determine the plan identity the evaluation
        # already fixed (a dict-only reopen — constructs no reader).
        posted = identity_client.post("/evaluate", data=ANSWER)
        review_id = posted.headers["location"].rsplit("/", 1)[-1]

        from src.workspace.owner import SHARED
        from src.workspace.pilot import reopen
        from src.workspace.pilot_store import load_review_under, plan_id_for

        stored_review = load_review_under(SHARED, review_id)
        expected = plan_id_for(reopen(stored_review))

        # Save while signed in.
        saved = identity_client.post(
            "/evaluate/save", data={"review_id": review_id, "picked": ""},
            headers={SUBJECT_HEADER: "alice"})
        assert saved.status_code == 303
        plan_id = saved.headers["location"].rsplit("/", 1)[-1]

        # The saved hash equals the user's confirmed strategy — not a benchmark.
        assert plan_id == expected, (
            "the saved plan id is not the content address of the confirmed "
            "user strategy")

        # And no benchmark alternative is inside the saved artifact.
        from src.workspace.pilot_store import load

        blob = json.dumps(load(plan_id), default=str)
        for benchmark in ("Hold cash", "Contribute to S&P 500",
                          "Your basket, bought and held"):
            assert benchmark not in blob, (
                f"benchmark {benchmark!r} was folded into the saved strategy")


# --- §6.C: the evidence panel -----------------------------------------------

class TestEvidencePanel:
    def test_present_fields_render_and_absent_fields_are_marked(self, client):
        html = _evaluate(client)
        evidence = _section(html, "evidence")

        # Present, from provenance the reading/result already carried.
        assert "run id / hash" in evidence
        assert "market-data snapshot" in evidence
        assert "read by" in evidence

        # Genuinely absent on this surface — marked, never fabricated.
        assert "not recorded on this surface" in evidence      # methodology/protocol
        assert "not classified here" in evidence                # performance class

        # A figure ran, so a real snapshot id is shown (not the absent marker).
        assert "no figure ran, or no snapshot recorded" not in evidence

    def test_the_panel_is_expandable(self, client):
        html = _evaluate(client)
        evidence = _section(html, "evidence")
        assert "<details>" in evidence and "<summary>" in evidence


# --- §7: research ↔ evaluator continuity ------------------------------------

class TestContinuity:
    def test_research_offers_a_public_semantics_move_into_evaluate(self, client):
        """/research links into /evaluate and carries only public strategy
        semantics — never holdings, tax or account state (§1 hard rule)."""
        page = client.get("/research")
        html = page.text
        assert 'href="/evaluate"' in html
        assert "Describe your own strategy" in html
        # The only pre-fill is a public catalogue pick.
        assert "/evaluate?picked=scheduled-funding" in html
        # No account-state parameters ride the affordance.
        for leaked in ("holdings", "tax", "income", "account", "balance"):
            assert f"{leaked}=" not in html

    def test_the_evidence_panel_links_back_to_a_methodology_page(self, client):
        html = _evaluate(client)
        evidence = _section(html, "evidence")
        assert 'href="/methodologies' in evidence

    def test_a_mapped_strategy_links_to_its_concept_page(self):
        """A picked strategy a published methodology covers points back at that
        concept's own /methodologies/{concept} page."""
        from src.workspace.catalog_intent import reading_for
        from src.workspace.pilot_routes import page, methodology_concept_for
        from src.workspace.routes import TEMPLATES

        reading = reading_for("cross-sectional-momentum",
                              "Hold the strongest few, rebalanced monthly.")
        ctx = page(reading, text="…", run={})
        ctx["methodology_concept"] = methodology_concept_for(
            "cross-sectional-momentum")
        html = TEMPLATES.env.get_template("pilot.html").render(**ctx)
        assert 'href="/methodologies/xsmom"' in html


# --- no recompute / no regression -------------------------------------------

class TestNoRecompute:
    def test_the_same_evaluation_yields_the_same_figure_and_hash(self, client):
        """The restructuring is presentation only: two reopens of one evaluation
        show the same figure and the same intent hash (no second parser)."""
        first = _evaluate(client)
        # Re-evaluate the identical submission and reopen again.
        second = _evaluate(client)

        def figure(html):
            m = re.search(r'class="figure">([^<]+)<', html)
            return m and m.group(1).strip()

        def intent(html):
            after = html[html.find("run id / hash"):]
            m = re.search(r"<code>([^<]+)</code>", after)
            return m and m.group(1)

        assert figure(first) and figure(first) == figure(second)
        assert intent(first) and intent(first) == intent(second)

    def test_page_context_helpers_call_no_model(self):
        """The added context is read off the run/result objects — the helpers
        take a plain dict and never construct a reader or call a provider."""
        from src.workspace import pilot_routes as pr

        empty = {}
        assert pr._alternatives(empty) == []
        assert pr._comparison_note(empty) is None
        assert pr._market_snapshot(empty) is None
        assert pr._disclosures(empty) == []


# --- the evaluate action ----------------------------------------------------

class TestEvaluateAction:
    """Running is a deliberate, gated act: one explicit Evaluate button, present
    whether or not anything is unresolved, that a browser is told to keep
    disabled until every required value is filled — clickable exactly when the
    ambiguities the run needs resolved are resolved."""

    def _render(self, parameters, needed):
        from src.workspace.catalog_intent import reading_for
        from src.workspace.pilot_routes import page
        from src.workspace.routes import TEMPLATES

        reading = reading_for("cross-sectional-momentum",
                              "Hold the strongest few, rebalanced monthly.")
        ctx = page(reading, text="Hold the strongest few, rebalanced monthly.",
                   run={})
        ctx["parameters"] = parameters
        ctx["needed"] = needed
        ctx["picked"] = "cross-sectional-momentum"
        return TEMPLATES.env.get_template("pilot.html").render(**ctx)

    def test_the_button_is_present_and_labelled_evaluate_when_resolved(self, client):
        """A fully resolved strategy still shows an explicit Evaluate control —
        the run is no longer only an implicit thing that happened on load."""
        html = _evaluate(client)
        assert 'class="prun"' in html
        assert ">Evaluate</button>" in html
        # Nothing unresolved: rendered ready, and no "fill in N" hint element
        # (the script still references the hint by attribute — assert on the
        # rendered span, not the attribute name).
        assert 'data-needs="0"' in html
        assert '<span class="prun-hint"' not in html

    def test_the_button_renders_enabled_for_a_scriptless_browser(self, client):
        """The disable is a JS enhancement, never the only gate — the server must
        not ship a `disabled` button a scriptless browser could never submit."""
        html = _evaluate(client)
        m = re.search(r'<button[^>]*class="prun"[^>]*>', html)
        assert m and "disabled" not in m.group(0)

    def test_an_unresolved_dimension_keeps_the_button_and_declares_the_gate(self):
        """When a required value is still missing the button is still present
        (always), and the gate is declared in the markup: a required count the
        script reads and a hint it turns into 'fill in the N required values'."""
        from src.workspace.parameters import Parameter

        html = self._render(
            parameters=[
                Parameter(name="assets", state="SETTLED", value="VTI",
                          author="USER"),
                Parameter(name="lookback", state="NEEDED", value=""),
            ],
            needed=["lookback"])
        assert ">Evaluate</button>" in html
        assert 'data-needs="1"' in html
        assert "data-run-hint" in html


# --- §6.B: failure feedback -------------------------------------------------

class TestFailureFeedback:
    """A run that produces no figure sorts its reason into the one dimension the
    reader can act on — a value they can change (`plan`), a data gap they cannot
    (`data_gap`), or neither (`internal`) — instead of one flat 'unavailable'
    line that never says whether there is anything to do."""

    def _refusal(self, *, kind, detail):
        from src.workspace.catalog_intent import reading_for
        from src.workspace.pilot_routes import page
        from src.workspace.routes import TEMPLATES

        reading = reading_for("cross-sectional-momentum",
                              "Hold the strongest few, rebalanced monthly.")
        run = {"result": None, "strategy_not_executed": True,
               "refusal_kind": kind, "unavailable": detail}
        ctx = page(reading, text="Hold the strongest few, rebalanced monthly.",
                   run=run)
        return TEMPLATES.env.get_template("pilot.html").render(**ctx)

    def test_a_data_gap_says_it_is_not_the_users_to_fix_and_shows_the_reason(self):
        html = self._refusal(
            kind="data_gap",
            detail=("No price history for ZZZZ over this period, so the scenario "
                    "cannot be replayed. This is a data gap, not a result."))
        assert "refusal-data_gap" in html
        # Told plainly that editing a value cannot help.
        assert "can't resolve this by changing a value" in html
        # And the engine's own sentence is still shown as the specific detail.
        assert "No price history for ZZZZ" in html

    def test_a_plan_refusal_points_the_user_at_the_values_to_change(self):
        html = self._refusal(
            kind="plan",
            detail=("this plan allocates by a computed strategy, which restores "
                    "its weights on a calendar."))
        assert "refusal-plan" in html
        # Sent back to the parameters, by the anchor that actually reaches them.
        assert 'href="#answers"' in html
        assert "this plan allocates by a computed strategy" in html

    def test_the_two_kinds_give_opposite_guidance(self):
        gap = self._refusal(kind="data_gap", detail="d")
        plan = self._refusal(kind="plan", detail="p")
        assert "can't resolve this by changing a value" in gap
        assert "can't resolve this by changing a value" not in plan
        assert "Adjust it in" in plan
        assert "Adjust it in" not in gap

    def test_an_internal_refusal_blames_neither_a_value_nor_the_data(self):
        html = self._refusal(
            kind="internal",
            detail=("This result is unavailable. The executed purchases and the "
                    "reported totals do not agree, so no figure is shown."))
        assert "refusal-internal" in html
        assert "can't resolve this by changing a value" not in html
        # Not sent to the parameters — nothing the person types changes it.
        assert 'href="#answers"' not in html
        assert "The executed purchases" in html


class TestOutcomeSurfacedAtTop:
    """A result page whose only result sits in §6.B — below a tall parameter
    table — reads at a glance as the page it was submitted from. The outcome is
    surfaced at the top so a figure or a refusal is the first thing seen, and the
    review page is visibly distinct from the /workspace/new draft (which has no
    run and must NOT grow the banner)."""

    def _render(self, run):
        from src.workspace.catalog_intent import reading_for
        from src.workspace.pilot_routes import page
        from src.workspace.routes import TEMPLATES

        reading = reading_for("cross-sectional-momentum",
                              "Hold the strongest few, rebalanced monthly.")
        ctx = page(reading, text="invest $50 into MSFT weekly for 5 years", run=run)
        return TEMPLATES.env.get_template("pilot.html").render(**ctx)

    def test_a_refusal_is_announced_at_the_top_above_the_parameters(self):
        html = self._render({
            "result": None, "strategy_not_executed": True, "refusal_kind": "data_gap",
            "unavailable": "There is no pricing data for MSFT in this deployment.",
        })
        assert 'class="outcome-top' in html
        assert "There is no pricing data for MSFT" in html
        # The banner comes before the parameter table — the reason it exists.
        assert html.index("outcome-top") < html.index('class="params"')

    def test_the_draft_page_has_no_outcome_banner(self, client):
        # /workspace/new renders with run={} (no evaluation yet), so nothing is
        # announced — the banner is what tells a review page apart from the draft.
        # (The class appears in the stylesheet; assert the rendered div is absent.)
        html = self._render({})
        assert '<div class="outcome-top' not in html

    def test_a_figure_is_surfaced_at_the_top_with_a_link_to_the_chart(self, client):
        html = _evaluate(client)
        assert 'class="outcome-top ok"' in html
        assert 'href="#outcome"' in html
        # The headline figure at the top is the same one §6.B renders below.
        m = re.search(r'class="outcome-top__figure">([^<]+)<', html)
        assert m and m.group(1).strip()
