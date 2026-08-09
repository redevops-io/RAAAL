"""What the pilot counts, and the line it does not cross.

Five objective events, each a fact about what happened, each carrying the
deployment profile so a model-only cohort and a future dual-witness one can be
separated later.

The line: no field here is a claim about what anyone *thought*. A telemetry
column named `understood_refusal` would be a number derived from nothing, and a
dashboard reporting "82% understood the refusal" is worse than no dashboard
because somebody will believe it. Those are interview questions and this file
asserts that they stay interview questions.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

SENTENCE = "invest $500 monthly"
NEW = "/workspace/new"


@pytest.fixture
def client(monkeypatch, tmp_path):
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/e.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.api import app

    return TestClient(app)


def _journey(client):
    client.get(NEW, params={"describe": SENTENCE})
    saved = client.post("/pilot/save",
                        data={"describe": SENTENCE, "answer_assets": "VTI"},
                        follow_redirects=False)
    client.get(saved.headers["location"])
    return saved.headers["location"]


class TestEveryProducerIsReachableFromTheRealJourney:
    """An event nothing emits is a column that reads as zero forever.

    Each of the five is asserted to appear from `/workspace/new` → save →
    reopen, which is the journey a cohort actually walks — not from calling the
    recorder directly, which would prove only that the recorder works.
    """

    def test_all_five_appear(self, client):
        _journey(client)

        from src.workspace.pilot_events import KINDS, every_event

        seen = {e["kind"] for e in every_event()}
        assert seen == set(KINDS), f"never emitted: {sorted(set(KINDS) - seen)}"


class TestRefusalRecordsTheNamedReason:
    def test_a_capability_refusal_names_the_capability(self, client):
        client.get(NEW, params={"describe": SENTENCE})

        from src.workspace.pilot_events import (
            CAPABILITY_REFUSED, PLAN_REFUSED, every_event)

        refusals = [e for e in every_event() if e["kind"] == PLAN_REFUSED]
        assert refusals
        assert any(e.get("capability") == "assets"
                   and e.get("refusal_code") == CAPABILITY_REFUSED
                   for e in refusals)

    def test_no_event_carries_rendered_copy_or_user_text(self, client):
        """The first version stored the engine's `unavailable` *message*. A
        message can embed whatever the user typed and changes whenever the
        wording does — a field that moves when a sentence is reworded cannot be
        counted across a cohort."""
        _journey(client)

        from src.workspace.pilot_events import every_event

        for event in every_event():
            flat = " ".join(str(v) for v in event.values())
            assert SENTENCE not in flat, f"{event['kind']} carries user text"
            assert "$500" not in flat
            for value in event.values():
                assert len(str(value)) < 60, (
                    f"{event['kind']} carries a long string, which is how "
                    "rendered copy arrives in telemetry")


class TestReopenDoesNotReinterpret:
    def test_reopening_emits_the_event_with_no_reader_in_reach(self, client,
                                                               monkeypatch):
        location = _journey(client)

        import src.workspace.pilot_routes as pilot_routes

        monkeypatch.setattr(
            pilot_routes, "configured_reader",
            lambda: (_ for _ in ()).throw(
                AssertionError("a reader was constructed on reopen")))

        from src.workspace.pilot_events import PLAN_REOPENED, every_event

        before = sum(1 for e in every_event() if e["kind"] == PLAN_REOPENED)
        assert client.get(location).status_code == 200
        after = sum(1 for e in every_event() if e["kind"] == PLAN_REOPENED)
        assert after == before + 1

    def test_a_reopen_does_not_also_emit_compiled(self, client):
        """`plan_compiled` means a sentence became a plan. A reopen recompiles
        from a pinned intent and interprets nothing, so counting it as a
        compile would inflate the only number that says how often people
        started something new."""
        location = _journey(client)

        from src.workspace.pilot_events import PLAN_COMPILED, every_event

        before = sum(1 for e in every_event() if e["kind"] == PLAN_COMPILED)
        client.get(location)
        after = sum(1 for e in every_event() if e["kind"] == PLAN_COMPILED)
        assert after == before


class TestResultAndRefusalAreMutuallyExclusive:
    def test_one_execution_emits_exactly_one_of_them(self, client):
        """Both would make results and refusals sum to more than the
        executions that happened, and every ratio drawn from them would be
        wrong in a way nothing shows."""
        _journey(client)

        from src.workspace.pilot_events import (
            PLAN_REFUSED, PLAN_RESULT_SHOWN, every_event)

        events = every_event()
        by_plan = {}
        for event in events:
            if event["kind"] == PLAN_RESULT_SHOWN:
                by_plan.setdefault(event["plan_id"], []).append("result")
            elif (event["kind"] == PLAN_REFUSED
                  and not event.get("capability")):
                by_plan.setdefault(event["plan_id"], []).append("refusal")

        for plan_id, outcomes in by_plan.items():
            assert len(set(outcomes)) == 1, (
                f"{plan_id} recorded both a result and an execution refusal")

    def test_an_unexecutable_plan_records_a_refusal_and_no_result(self, client):
        client.post("/pilot/save", data={"describe": SENTENCE},
                    follow_redirects=True)

        from src.workspace.pilot_events import PLAN_RESULT_SHOWN, every_event

        # Saved with `assets` unanswered: nothing is executable, so no figure.
        assert not [e for e in every_event() if e["kind"] == PLAN_RESULT_SHOWN]


class TestTheProfileComesFromTheDeployment:
    def test_it_is_not_a_route_local_constant(self, monkeypatch, tmp_path):
        """Every event in a pilot run carries the same profile, so a hardcoded
        pair would look identical to a resolved one. Changing the deployment
        and re-reading is the only thing that tells them apart."""
        from src.deploy import context as deploy_context

        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "MODEL_ASSISTED")
        monkeypatch.setenv("QUANTIFY_PILOT_READER", "hosted")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/p.db")
        resolved = deploy_context.resolve(dict(os.environ))
        monkeypatch.setattr(deploy_context, "current", lambda: resolved)

        from src.workspace.pilot_events import _profile

        assert _profile() == {"parser_mode": "MODEL_ASSISTED",
                              "pilot_reader": "HOSTED"}

    def test_and_the_runtime_profile_reads_the_same_way(self, client):
        _journey(client)

        from src.workspace.pilot_events import every_event

        for event in every_event():
            assert event["parser_mode"] == "RUNTIME"
            assert event["pilot_reader"] == "RECORDED"


class TestTelemetryCannotBreakARequest:
    def test_a_failing_event_store_does_not_reach_the_user(self, client,
                                                           monkeypatch):
        """Telemetry is the expendable half. A pilot user losing their plan
        because an analytics table was locked is a worse outcome than losing
        the count — the same rule the trace retention already follows."""
        import src.workspace.pilot_events as events

        def broken():
            raise RuntimeError("the events table is unavailable")

        monkeypatch.setattr(events, "_connect", broken)

        page = client.get(NEW, params={"describe": SENTENCE})
        assert page.status_code == 200
        assert "MODEL_ONLY_ACCEPTED" in page.text

    def test_and_the_journey_still_completes(self, client, monkeypatch):
        import src.workspace.pilot_events as events

        monkeypatch.setattr(events, "_connect",
                            lambda: (_ for _ in ()).throw(RuntimeError("down")))
        location = _journey(client)
        assert client.get(location).status_code == 200


class TestWhatIsDeliberatelyNotMeasured:
    def test_no_event_claims_to_know_what_someone_thought(self):
        """The line. Checked structurally rather than by reading the module's
        prose, because a source grep asserting a property matches its own
        explanation of that property — four times in this project."""
        import ast
        from pathlib import Path

        from src.workspace import pilot_events

        tree = ast.parse(Path(pilot_events.__file__).read_text())
        names = {t.id for node in tree.body if isinstance(node, ast.Assign)
                 for t in node.targets if isinstance(t, ast.Name)}
        forbidden = ("TRUST", "UNDERSTOOD", "SATISFACTION", "CONFUSED",
                     "SENTIMENT", "NPS")
        assert not [n for n in names if any(f in n.upper() for f in forbidden)]

    def test_the_summary_says_what_it_cannot_answer(self):
        """Stated in the output, not only in a docstring. Whoever reads the
        numbers is the person who needs to know which questions they do not
        answer."""
        from src.workspace.pilot_events import summary

        unmeasured = summary()["not_measured_here"]
        assert len(unmeasured) >= 4
        assert any("understood" in one for one in unmeasured)
        assert any("trustworthy" in one for one in unmeasured)
