"""A reopened plan runs the period its owner stated.

`plan_detail` does not use the scenario it just compiled. It rebuilds the one
that was saved — `scenario_from_stored` → `rebuild_scenario` — because what a
user confirmed is what must be replayed, not today's reading of their words.

That rebuild restored one provenance field of eight. So the stored body held
`"over the past five years"`, the rebuilt scenario held nothing, and
`_resolve_window` had no window to slice by: every reopened plan reported a
figure over the entire snapshot. F1 was fixed on the compile path and undone
on the reopen path, and the two are different code.

The assertion that matters is the session count, not the presence of a field.
A round-trip test would have passed the moment `time_window` was carried
across, without establishing that anything downstream consults it — and the
original defect was precisely a value that existed and reached nothing.
"""
from __future__ import annotations

import pytest

CONTROL = ("I buy $1,000 of SPY whenever it crosses below its 200-day moving "
           "average, over the past five years.")

#: Same strategy, no stated period. The control for the session count: without
#: it, a windowed figure and an unwindowed one are indistinguishable.
NO_PERIOD = ("I buy $1,000 of SPY whenever it crosses below its 200-day "
             "moving average.")


@pytest.fixture
def deployment(monkeypatch):
    from src.deploy.context import bind, resolve, unbind

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
    try:
        yield
    finally:
        unbind()


def compiled(text):
    from src.workspace.draft import compile_draft

    return compile_draft(text, name="p", context="window test").scenario


class TestThePremise:
    def test_the_snapshot_is_longer_than_the_stated_window(self, deployment):
        """Five years of a five-year snapshot would slice to itself, and the
        test below could not fail."""
        import src.workspace.routes as routes

        frame = routes._market_data("window premise").frame
        span = frame.index[-1] - frame.index[0]
        assert span.days > 5 * 365 + 200, (
            f"the snapshot covers {span.days} days; a five-year window "
            f"cannot be shown to narrow it")

    def test_the_control_states_a_period_and_the_other_does_not(self,
                                                               deployment):
        assert compiled(CONTROL).provenance.time_window is not None
        assert compiled(NO_PERIOD).provenance.time_window is None


class TestTheRebuiltScenarioStillHasIt:
    def test_the_window_survives_the_round_trip(self, deployment):
        from src.mission.evolution import rebuild_scenario

        scenario = compiled(CONTROL)
        rebuilt = rebuild_scenario(scenario.to_json())
        assert rebuilt is not None
        assert rebuilt.provenance.time_window == scenario.provenance.time_window

    def test_an_absent_window_stays_absent(self, deployment):
        """`provenance@1` bodies have no such key, and a default would be an
        assumption about a plan that recorded nothing."""
        from src.mission.evolution import rebuild_scenario

        body = compiled(CONTROL).to_json()
        body["provenance"].pop("time_window", None)
        assert rebuild_scenario(body).provenance.time_window is None

    def test_a_malformed_window_does_not_become_a_default_one(self, deployment):
        from src.mission.evolution import rebuild_scenario

        body = compiled(CONTROL).to_json()
        body["provenance"]["time_window"] = {"kind": "not-a-kind", "years": 5}
        assert rebuild_scenario(body).provenance.time_window is None

    def test_supportability_is_recomputed_not_restored(self, deployment):
        """A stored `supported: true` beside a kind this build cannot handle
        would let a plan assert its own supportability."""
        from src.mission.evolution import rebuild_scenario

        body = compiled(CONTROL).to_json()
        body["provenance"]["time_window"]["supported"] = False
        rebuilt = rebuild_scenario(body)
        assert rebuilt.provenance.time_window.supported is True

    def test_exclusions_survive_too(self, deployment):
        """The coverage gate consults them. Without them a replayed plan looks
        like one that declared nothing it could not model."""
        from src.mission.evolution import rebuild_scenario

        body = compiled(CONTROL).to_json()
        body["provenance"]["excluded"] = [
            {"item": "employer matching", "reason": "no representation",
             "decision": "PROCEED_WITHOUT_MODELLING", "acknowledged_at": "t"}]
        rebuilt = rebuild_scenario(body)
        assert [one.item for one in rebuilt.provenance.excluded] == \
            ["employer matching"]


class TestTheFigureIsActuallyNarrowed:
    """The claim the field-level tests cannot make: the restored window
    reaches the engine and changes the answer."""

    def sessions(self, scenario):
        import src.workspace.routes as routes

        access = routes._market_data("window run")
        run = routes._run(scenario, access, stated_text=CONTROL)
        assert run is not None, "the plan did not run at all"
        return run

    def test_a_rebuilt_plan_runs_the_stated_period(self, deployment):
        from src.mission.evolution import rebuild_scenario

        rebuilt = rebuild_scenario(compiled(CONTROL).to_json())
        windowed = self.sessions(rebuilt)
        whole = self.sessions(rebuild_scenario(compiled(NO_PERIOD).to_json()))

        assert windowed["result"] is not None, windowed.get("unavailable")
        assert whole["result"] is not None, whole.get("unavailable")
        assert len(windowed['result'].time_weighted) < len(whole['result'].time_weighted), (
            f"the reopened plan ran {len(windowed['result'].time_weighted)} sessions "
            f"against {len(whole['result'].time_weighted)} for the same strategy with "
            f"no stated period; the window reached nothing")

    def test_it_matches_what_the_freshly_compiled_plan_runs(self, deployment):
        """The rebuilt plan and the compiled one are the same plan, so they
        must produce the same figure. This is what a user comparing the draft
        page with the saved page sees."""
        from src.mission.evolution import rebuild_scenario

        scenario = compiled(CONTROL)
        fresh = self.sessions(scenario)
        rebuilt = self.sessions(rebuild_scenario(scenario.to_json()))
        assert len(fresh['result'].time_weighted) == len(rebuilt['result'].time_weighted)
