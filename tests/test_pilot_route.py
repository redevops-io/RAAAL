"""The acceptance invariant, proved through HTTP rather than by reading source.

> A plan created through the pilot workspace must prove that `compile_intent`
> was reached and `compile_scenario` was not.

Source inspection would not prove it. An import can exist and never be called;
a call can be added on a branch the test never takes. So this drives real
requests and establishes both halves by evidence:

    reached      the stored artifact carries `compiled_by=quantify-mission@1`
                 and `compiled_from=<the intent's own hash>`, which nothing but
                 `compile_intent` produces

    not reached  `compile_scenario` is replaced with a function that raises.
                 If the journey completes, the legacy compiler was not called —
                 and the substitution is verified to be effective by a control
                 that calls it and fails.

That control matters more than it looks. Without it, a typo in the patch target
would make every "the legacy path was not reached" assertion pass by patching
nothing.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

SENTENCE = "invest $500 monthly"


@pytest.fixture
def pilot_client(monkeypatch, tmp_path):
    """An app whose deployment declares the runtime and reads from recordings."""
    from src.deploy import context as deploy_context

    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used-by-recordings")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/pilot.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.api import app

    return TestClient(app)


def _legacy_must_not_run(monkeypatch):
    """Replace the legacy compiler with something that cannot be mistaken for
    success, and return a probe that proves the replacement took effect."""
    import src.mission.compiler as compiler
    import src.workspace.routes as routes

    def refuse(*args, **kwargs):
        raise AssertionError(
            "compile_scenario was reached on a pilot journey; the deployment "
            "declares RUNTIME and a plan compiled from prose is a different "
            "artifact from one compiled from a pinned intent")

    monkeypatch.setattr(compiler, "compile_scenario", refuse)
    monkeypatch.setattr(routes, "compile_scenario", refuse, raising=False)
    return refuse


class TestTheRuntimeIsReached:
    def test_a_submission_produces_a_plan_compiled_from_an_intent(
            self, pilot_client, monkeypatch):
        _legacy_must_not_run(monkeypatch)

        page = pilot_client.get("/pilot", params={"describe": SENTENCE})
        assert page.status_code == 200
        body = page.text

        # Evidence in the response, not in the import graph.
        assert "quantify-mission@1" in body or "compiled by" in body
        assert "claude-sonnet-5@1" in body

    def test_the_saved_artifact_names_the_intent_it_was_compiled_from(
            self, pilot_client, monkeypatch, tmp_path):
        _legacy_must_not_run(monkeypatch)

        saved = pilot_client.post(
            "/pilot/save",
            data={"describe": SENTENCE, "answer_assets": "VTI"},
            follow_redirects=True)
        assert saved.status_code == 200

        from src.workspace.pilot_store import every_plan

        (plan,) = every_plan()
        derivation = plan["derivation"]
        assert derivation["compiled_by"] == "quantify-mission@1"
        assert derivation["compiled_from"] == plan["intent"]["intent_hash"]

    def test_the_control_proves_the_substitution_works(self, monkeypatch):
        """Without this, a typo in the patch target would make every
        "the legacy compiler was not reached" assertion pass by patching
        nothing at all."""
        _legacy_must_not_run(monkeypatch)

        import src.mission.compiler as compiler

        with pytest.raises(AssertionError, match="compile_scenario was reached"):
            compiler.compile_scenario("anything")


class TestTheLoopThroughHttp:
    def test_a_question_is_asked_before_the_plan_can_run(self, pilot_client):
        page = pilot_client.get("/pilot", params={"describe": SENTENCE})
        assert "assets" in page.text
        assert "needs an answer" in page.text

    def test_answering_it_and_saving_reopens_to_the_same_plan(
            self, pilot_client, monkeypatch):
        _legacy_must_not_run(monkeypatch)

        saved = pilot_client.post(
            "/pilot/save",
            data={"describe": SENTENCE, "answer_assets": "VTI"},
            follow_redirects=False)
        assert saved.status_code == 303
        location = saved.headers["location"]

        first = pilot_client.get(location)
        second = pilot_client.get(location)
        assert first.status_code == 200
        assert "Reopened from the saved plan" in first.text
        assert first.text == second.text, (
            "two reopens of one plan rendered differently; replay from a "
            "pinned intent is the property this page exists to demonstrate")

    def test_reopening_does_not_read_the_sentence_again(
            self, pilot_client, monkeypatch):
        """The reader is replaced with one that raises. If the page still
        renders, nothing on the reopen path consulted it."""
        _legacy_must_not_run(monkeypatch)

        saved = pilot_client.post(
            "/pilot/save",
            data={"describe": SENTENCE, "answer_assets": "VTI"},
            follow_redirects=False)
        location = saved.headers["location"]

        import src.workspace.pilot_routes as pilot_routes

        def no_reader():
            raise AssertionError(
                "a reader was constructed on the reopen path; a plan reopened "
                "by re-reading its sentence is a fresh request wearing an old "
                "name")

        monkeypatch.setattr(pilot_routes, "configured_reader", no_reader)
        page = pilot_client.get(location)
        assert page.status_code == 200
        assert "Reopened from the saved plan" in page.text


class TestWhatTheUserIsShown:
    def test_one_witness_is_reported_as_one_witness(self, pilot_client):
        """`MODEL_ONLY_ACCEPTED`, never `AGREE`. A pilot page displaying
        agreement while running a single reader would be showing the user
        corroboration that does not exist."""
        page = pilot_client.get("/pilot", params={"describe": SENTENCE}).text
        assert "MODEL_ONLY_ACCEPTED" in page
        assert "one reader only" in page

    def test_a_refusal_is_named(self, pilot_client):
        page = pilot_client.get("/pilot", params={"describe": SENTENCE}).text
        assert "will not do" in page or "needs an answer" in page
        assert "assets" in page


class TestTheGate:
    def test_a_deployment_that_has_not_declared_the_runtime_cannot_reach_it(
            self, monkeypatch, tmp_path):
        """Knowing the URL is not a declaration. The route checks rather than
        the mount, so the endpoint stays in the table the boundary sweep
        audits."""
        from src.deploy import context as deploy_context

        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "DETERMINISTIC")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/x.db")
        resolved = deploy_context.resolve(dict(os.environ))
        monkeypatch.setattr(deploy_context, "current", lambda: resolved)

        from src.api import app

        page = TestClient(app).get("/pilot", params={"describe": SENTENCE})
        assert page.status_code == 404
        assert "does not declare" in page.text

    def test_the_runtime_mode_requires_a_model_like_model_assisted_does(self):
        """Adding `RUNTIME` to the enum without adding it to the coherence
        check would have let a deployment declare the pilot interpreter with no
        key, pass the preflight, and refuse every description at request time
        with the startup proof reporting a valid configuration."""
        from src.deploy.context import ModelTarget, ParserMode

        target = ModelTarget(_api_key=None, model=None,
                             mode=ParserMode.RUNTIME, declared=True)
        problems = target.problems()
        assert len(problems) == 2
        assert all("RUNTIME" in p for p in problems)


class TestTheDeploymentControlsTheReader:
    """The selection must be *controlled by the context*, not hardwired.

    Every other test in this file runs with `PilotReader.RECORDED`, so all of
    them would pass equally well if `configured_reader` ignored the deployment
    and returned a recording unconditionally. That is the shape of a test
    suite proving its own fixture. The reciprocal closes it.
    """

    def _reader_under(self, monkeypatch, tmp_path, declared: str):
        from src.deploy import context as deploy_context

        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
        monkeypatch.setenv("QUANTIFY_PILOT_READER", declared)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "unused-here")
        monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/r.db")

        resolved = deploy_context.resolve(dict(os.environ))
        monkeypatch.setattr(deploy_context, "current", lambda: resolved)

        from src.workspace.pilot_routes import configured_reader

        return configured_reader()

    def test_recorded_gives_the_replaying_reader(self, monkeypatch, tmp_path):
        from src.discovery.hosted_recording import RecordedHostedReader

        reader = self._reader_under(monkeypatch, tmp_path, "recorded")
        assert isinstance(reader, RecordedHostedReader)

    def test_hosted_gives_the_provider_reader(self, monkeypatch, tmp_path):
        """Constructed, not called — this asserts the *selection*, and calling
        it would put a provider request in the suite."""
        from src.discovery.readers_quantify import HostedReader

        reader = self._reader_under(monkeypatch, tmp_path, "hosted")
        assert isinstance(reader, HostedReader)
        assert reader.model == "claude-sonnet-5"

    def test_an_absent_declaration_defaults_to_the_provider(self, monkeypatch,
                                                            tmp_path):
        """Not to recordings. A deployment that silently replayed fixtures
        because nobody declared a reader would serve answers from a file and
        report them as the model's — the failure the enum's docstring names."""
        from src.discovery.readers_quantify import HostedReader

        monkeypatch.delenv("QUANTIFY_PILOT_READER", raising=False)
        reader = self._reader_under(monkeypatch, tmp_path, "")
        assert isinstance(reader, HostedReader)


class TestThePersistedArtifactRecordsTheProfile:
    """Not only the page. A pilot's conclusions are drawn from what was stored,
    and a page that displayed the profile while the artifact omitted it would
    leave the analysis unable to separate model-only plans from dual-witness
    ones."""

    def test_the_stored_plan_says_one_witness_and_says_why(
            self, pilot_client, monkeypatch):
        _legacy_must_not_run(monkeypatch)

        pilot_client.post("/pilot/save",
                          data={"describe": SENTENCE, "answer_assets": "VTI"},
                          follow_redirects=True)

        from src.workspace.pilot_store import every_plan

        (plan,) = every_plan()
        assert plan["profile"]["single_witness"] is True
        assert plan["profile"]["available"] == ["model"]
        assert "not installed" in plan["profile"]["reason"]

    def test_every_settled_field_is_model_only_and_none_claim_agreement(
            self, pilot_client, monkeypatch):
        _legacy_must_not_run(monkeypatch)

        pilot_client.post("/pilot/save",
                          data={"describe": SENTENCE, "answer_assets": "VTI"},
                          follow_redirects=True)

        from src.workspace.pilot_store import every_plan

        (plan,) = every_plan()
        provenances = {f["provenance"] for f in plan["settled"]}
        assert "AGREE" not in provenances
        assert provenances <= {"MODEL_ONLY_ACCEPTED", "USER_ANSWERED"}

    def test_the_human_answer_is_recorded_as_the_human_s(
            self, pilot_client, monkeypatch):
        """"The user agreed" and "the model proposed" must never be the same
        record — the distinction Mission's declared defaults depend on."""
        _legacy_must_not_run(monkeypatch)

        pilot_client.post("/pilot/save",
                          data={"describe": SENTENCE, "answer_assets": "VTI"},
                          follow_redirects=True)

        from src.workspace.pilot_store import every_plan

        (plan,) = every_plan()
        authors = {name: field["author"]
                   for name, field in plan["intent"]["fields"].items()}
        assert authors["assets"] == "USER"
        assert authors["cadence"] == "MODEL"


#: The workspace router carries a prefix, so the entry point a cohort actually
#: types is `/workspace/new`. Named once here rather than spelled out at each
#: call site — the first version of these tests used `/new`, got 404 from every
#: one of them, and the failure looked like the branch not working.
NEW = "/workspace/new"


class TestNewIsTheEntryPoint:
    """`/workspace/new` is where a cohort arrives, so that is what must be
    tested.

    A separate `/pilot` URL would have left the legacy path as the default
    experience, and an experiment about whether the runtime improves *this*
    workspace cannot be run on a surface people do not arrive at. So the branch
    is on the resolved deployment mode and the route chooses nothing itself.
    """

    def test_in_pilot_mode_new_reaches_the_runtime(self, pilot_client,
                                                   monkeypatch):
        _legacy_must_not_run(monkeypatch)

        page = pilot_client.get(NEW, params={"describe": SENTENCE})
        assert page.status_code == 200
        assert "MODEL_ONLY_ACCEPTED" in page.text
        assert "claude-sonnet-5@1" in page.text

    def test_and_the_artifact_it_saves_records_the_witness_profile(
            self, pilot_client, monkeypatch):
        _legacy_must_not_run(monkeypatch)

        pilot_client.get(NEW, params={"describe": SENTENCE})
        pilot_client.post("/pilot/save",
                          data={"describe": SENTENCE, "answer_assets": "VTI"},
                          follow_redirects=True)

        from src.workspace.pilot_store import every_plan

        (plan,) = every_plan()
        assert plan["derivation"]["compiled_by"] == "quantify-mission@1"
        assert plan["profile"]["available"] == ["model"]
        assert plan["profile"]["single_witness"] is True

    def test_the_alias_and_the_entry_point_render_the_same_page(
            self, pilot_client, monkeypatch):
        """Delegation, not duplication. Two implementations of one journey
        drift, and the drift is invisible until a user reports a difference
        nobody can reproduce."""
        _legacy_must_not_run(monkeypatch)

        through_new = pilot_client.get(NEW, params={"describe": SENTENCE})
        through_alias = pilot_client.get("/pilot", params={"describe": SENTENCE})
        assert through_new.text == through_alias.text


class TestTheBranchSelectionIsRecorded:
    """Which execution path produced a plan, stated rather than inferred.

    `compiled_by=quantify-mission@1` and `single_witness: true` both *imply*
    the runtime path, and a later analyst can reconstruct it from them. An
    inference is not a statement: a pilot conclusion about model-only versus
    dual-witness plans should rest on what the deployment declared, recorded at
    the moment the plan was saved, not on what the artifact happens to contain.
    """

    def test_the_artifact_names_the_declared_mode_and_reader(
            self, pilot_client, monkeypatch):
        _legacy_must_not_run(monkeypatch)

        pilot_client.post("/pilot/save",
                          data={"describe": SENTENCE, "answer_assets": "VTI"},
                          follow_redirects=True)

        from src.workspace.pilot_store import every_plan

        (plan,) = every_plan()
        assert plan["deployment"]["parser_mode"] == "RUNTIME"
        assert plan["deployment"]["pilot_reader"] == "RECORDED"

    def test_the_recorded_mode_is_read_from_the_deployment(self, monkeypatch,
                                                            tmp_path):
        """The discriminating opposite. Without it, `parser_mode: RUNTIME`
        could be a constant — every pilot plan is saved under that mode, so
        nothing in the pilot journey alone can tell a recorded fact from a
        hardcoded string."""
        from src.deploy import context as deploy_context

        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "MODEL_ASSISTED")
        monkeypatch.setenv("QUANTIFY_PILOT_READER", "hosted")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/d.db")
        resolved = deploy_context.resolve(dict(os.environ))
        monkeypatch.setattr(deploy_context, "current", lambda: resolved)

        from src.workspace.pilot_store import _deployment_record

        record = _deployment_record()
        assert record["parser_mode"] == "MODEL_ASSISTED"
        assert record["pilot_reader"] == "HOSTED"


class TestLegacyModeIsUntouched:
    """The reciprocal. Without it, "pilot mode reaches the runtime" would be
    satisfied by a build that reached the runtime always — and the legacy path,
    which is what every existing user is on, would have been replaced silently.
    """

    @pytest.fixture
    def legacy_client(self, monkeypatch, tmp_path):
        from src.deploy import context as deploy_context

        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "DETERMINISTIC")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL",
                           f"sqlite:///{tmp_path}/legacy.db")
        monkeypatch.delenv("QUANTIFY_PILOT_READER", raising=False)

        resolved = deploy_context.resolve(dict(os.environ))
        monkeypatch.setattr(deploy_context, "current", lambda: resolved)

        from src.api import app

        return TestClient(app)

    def test_new_still_goes_through_the_legacy_compiler(self, legacy_client,
                                                        monkeypatch):
        """Proved by making it fail: the legacy compiler is replaced with a
        probe, and a legacy request must reach it.

        The probe targets `draft.compile_draft`, not `compile_scenario`. The
        first version patched the latter and the test failed — correctly, and
        for a reason that was about my assumption rather than the code: the
        *draft* path has compiled through `compile_draft` since a defect where
        the preview and the save path used different functions. Patching the
        one the route does not call would have asserted nothing while looking
        like a reciprocal.
        """
        import src.workspace.draft as draft

        reached = {"yes": False}

        def probe(*args, **kwargs):
            reached["yes"] = True
            raise RuntimeError("reached, as expected")

        monkeypatch.setattr(draft, "compile_draft", probe)
        try:
            legacy_client.get(NEW, params={"describe": SENTENCE})
        except Exception:
            pass
        assert reached["yes"], (
            "a deterministic deployment did not reach the legacy compiler; "
            "the pilot branch is serving users who never opted into it")

    def test_and_never_touches_the_pilot_pipeline(self, legacy_client,
                                                  monkeypatch):
        """`compile_intent` is replaced with a function that raises. A legacy
        journey completing is evidence it was not called."""
        import src.workspace.pilot as pilot

        def refuse(*args, **kwargs):
            raise AssertionError(
                "compile_intent was reached on a deterministic deployment; "
                "the runtime is serving a deployment that did not declare it")

        monkeypatch.setattr(pilot, "compile_intent", refuse)
        page = legacy_client.get(NEW, params={"describe": SENTENCE})
        assert page.status_code in (200, 500)

    def test_the_pilot_alias_is_refused_in_legacy_mode(self, legacy_client):
        page = legacy_client.get("/pilot", params={"describe": SENTENCE})
        assert page.status_code == 404
