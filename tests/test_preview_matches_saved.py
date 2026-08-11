"""The scenario shown for confirmation is the scenario persisted.

F11: the preview compiled an executable event-funded plan and the save route
recompiled the same description without `priceable`, so no funding policy
could be built. The page showed a plan that ran and the stored artifact could
never execute. Both bodies were internally consistent, so `content_hash` did
not notice — it covers the compiled artifact, not the inputs that decided it.

    The system showed one compiled scenario to the user and persisted another.

That is one invariant, not three missing arguments. Two mechanisms answer it
and this file tests both:

* `draft.compile_draft` is the only compile. Structural — there is no second
  argument list to keep in step.
* the draft token is a tripwire under it, because `compile_draft` is a
  convention a future call site can still bypass.

**On where the comparison is made.** The classes below that compare a preview
scenario with a saved one do it through the HTTP routes, not by calling
`compile_scenario` twice from the test. A test that builds both sides itself
is testing its own copy of the two paths; F11 was precisely a disagreement
between two call sites, so a test that constructs one call and reuses it for
both cannot see it.

**On the premise.** The control is event-funded over a stated window. A
scheduled plan is included deliberately as a negative control: it compiles
identically with and without `priceable`, because only an event trigger needs
a priceable subject. That is a fact about the product, and it is why a suite
of buy-and-hold or monthly-contribution prompts could not have caught F11.
"""
from __future__ import annotations

import dataclasses
import html as html_module
import re

import pytest

from src.mission.compiler import compile_scenario
from src.mission.spec import ScenarioAmendment
from src.workspace import draft

#: Event-funded, over a stated window, watching and holding the same
#: instrument. Nontrivial in every way the digest is supposed to cover.
CONTROL = ("I buy $1,000 of SPY whenever it crosses below its 200-day moving "
           "average, over the past five years.")

#: Two instruments in different roles.
SPLIT_ROLES = ("Buy VOO with $1,000 whenever SPY crosses below its 200-day "
               "moving average, over the past five years.")

#: The negative control. Scheduled funding needs no priceable subject.
SCHEDULED = "I put $500 into VTI every month for the past 5 years."

ANSWERS = (
    ScenarioAmendment(question_id="account_type", answer="TAXABLE",
                      recorded_at="t"),
    ScenarioAmendment(question_id="funding_source", answer="contribution",
                      recorded_at="t"),
)


@pytest.fixture(autouse=True)
def synthetic(monkeypatch):
    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")


@pytest.fixture
def deployment():
    from src.deploy.context import bind, resolve, unbind

    bind(resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}))
    try:
        yield
    finally:
        unbind()


@pytest.fixture
def priceable(deployment):
    return draft.priceable_for("equivalence premise")


def compiled(prompt, priceable):
    import src.workspace.routes as routes

    return compile_scenario(
        prompt, name="plan", version=1, amendments=ANSWERS,
        benchmark_rule=routes.BENCHMARK_RULE,
        priceable=priceable).scenario


# --------------------------------------------------------------------------
# 1. The premise: the fixtures can produce the defect.
# --------------------------------------------------------------------------

class TestThePremise:
    def test_the_deployment_can_price_something(self, priceable):
        """Without this every case below compares two empty compiles."""
        assert "SPY" in priceable and "VOO" in priceable

    def test_the_control_is_event_funded_over_a_window(self, priceable):
        scenario = compiled(CONTROL, priceable)
        assert scenario.is_event_funded, "not the shape F11 destroyed"
        assert scenario.provenance.time_window is not None
        assert scenario.funding.trigger.subject == "SPY"
        assert "SPY" in scenario.allocation_rule.assets

    def test_the_scheduled_control_cannot_detect_the_defect(self, priceable):
        """Stated as a test so it is a recorded fact rather than a belief.

        A scheduled policy is built from the cadence and the amount alone. It
        is identical with and without a priceable set, so a suite made of
        prompts like this would have reported F11 as absent.
        """
        assert (compiled(SCHEDULED, priceable).semantic_digest
                == compiled(SCHEDULED, ()).semantic_digest)


# --------------------------------------------------------------------------
# 2. The digest covers every result-changing field, and nothing else.
# --------------------------------------------------------------------------

class TestTheDigestCoversWhatMatters:
    """A gate is only as good as the digest under it. Each case removes or
    changes one thing that changes the answer and requires the digest to
    move — otherwise the comparison would pass over a real divergence."""

    def test_dropping_priceable_moves_it(self, priceable):
        """F11 itself, at the compiler."""
        for prompt in (CONTROL, SPLIT_ROLES):
            assert (compiled(prompt, priceable).semantic_digest
                    != compiled(prompt, ()).semantic_digest), prompt

    def test_it_is_stable_under_the_same_inputs(self, priceable):
        """Or the gate would fire when nothing had changed, and be removed."""
        assert (compiled(CONTROL, priceable).semantic_digest
                == compiled(CONTROL, priceable).semantic_digest)

    @pytest.mark.parametrize("name,mutate", [
        ("the funding policy",
         lambda s: dataclasses.replace(s, funding=None)),
        ("the contribution amount",
         lambda s: dataclasses.replace(
             s, funding=dataclasses.replace(s.funding, amount=1))),
        ("the watched asset",
         lambda s: dataclasses.replace(
             s, funding=dataclasses.replace(
                 s.funding,
                 trigger=dataclasses.replace(s.funding.trigger,
                                             subject="VOO")))),
        ("the moving-average window",
         lambda s: dataclasses.replace(
             s, funding=dataclasses.replace(
                 s.funding,
                 trigger=dataclasses.replace(s.funding.trigger, window=50)))),
        ("the execution timing",
         lambda s: dataclasses.replace(
             s, funding=dataclasses.replace(
                 s.funding, execution_timing=_other_timing(s)))),
        ("the held asset",
         lambda s: dataclasses.replace(
             s, allocation_rule=dataclasses.replace(
                 s.allocation_rule, assets=("VOO",)))),
        ("the time window",
         lambda s: dataclasses.replace(
             s, provenance=dataclasses.replace(s.provenance,
                                               time_window=None))),
        ("a rule step",
         lambda s: dataclasses.replace(s, event_program=[])),
        # Added rather than removed: the control carries none of these, and a
        # mutation that changes nothing proves nothing about the digest. The
        # `changed == scenario` guard below would catch that, but stating it
        # in the direction that works is clearer than relying on the guard.
        ("an exclusion", lambda s: _with_provenance(
            s, excluded=(_an_exclusion(),))),
        ("an asset resolution", lambda s: _with_provenance(
            s, asset_resolutions=(_a_resolution(),))),
        ("an amendment",
         lambda s: dataclasses.replace(
             s, provenance=dataclasses.replace(s.provenance, amended=()))),
        ("a declared inference",
         lambda s: dataclasses.replace(
             s, provenance=dataclasses.replace(s.provenance, inferred=()))),
        ("the tax treatment",
         lambda s: dataclasses.replace(s, tax_treatment="ROTH")),
    ])
    def test_a_result_changing_field_moves_it(self, priceable, name, mutate):
        scenario = compiled(CONTROL, priceable)
        changed = mutate(scenario)
        if changed == scenario:                              # pragma: no cover
            pytest.fail(f"the mutation for {name} changed nothing; this case "
                        f"proves nothing about the digest")
        assert changed.semantic_digest != scenario.semantic_digest, (
            f"the digest does not cover {name}; a save that lost it would "
            f"pass the equivalence gate")

    @pytest.mark.parametrize("name,mutate", [
        ("a different title",
         lambda s: dataclasses.replace(s, name="something else entirely")),
        ("a different version number",
         lambda s: dataclasses.replace(s, version=7)),
        ("a generated identifier",
         lambda s: dataclasses.replace(s, intent_ref="intent-9f2c")),
        ("the amendment timestamps",
         lambda s: dataclasses.replace(
             s, provenance=dataclasses.replace(
                 s.provenance,
                 amended=tuple(dataclasses.replace(one, recorded_at="later")
                               for one in s.provenance.amended)))),
        ("the order of unordered fields",
         lambda s: dataclasses.replace(
             s, provenance=dataclasses.replace(
                 s.provenance,
                 asset_resolutions=tuple(
                     reversed(s.provenance.asset_resolutions or ())),
                 inferred=tuple(reversed(s.provenance.inferred))))),
    ])
    def test_presentation_does_not_move_it(self, priceable, name, mutate):
        """A gate that tripped on any of these would be switched off within a
        week, and then it would catch nothing at all."""
        scenario = compiled(CONTROL, priceable)
        assert mutate(scenario).semantic_digest == scenario.semantic_digest, (
            f"the digest depends on {name}, which cannot change a result")


def _other_timing(scenario):
    from src.mission.funding import ExecutionTiming

    return next(one for one in ExecutionTiming
                if one != scenario.funding.execution_timing)


def _with_provenance(scenario, **fields):
    return dataclasses.replace(
        scenario, provenance=dataclasses.replace(scenario.provenance, **fields))


def _an_exclusion():
    from src.mission.spec import ScenarioExclusion

    return ScenarioExclusion(item="employer matching contributions",
                             reason="no representation for a match schedule")


def _a_resolution():
    from src.mission.spec import AssetResolution

    return AssetResolution(observed_phrase="SP500 ETF", registry_digest="d1",
                           chosen_instrument_id="SPY")


# --------------------------------------------------------------------------
# 3. The comparison of inputs, which decides whether the gate may conclude.
# --------------------------------------------------------------------------

class TestTheGateChecksItsOwnPremise:
    """The gate replays the preview's stated inputs. Two things must hold:
    the statement identifies the same compile across requests, and anything
    it cannot read is reported as unchecked rather than as agreement."""

    @pytest.fixture
    def stated(self):
        return draft.DraftInputs.of(amendments=ANSWERS).encode()

    def test_timestamps_do_not_count_as_a_different_input_set(self):
        """`recorded_at` is stamped when the form is handled, so if it counted
        the token would never match its own inputs and the gate would report
        `NOT_COMPARED` on every save, for ever, while looking like it worked."""
        later = tuple(dataclasses.replace(one, recorded_at="much later")
                      for one in ANSWERS)
        assert (draft.input_digest(CONTROL, parsed=None, amendments=ANSWERS)
                == draft.input_digest(CONTROL, parsed=None, amendments=later))

    def test_field_order_does_not_count_as_a_different_input_set(self):
        assert (draft.input_digest(CONTROL, parsed=None, amendments=ANSWERS)
                == draft.input_digest(CONTROL, parsed=None,
                                      amendments=tuple(reversed(ANSWERS))))

    @pytest.mark.parametrize("name,kwargs", [
        ("the description", {"describe": SPLIT_ROLES}),
        ("an answer", {"amendments": ANSWERS[:1]}),
        ("a changed answer", {"amendments": (
            dataclasses.replace(ANSWERS[0], answer="TAX_DEFERRED"),
            ANSWERS[1])}),
    ])
    def test_real_new_input_is_recognised(self, name, kwargs):
        base = dict(describe=CONTROL, parsed=None, amendments=ANSWERS)
        assert draft.input_digest(**base) != draft.input_digest(
            **{**base, **kwargs}), f"{name} is invisible to the premise check"

    def test_the_stated_inputs_survive_a_round_trip(self):
        encoded = draft.DraftInputs.of(amendments=ANSWERS).encode()
        assert draft.DraftInputs.decode(encoded).as_amendments("t") == tuple(
            sorted(ANSWERS, key=lambda one: (one.question_id, one.answer)))

    def check(self, token, stated, priceable):
        return draft.check(token, stated, CONTROL, parsed=None, name="plan",
                           at="t", context="equivalence")

    def test_a_replay_of_the_same_inputs_agrees(self, priceable, stated):
        scenario = compiled(CONTROL, priceable)
        token = draft.token_for(scenario, CONTROL, parsed=None,
                                amendments=ANSWERS)
        assert self.check(token, stated, priceable).state == \
            draft.DraftCheck.AGREED

    def test_a_replay_that_reaches_another_scenario_diverges(self, priceable,
                                                             stated):
        """The condition F11 met: same inputs, different compile. Forged here
        by claiming a semantic digest the replay cannot reproduce."""
        honest = draft.DraftToken.decode(
            draft.token_for(compiled(CONTROL, priceable), CONTROL,
                            parsed=None, amendments=ANSWERS))
        wrong = draft.DraftToken(inputs=honest.inputs,
                                 semantic=compiled(CONTROL, ()).semantic_digest)
        outcome = self.check(wrong.encode(), stated, priceable)
        assert outcome.diverged
        assert outcome.reason

    @pytest.mark.parametrize("token", [
        "", "garbage", "draft-token@1:onlytwo", "other@9:a:b",
        "draft-token@1::", "draft-token@1:a:",
    ])
    def test_an_unreadable_token_is_not_reported_as_agreement(self, token,
                                                              priceable,
                                                              stated):
        """The vacuity trap. A submission carrying no readable claim about
        what was shown has not been checked, and must not count as checked."""
        outcome = self.check(token, stated, priceable)
        assert outcome.state == draft.DraftCheck.NOT_COMPARED
        assert not outcome.diverged
        assert outcome.reason

    @pytest.mark.parametrize("stated", ["", "not json", "[]", "null",
                                        '{"amendments": "wrong"}'])
    def test_unreadable_inputs_are_not_reported_as_agreement(self, stated,
                                                             priceable):
        token = draft.token_for(compiled(CONTROL, priceable), CONTROL,
                                parsed=None, amendments=ANSWERS)
        assert self.check(token, stated, priceable).state == \
            draft.DraftCheck.NOT_COMPARED

    def test_inputs_that_contradict_the_token_are_not_agreement(self,
                                                                priceable):
        """The two hidden fields are one claim. If they disagree with each
        other, nothing follows from either."""
        token = draft.token_for(compiled(CONTROL, priceable), CONTROL,
                                parsed=None, amendments=ANSWERS)
        elsewhere = draft.DraftInputs.of(amendments=ANSWERS[:1]).encode()
        assert self.check(token, elsewhere, priceable).state == \
            draft.DraftCheck.NOT_COMPARED


# --------------------------------------------------------------------------
# 4. The live path: the page emits the token, the save reads it, and the
#    comparison actually happens on a real journey.
# --------------------------------------------------------------------------

class TestTheLivePath:
    """Everything above runs the compiler and `draft` directly. F11 lived
    between the compiler and the page, so the claim has to be made where the
    product is."""

    @pytest.fixture
    def client(self, tmp_path, deployment, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api
        import src.workspace.routes as routes
        from src.workspace.store import WorkspaceStore

        class Refusing:
            """The parse is pinned after the first GET; anything that reaches
            the provider again would be a second reading of the same text."""

            def complete(self, *, system, user):
                raise TimeoutError("not answered")

        store = WorkspaceStore(tmp_path / "w.db")
        monkeypatch.setattr(routes, "_parser_client", lambda: Refusing())
        monkeypatch.setattr(routes, "_store", lambda: store)
        api._bootstrap()
        return TestClient(api.app), store

    @pytest.fixture
    def observed(self, monkeypatch):
        """Every outcome the gate reached, in order.

        Without this the live tests could pass while the gate returned
        NOT_COMPARED on every request — a save that was never examined is
        indistinguishable from a save that agreed, if you only look at the
        status code.
        """
        import src.workspace.routes as routes

        seen = []
        real = draft.check

        def recording(*args, **kwargs):
            outcome = real(*args, **kwargs)
            seen.append(outcome.state)
            return outcome

        monkeypatch.setattr(routes.draft, "check", recording)
        return seen

    @staticmethod
    def hidden(body):
        """Every hidden input, as a browser would carry them.

        The earlier harness picked out `parse` by name and dropped the rest,
        which discarded prior answers and read as a product that forgot them.
        Taking the whole set is both more faithful and how the draft token
        reaches the save without the test knowing it exists.
        """
        return {name: html_module.unescape(value) for name, value in
                re.findall(r'<input type="hidden" name="([^"]+)" '
                           r'value="([^"]*)"', body)}

    def walk(self, http, describe=CONTROL, title="control"):
        body = http.get("/workspace/new", params={"describe": describe}).text
        payload = {"describe": describe, "title": title}
        payload.update(self.hidden(body))
        for _ in range(4):
            selects = dict(re.findall(
                r'<select[^>]*name="(answer:[^"]+|confirm:[^"]+)"[^>]*>'
                r'(.*?)</select>', body, re.S))
            checkboxes = dict(re.findall(
                r'<input type="checkbox" name="(confirm:[^"]+)"\s+'
                r'value="([^"]*)"', body))
            if not selects and not checkboxes:
                break
            for name, block in selects.items():
                options = [html_module.unescape(one) for one in
                           re.findall(r'<option value="([^"]*)"', block) if one]
                if options:
                    payload[name] = options[0]
            for name, value in checkboxes.items():
                payload[name] = html_module.unescape(value)
            response = http.post("/workspace/save", data=payload,
                                 follow_redirects=False)
            if response.status_code in (302, 303):
                return response, payload
            body = response.text
            payload.update(self.hidden(body))
        return response, payload

    def test_the_page_emits_a_token(self, client):
        http, _ = client
        body = http.get("/workspace/new", params={"describe": CONTROL}).text
        token = self.hidden(body).get("draft", "")
        assert draft.DraftToken.decode(token) is not None, (
            "the preview renders no draft token, so no save can be checked")

    def test_the_journey_saves(self, client, observed):
        """The premise for everything below. A gate on a journey that cannot
        complete proves nothing."""
        http, _ = client
        response, _ = self.walk(http)
        assert response.status_code in (302, 303), response.text[:400]

    def test_the_gate_actually_compared_something(self, client, observed):
        """Not merely 'the save succeeded'."""
        http, _ = client
        self.walk(http)
        assert draft.DraftCheck.AGREED in observed, (
            f"the gate never reached a comparison: {observed or 'never called'}"
            f" — a save it did not examine is not a save that agreed")
        assert draft.DraftCheck.DIVERGED not in observed

    def test_the_saved_plan_is_the_one_the_page_showed(self, client):
        """Independent of the gate's own report. Read the stored body rather
        than trusting the check that let it through — F11's stored artifact
        was exactly a plan with no funding policy."""
        http, store = client
        response, _ = self.walk(http)
        plan_id = response.headers["location"].rsplit("/", 1)[-1]
        body = store.get_plan(plan_id, "pilot")["scenario"]
        funding = (body.get("flows") or {}).get("funding") or {}
        assert funding.get("kind") == "EVENT_TRIGGERED", (
            "the saved plan has no event funding — the F11 shape, out of the "
            "route that is supposed to prevent it")
        assert (funding.get("trigger") or {}).get("subject") == "SPY"

    def test_a_forged_token_is_refused_rather_than_believed(self, client):
        """The token is a tripwire, not an authority. Tampering can suppress
        the check on your own submission or trip it; it can never introduce a
        value into the plan, because nothing is read back out of it."""
        http, store = client
        body = http.get("/workspace/new", params={"describe": CONTROL}).text
        honest = draft.DraftToken.decode(self.hidden(body)["draft"])
        forged = draft.DraftToken(inputs=honest.inputs,
                                  semantic="0" * 64).encode()

        response, _ = self._walk_with(http, lambda p: p.update(draft=forged))
        assert response.status_code == 409
        assert draft.DRAFT_DIVERGED in response.text
        assert not store.list_plans("pilot")

    def _walk_with(self, http, mutate, describe=CONTROL):
        body = http.get("/workspace/new", params={"describe": describe}).text
        payload = {"describe": describe, "title": "control"}
        payload.update(self.hidden(body))
        response = None
        for _ in range(4):
            selects = dict(re.findall(
                r'<select[^>]*name="(answer:[^"]+|confirm:[^"]+)"[^>]*>'
                r'(.*?)</select>', body, re.S))
            checkboxes = dict(re.findall(
                r'<input type="checkbox" name="(confirm:[^"]+)"\s+'
                r'value="([^"]*)"', body))
            if not selects and not checkboxes:
                break
            for name, block in selects.items():
                options = [html_module.unescape(one) for one in
                           re.findall(r'<option value="([^"]*)"', block) if one]
                if options:
                    payload[name] = options[0]
            for name, value in checkboxes.items():
                payload[name] = html_module.unescape(value)
            mutate(payload)
            response = http.post("/workspace/save", data=payload,
                                 follow_redirects=False)
            if response.status_code in (302, 303, 409):
                return response, payload
            body = response.text
            payload.update(self.hidden(body))
        return response, payload


# --------------------------------------------------------------------------
# 5. Falsification: reintroduce F11 and require the live path to refuse.
# --------------------------------------------------------------------------

class TestReintroducingTheDefectIsCaught(TestTheLivePath):
    """The mutations are applied to the save path only, which is the exact
    asymmetry F11 had. Each must produce a refusal, not a save."""

    def falsify(self, monkeypatch, *, when, then):
        """Make `priceable_for` answer differently for the save context.

        This is F11's mechanism, reproduced: the preview and the save were
        given different views of what could be priced. Patching the shared
        resolver by context is the only way to recreate that now, because
        there is no longer a second call site to edit — which is itself the
        point of the structural fix.
        """
        real = draft.priceable_for

        def divergent(context):
            return then if when in context else real(context)

        monkeypatch.setattr(draft, "priceable_for", divergent)

    def test_the_save_seeing_nothing_priceable_is_refused(self, client,
                                                          monkeypatch,
                                                          observed):
        http, store = client
        self.falsify(monkeypatch, when="save", then=())
        response, _ = self.walk(http)
        assert response.status_code == 409, (
            f"the save recompiled without a priceable set and was not "
            f"refused (got {response.status_code}); the gate is inert")
        assert draft.DRAFT_DIVERGED in response.text
        assert draft.DraftCheck.DIVERGED in observed

    def test_the_save_seeing_a_narrowed_universe_is_refused(self, client,
                                                            monkeypatch):
        """Not only the empty case. A save that could price the held asset but
        not the watched one loses the trigger and keeps the holding."""
        http, _ = client
        self.falsify(monkeypatch, when="save", then=("VOO", "VTI"))
        response, _ = self.walk(http, describe=SPLIT_ROLES, title="split")
        assert response.status_code == 409, response.status_code

    def test_nothing_was_stored_when_it_refused(self, client, monkeypatch):
        """A refusal that still wrote the plan would be worse than no gate —
        it would refuse and persist the wrong version."""
        http, store = client
        self.falsify(monkeypatch, when="save", then=())
        self.walk(http)
        written = store.list_plans("pilot")
        assert not written, f"{len(written)} plan(s) written by a refused save"

    def test_the_unmutated_journey_still_saves(self, client, observed):
        """The control for the two above. Without it, a gate that refused
        every save would pass both of them."""
        http, _ = client
        response, _ = self.walk(http)
        assert response.status_code in (302, 303), response.text[:400]
