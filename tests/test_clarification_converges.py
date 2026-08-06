"""The clarification loop settles, and costs one model call.

Found by a deterministic journey harness walking nine descriptions through the
whole loop: five never converged in six rounds. The instrument question came
back every round under a different id, because the id was built from the
model's own account of the phrase —

    asset_identity:SP500 ETF (company/product name, not a literal ticker…)
    asset_identity:SP500 ETF (ticker symbol not specified)
    asset_identity:SP500 ETF (fund name given, not a literal ticker symbol)

— and an answer to one did not match the next. `funding_source` and
`account_type` oscillated for the same reason: `_builder_context` took the
pinned parse and re-parsed anyway, so every submission was a fresh model
reading with fresh wording.

Two causes, two invariants:

    the persistent key derives from the observed subject, never from prose
    a journey parses once

The second is not implied by the first. A stable key would settle the answer
and still leave the question set drifting under the user, so this file counts
provider calls as well as ids.

`test_the_original_prompt.py` covered this exact description throughout and
passed: it built the amendment id with the same expression the compiler used,
so the two agreed no matter what either produced.
"""
from __future__ import annotations

import pytest

from src.mission.compiler import ParsedUtterance, canonical_key, compile_scenario
from src.mission.spec import ScenarioAmendment

PROMPT = ("I buy $1,000 of SP500 ETF every time the S&P 500 crosses below its "
          "200-day moving average for the past 5 years.")

#: One question, six accounts of it. Taken from a recorded journey rather than
#: invented — these are the strings the live model actually produced across six
#: rounds for one description.
REWORDINGS = (
    "SP500 ETF (company/product name, not a literal ticker symbol)",
    "SP500 ETF (ticker symbol not specified)",
    "SP500 ETF (fund/company name, not a verified ticker symbol)",
    "SP500 ETF (company/fund name, not a literal ticker)",
    "SP500 ETF (fund name given, not a literal ticker symbol)",
    "SP500 ETF",
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


def compile_with(phrase: str, amendments=(), deployment_priceable=()):
    return compile_scenario(
        PROMPT, name="p", version=1,
        parsed=ParsedUtterance(text=PROMPT, unclear=(phrase,)),
        amendments=tuple(amendments), benchmark_rule="b",
        priceable=tuple(deployment_priceable)).scenario


def asset_questions(scenario):
    return [one.field for one in scenario.provenance.unresolved
            if one.field.startswith("asset_identity:")]


#: The same subject with the explanation carried by different punctuation.
#: The first implementation handled parentheses and dash clauses — the forms
#: the live model had produced — and let the other three through, giving a
#: different key for each. Matching observed punctuation is implementing the
#: example; these test the invariant.
PROSE_FORMS = (
    "SP500 ETF (fund name, not a ticker)",
    "SP500 ETF — this is a fund name, not a verified ticker",
    "SP500 ETF. A ticker symbol was not provided.",
    "SP500 ETF: fund name rather than a ticker",
    "SP500 ETF, which is a product name not a ticker",
    "SP500 ETF; ticker not given",
    "SP500 ETF",
)


class TestTheKeyComesFromTheSubject:
    @pytest.mark.parametrize("phrase", PROSE_FORMS)
    def test_any_punctuation_carrying_the_reason_is_cut(self, phrase):
        assert canonical_key("asset_identity", phrase) == \
            "asset_identity:sp500-etf"

    def test_a_comma_inside_a_name_is_not_a_boundary(self):
        """Stability must not become truncation. A comma ends the subject only
        when a clause follows it, or every fund with a suffix loses its name."""
        assert canonical_key("asset_identity", "SPDR S&P 500 ETF Trust, Inc") \
            != canonical_key("asset_identity", "SPDR S&P 500 ETF Trust")

    def test_every_rewording_produces_one_key(self):
        keys = {canonical_key("asset_identity", phrase)
                for phrase in REWORDINGS}
        assert len(keys) == 1, keys
        assert keys == {"asset_identity:sp500-etf"}

    def test_the_rewordings_actually_differ(self):
        """The premise. Six identical strings would collapse to one key under
        any implementation, including the broken one."""
        assert len(set(REWORDINGS)) == len(REWORDINGS)

    def test_a_different_subject_keeps_a_different_key(self):
        """Stability must not become collision: two subjects sharing one key
        would settle each other, which is worse than asking twice."""
        assert (canonical_key("asset_identity", "SP500 ETF")
                != canonical_key("asset_identity", "Nasdaq ETF"))

    def test_unclear_items_follow_the_same_rule(self):
        assert (canonical_key("unclear", "every so often (unclear cadence)")
                == canonical_key("unclear", "every so often"))

    def test_a_phrase_that_normalises_away_still_gets_a_key(self):
        """A subject of only punctuation must not produce a bare prefix that
        every other such phrase would share."""
        first = canonical_key("unclear", "(...)")
        second = canonical_key("unclear", "???")
        assert first != "unclear:" and second != "unclear:"
        assert first != second


class TestOneAnswerSettlesTheQuestion:
    def test_the_question_is_asked_before_it_is_answered(self, deployment):
        """The premise. If the compile asks nothing, every settle assertion
        below holds vacuously."""
        assert asset_questions(compile_with(REWORDINGS[0]))

    @pytest.mark.parametrize("phrase", REWORDINGS)
    def test_an_answer_from_round_one_settles_every_rewording(
            self, deployment, phrase):
        answered = compile_with(phrase, amendments=(ScenarioAmendment(
            question_id=canonical_key("asset_identity", REWORDINGS[0]),
            answer="SPY", recorded_at="t"),))
        assert not asset_questions(answered)

    def test_the_answer_reaches_the_plan(self, deployment):
        """Settled and consumed are different. A question that stops being
        asked while the instrument stays unresolved would pass the case above
        and produce a plan nobody asked for."""
        answered = compile_with(REWORDINGS[3], amendments=(ScenarioAmendment(
            question_id=canonical_key("asset_identity", REWORDINGS[0]),
            answer="SPY", recorded_at="t"),))
        assert "SPY" in answered.allocation_rule.assets


class CountingClient:
    """A parser client that records how often stage 1 asked a model."""

    def __init__(self):
        self.calls = 0

    def complete(self, *, system, user):
        self.calls += 1
        raise TimeoutError("counted, not answered")


class TestAJourneyParsesOnce:
    """A stable key settles the answer and would still leave the question set
    drifting if the model ran again each round. Counted, not assumed."""

    def test_the_first_render_parses(self, deployment, monkeypatch):
        import src.workspace.routes as routes

        client = CountingClient()
        monkeypatch.setattr(routes, "_parser_client", lambda: client)
        routes._pinned_or_parse(PROMPT, "")
        assert client.calls == 1, "the first read must reach the parser"

    def test_a_replay_calls_no_model(self, deployment, monkeypatch):
        import json

        import src.workspace.routes as routes
        from src.mission.compiler import parse

        client = CountingClient()
        monkeypatch.setattr(routes, "_parser_client", lambda: client)
        pinned = json.dumps(parse(PROMPT).to_json())
        routes._pinned_or_parse(PROMPT, pinned)
        assert client.calls == 0, (
            "the clarification loop re-read the description; the question set "
            "can drift under the user between rounds")

    def test_a_replay_reports_itself_as_a_replay(self, deployment, monkeypatch):
        """Provenance must not claim a model call that did not happen."""
        import json

        import src.workspace.routes as routes
        from src.mission.compiler import parse

        monkeypatch.setattr(routes, "_parser_client", lambda: CountingClient())
        pinned = json.dumps(parse(PROMPT).to_json())
        replayed = routes._pinned_or_parse(PROMPT, pinned)
        assert replayed.provenance.mode == "PINNED_REPLAY"
        assert replayed.provenance.model is None

    def test_a_mismatched_token_falls_back_rather_than_losing_the_page(
            self, deployment, monkeypatch):
        """A pin that does not match the description is unusable. Re-reading
        costs a call and keeps the journey; failing the render would lose
        every answer the user had given."""
        import src.workspace.routes as routes

        client = CountingClient()
        monkeypatch.setattr(routes, "_parser_client", lambda: client)
        routes._pinned_or_parse(PROMPT, '{"text": "a different plan"}')
        assert client.calls == 1


class TestUnclearItemsSettleToo:
    """The same rule, through the compiler rather than through the function.

    A test that called `canonical_key` directly passed while the compiler
    still built `unclear:` ids from the model's prose — it proved the helper
    correct and never asked whether the compiler used it. The mutation
    survived, which is the only reason this class exists.

    Unclear items are settled by exclusion rather than by answer: the user
    acknowledges that a phrase cannot be modelled and proceeds. The
    acknowledgement must survive the model rewording the phrase, or the item
    returns and the plan cannot be saved.
    """

    UNCLEAR_REWORDINGS = (
        "every so often (unclear cadence)",
        "every so often (frequency not specified)",
        "every so often",
    )

    def unclear_fields(self, phrase, exclusions=()):
        from src.mission.spec import ScenarioExclusion

        scenario = compile_scenario(
            "I put money into tech every so often.", name="p", version=1,
            parsed=ParsedUtterance(text="I put money into tech every so often.",
                                   unclear=(phrase,)),
            exclusions=tuple(exclusions), benchmark_rule="b")
        return [one.field for one in scenario.scenario.provenance.unresolved
                if one.field.startswith("unclear:")]

    def test_the_item_is_raised_before_it_is_excluded(self, deployment):
        assert self.unclear_fields(self.UNCLEAR_REWORDINGS[0])

    @pytest.mark.parametrize("phrase", UNCLEAR_REWORDINGS)
    def test_an_exclusion_survives_the_rewording(self, deployment, phrase):
        from src.mission.spec import ScenarioExclusion

        settled = ScenarioExclusion(
            item=canonical_key("unclear", self.UNCLEAR_REWORDINGS[0]),
            reason="acknowledged", acknowledged_at="t")
        assert not self.unclear_fields(phrase, exclusions=(settled,)), (
            "the acknowledgement did not match this rewording; the item "
            "returns and the plan cannot be saved")


class TestTheWholeJourneyParsesOnce:
    """Counted through the routes, not through the helper.

    The helper-level cases above passed while `_builder_context` called
    `parse_with_model` directly — the mutation bypassed the function the tests
    were watching, and every test stayed green. A count taken at the seam
    proves the seam; only a count taken across the journey proves the journey.
    """

    @pytest.fixture
    def client(self, tmp_path, deployment, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api
        import src.workspace.routes as routes
        from src.workspace.store import WorkspaceStore

        counter = CountingClient()
        monkeypatch.setattr(routes, "_parser_client", lambda: counter)
        monkeypatch.setattr(routes, "_store",
                            lambda: WorkspaceStore(tmp_path / "w.db"))
        api._bootstrap()
        return TestClient(api.app), counter

    @staticmethod
    def pinned_token(body: str) -> str:
        import html as html_module
        import re

        found = re.search(r'name="parse" value="([^"]*)"', body)
        return html_module.unescape(found.group(1)) if found else ""

    def test_the_first_screen_parses_once(self, client):
        http, counter = client
        response = http.get("/workspace/new", params={"describe": PROMPT})
        assert response.status_code == 200
        assert counter.calls == 1

    def test_a_round_trip_adds_no_further_parse(self, client):
        """The property that matters. Each submission used to be another
        provider call and another chance for the reading to drift."""
        http, counter = client
        first = http.get("/workspace/new", params={"describe": PROMPT})
        token = self.pinned_token(first.text)
        assert token, "no pinned parse was rendered; the journey cannot pin"

        before = counter.calls
        http.post("/workspace/save", data={"describe": PROMPT, "title": "t",
                                           "parse": token})
        assert counter.calls == before, (
            f"the round trip parsed again ({counter.calls - before} extra "
            f"call(s)); the question set can drift between rounds")

    def test_three_round_trips_still_parse_once(self, client):
        http, counter = client
        first = http.get("/workspace/new", params={"describe": PROMPT})
        token = self.pinned_token(first.text)
        for _ in range(3):
            response = http.post("/workspace/save",
                                 data={"describe": PROMPT, "title": "t",
                                       "parse": token})
            token = self.pinned_token(response.text) or token
        assert counter.calls == 1, (
            f"{counter.calls} provider calls for one journey")


class TestAnAnsweredWindowSettlesAndIsUsed:
    """A stable id is not a settle site.

    `moving_average_window` was added with a closed vocabulary, so its key was
    stable from the start and the answer matched — and the compiler never read
    it, so a journey harness watched the question return in every round. Same
    family as the asset-identity defect, one stage later:

        asset_identity            unstable key  -> the answer could not match
        moving_average_window     stable key    -> the answer matched, and
                                                   nothing consumed it

    Driven through the routes rather than the compiler. Two mutations in this
    slice survived helper-level tests by bypassing the helper.
    """

    DESCRIPTION = ("I buy $1,000 of VOO every time it crosses below its "
                   "moving average, over the past 5 years.")

    @pytest.fixture
    def client(self, tmp_path, deployment, monkeypatch):
        from fastapi.testclient import TestClient

        import src.api as api
        import src.workspace.routes as routes
        from src.workspace.store import WorkspaceStore

        monkeypatch.setattr(routes, "_store",
                            lambda: WorkspaceStore(tmp_path / "w.db"))
        api._bootstrap()
        return TestClient(api.app)

    @staticmethod
    def token(body: str) -> str:
        import html as html_module
        import re

        found = re.search(r'name="parse" value="([^"]*)"', body)
        return html_module.unescape(found.group(1)) if found else ""

    def first(self, client):
        response = client.get("/workspace/new",
                              params={"describe": self.DESCRIPTION})
        assert response.status_code == 200, response.text
        return response.text

    def test_the_question_is_asked(self, client):
        """The premise. A description that states its window would settle the
        field without any of this being exercised."""
        assert 'data-field="moving_average_window"' in self.first(client)

    def test_a_control_is_rendered_for_it(self, client):
        """A question with no way to answer it dead-ends the journey."""
        body = self.first(client)
        assert 'name="answer:moving_average_window"' in body

    def test_answering_removes_it_from_the_next_round(self, client):
        body = self.first(client)
        second = client.post("/workspace/save", data={
            "describe": self.DESCRIPTION, "title": "t",
            "parse": self.token(body),
            "answer:moving_average_window": "200",
            "answer:trigger_semantics": "crossing_event",
            "answer:account_type": "TAXABLE"})
        assert 'data-field="moving_average_window"' not in second.text, (
            "the answered question came back; the amendment matched and "
            "nothing consumed it")

    @pytest.mark.parametrize("window", ("50", "200"))
    def test_the_chosen_window_reaches_the_compiled_rule(self, client, window):
        """Settled and consumed are different. A question that stops being
        asked while the rule keeps its own window would pass the case above."""
        from src.mission.compiler import compile_scenario
        from src.mission.spec import ScenarioAmendment

        import src.workspace.routes as routes

        access = routes._market_data("test")
        plan = compile_scenario(
            self.DESCRIPTION, name="p", version=1,
            amendments=(
                ScenarioAmendment(question_id="trigger_semantics",
                                  answer="crossing_event", recorded_at="t"),
                ScenarioAmendment(question_id="moving_average_window",
                                  answer=window, recorded_at="t")),
            benchmark_rule="b", priceable=tuple(access.frame.columns))
        assert plan.scenario.funding.trigger.window == int(window)

    def test_a_nonsense_answer_does_not_become_a_window(self, client):
        """The value is a count of sessions. Accepting anything would let a
        free-text answer silently define the rule."""
        from src.mission.compiler import compile_scenario
        from src.mission.spec import ScenarioAmendment

        plan = compile_scenario(
            self.DESCRIPTION, name="p", version=1,
            amendments=(ScenarioAmendment(question_id="moving_average_window",
                                          answer="a while", recorded_at="t"),),
            benchmark_rule="b")
        assert any(one.field == "moving_average_window"
                   for one in plan.scenario.provenance.unresolved)
