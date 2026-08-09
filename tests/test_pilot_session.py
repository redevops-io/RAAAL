"""The participant token, the transcript store, and the line between them.

Two stores exist because they hold different things for different reasons:

    pilot_events        counts and codes, no prose, needs no permission
    pilot_transcripts   what someone typed, kept only if declared

Most of this file is about that boundary holding in both directions — prose
never reaching the counts, and the counts never becoming unavailable just
because a deployment declined to keep prose.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

SENTENCE = "invest $500 monthly"
REVISED = "contribute $500 monthly, rebalanced annually"
NEW = "/workspace/new"


def _client(monkeypatch, tmp_path, *, transcripts: str = ""):
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/s.db")
    if transcripts:
        monkeypatch.setenv("QUANTIFY_PILOT_TRANSCRIPTS", transcripts)
    else:
        monkeypatch.delenv("QUANTIFY_PILOT_TRANSCRIPTS", raising=False)

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.api import app

    return TestClient(app)


@pytest.fixture
def counting_only(monkeypatch, tmp_path):
    """The default deployment: counts, no prose."""
    return _client(monkeypatch, tmp_path)


@pytest.fixture
def keeping_words(monkeypatch, tmp_path):
    """A deployment that declared transcript retention."""
    return _client(monkeypatch, tmp_path, transcripts="yes")


class TestRetentionIsDeclaredAndOffByDefault:
    def test_a_deployment_that_says_nothing_keeps_nothing(self, counting_only):
        counting_only.get(NEW, params={"describe": SENTENCE})

        from src.workspace.pilot_session import every_participant, retained

        assert retained() is False
        assert not every_participant(), (
            "a deployment that never declared transcript retention kept what "
            "somebody typed")

    def test_declaring_it_keeps_the_sentence_verbatim(self, keeping_words):
        keeping_words.get(NEW, params={"describe": SENTENCE})

        from src.workspace.pilot_session import every_participant, transcript

        who = every_participant()
        assert len(who) == 1
        entries = transcript(who[0])
        assert [e["text"] for e in entries] == [SENTENCE], (
            "the point of retaining prose is quoting it accurately later; a "
            "normalised or truncated copy would not be the person's sentence")

    @pytest.mark.parametrize("raw", ["", "no", "false", "0", "maybe", "off"])
    def test_only_an_explicit_yes_turns_it_on(self, raw):
        """A typo must fail towards not keeping people's words."""
        from src.deploy.context import _affirmative

        assert _affirmative(raw) is False

    @pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "YES", " True "])
    def test_and_the_affirmatives_all_work(self, raw):
        from src.deploy.context import _affirmative

        assert _affirmative(raw) is True


class TestCountingSurvivesWithoutProse:
    """The failure this guards: tying the usability signal to prose retention.

    Resubmission is the most informative thing the pilot counts and it needs no
    permission from anyone. If it only worked in deployments that also kept
    people's words, the default deployment would be blind to it.
    """

    def test_resubmission_is_counted_with_transcripts_off(self, counting_only):
        counting_only.get(NEW, params={"describe": SENTENCE})
        counting_only.get(NEW, params={"describe": REVISED})

        from src.workspace.pilot_events import PLAN_RESUBMITTED, every_event

        resubmits = [e for e in every_event() if e["kind"] == PLAN_RESUBMITTED]
        assert len(resubmits) == 1
        assert resubmits[0]["attempt"] == 2

    def test_but_it_reports_unknown_rather_than_unchanged(self, counting_only):
        """`None`, not `False`.

        `False` reads in a summary as "they resubmitted the identical
        sentence", which is a real and interesting finding. Claiming it from a
        deployment that stored no previous sentence to compare against would be
        an assertion nothing checked.
        """
        counting_only.get(NEW, params={"describe": SENTENCE})
        counting_only.get(NEW, params={"describe": REVISED})

        from src.workspace.pilot_events import PLAN_RESUBMITTED, every_event

        resubmit = [e for e in every_event()
                    if e["kind"] == PLAN_RESUBMITTED][0]
        assert resubmit["text_changed"] is None

    def test_and_says_so_truthfully_when_it_can_compare(self, keeping_words):
        keeping_words.get(NEW, params={"describe": SENTENCE})
        keeping_words.get(NEW, params={"describe": REVISED})

        from src.workspace.pilot_events import PLAN_RESUBMITTED, every_event

        resubmit = [e for e in every_event()
                    if e["kind"] == PLAN_RESUBMITTED][0]
        assert resubmit["text_changed"] is True

    def test_and_a_verbatim_resend_is_still_a_resubmission(self, keeping_words):
        """Someone who resends the same sentence has still been round the loop.

        Counting only reworded submissions would hide the person who read the
        question, did not understand what was wanted, and tried again
        identically — which is a worse experience than the one that gets
        counted, not a lesser one.
        """
        keeping_words.get(NEW, params={"describe": SENTENCE})
        keeping_words.get(NEW, params={"describe": SENTENCE})

        from src.workspace.pilot_events import PLAN_RESUBMITTED, every_event

        resubmits = [e for e in every_event() if e["kind"] == PLAN_RESUBMITTED]
        assert len(resubmits) == 1
        assert resubmits[0]["text_changed"] is False


class TestTheTokenLinksAttemptsAndNothingElse:
    def test_two_participants_are_two_chains(self, monkeypatch, tmp_path):
        """One shared database, two browsers.

        Without this, ten attempts is one struggling person or ten easy
        successes, and the summary cannot tell those apart.
        """
        first = _client(monkeypatch, tmp_path)
        first.get(NEW, params={"describe": SENTENCE})
        first.get(NEW, params={"describe": REVISED})

        second = _client(monkeypatch, tmp_path)
        second.cookies.clear()
        second.get(NEW, params={"describe": SENTENCE})

        from src.workspace.pilot_events import (PLAN_RESUBMITTED, every_event,
                                                summary)

        assert summary()["participants"] == 2
        resubmits = [e for e in every_event() if e["kind"] == PLAN_RESUBMITTED]
        assert len(resubmits) == 1, (
            "the second browser's first attempt was counted as a revision of "
            "the first browser's")

    def test_the_token_is_not_derived_from_the_request(self):
        """Random, not a fingerprint.

        A token derived from an address or a user agent would be stable across
        browsers and reconstructible without the cookie — identifying in a way
        nobody agreed to, and no longer deletable by the person it identifies.
        """
        from src.workspace.pilot_session import new_participant

        assert len({new_participant() for _ in range(50)}) == 50

    def test_it_carries_nothing_about_anyone(self, counting_only):
        counting_only.get(NEW, params={"describe": SENTENCE})

        from src.workspace.pilot_session import COOKIE

        token = counting_only.cookies.get(COOKIE)
        assert token and token.startswith("p-")
        assert SENTENCE not in token
        assert "@" not in token


class TestTheProseAndTheCountsStayApart:
    def test_the_transcript_holds_what_the_events_refuse_to(self,
                                                            keeping_words):
        """Both halves in one test, because either alone is satisfiable the
        wrong way: events with no prose and no transcript is just amnesia, and
        a transcript whose text also leaked into the events is the defect."""
        keeping_words.get(NEW, params={"describe": SENTENCE})

        from src.workspace.pilot_events import every_event
        from src.workspace.pilot_session import every_participant, transcript

        kept = transcript(every_participant()[0])
        assert any(SENTENCE in e["text"] for e in kept)

        for event in every_event():
            assert SENTENCE not in " ".join(str(v) for v in event.values())

    def test_the_two_tables_agree_on_the_attempt_number(self, keeping_words):
        """One source for the number.

        A transcript-local counter would drift from the event-derived one in
        any deployment that had retention off for a while, and a chain read
        beside the counts would show attempt 2 next to attempt 5 for one
        action.
        """
        keeping_words.get(NEW, params={"describe": SENTENCE})
        keeping_words.get(NEW, params={"describe": REVISED})

        from src.workspace.pilot_events import PLAN_RESUBMITTED, every_event
        from src.workspace.pilot_session import every_participant, transcript

        kept = transcript(every_participant()[0])
        assert [e["attempt"] for e in kept] == [1, 2]

        resubmit = [e for e in every_event()
                    if e["kind"] == PLAN_RESUBMITTED][0]
        assert resubmit["attempt"] == kept[-1]["attempt"]


class TestThePromiseAboutRetentionIsKept:
    def test_forgetting_a_participant_removes_their_words(self, keeping_words):
        """Someone in a ten-person study who asks to be removed is asking a
        person, not a form — but the answer must not be "we would have to write
        something"."""
        keeping_words.get(NEW, params={"describe": SENTENCE})

        from src.workspace.pilot_session import (every_participant, forget,
                                                 transcript)

        who = every_participant()[0]
        assert forget(who) == 1
        assert transcript(who) == []

    def test_expiry_honours_the_declared_window(self, keeping_words):
        """A retention setting nothing enforces is a promise, and this pilot
        makes that promise to about ten people in person."""
        from datetime import datetime, timedelta, timezone

        keeping_words.get(NEW, params={"describe": SENTENCE})

        from src.deploy.context import current
        from src.workspace.pilot_session import (every_participant, expire,
                                                 transcript)

        who = every_participant()[0]
        days = current().study.retention_days

        assert expire(now=datetime.now(timezone.utc)) == 0, (
            "a fresh transcript was expired inside its own window")

        later = datetime.now(timezone.utc) + timedelta(days=days + 1)
        assert expire(now=later) == 1
        assert transcript(who) == []


class TestDepartureMeansLeaving:
    def test_a_participant_reaching_the_legacy_workspace_is_recorded(
            self, counting_only):
        counting_only.get(NEW, params={"describe": SENTENCE})
        counting_only.get("/workspace/")

        from src.workspace.pilot_events import LEFT_FOR_LEGACY, every_event

        assert [e for e in every_event() if e["kind"] == LEFT_FOR_LEGACY]

    def test_a_first_time_visitor_has_not_left_anything(self, monkeypatch,
                                                        tmp_path):
        """Someone who arrives at `/workspace/` having never been in the pilot
        is ordinary traffic. Counting them would turn every visit into evidence
        of abandonment and make the one signal that means something useless."""
        client = _client(monkeypatch, tmp_path)
        client.cookies.clear()
        client.get("/workspace/")

        from src.workspace.pilot_events import LEFT_FOR_LEGACY, every_event

        assert not [e for e in every_event() if e["kind"] == LEFT_FOR_LEGACY]

    def test_the_pilot_page_itself_is_not_a_departure(self, counting_only):
        """`/workspace/new` serves the pilot under this mode. Recording it
        would report every single participant as having left immediately."""
        counting_only.get(NEW, params={"describe": SENTENCE})
        counting_only.get(NEW, params={"describe": SENTENCE})

        from src.workspace.pilot_events import LEFT_FOR_LEGACY, every_event

        assert not [e for e in every_event() if e["kind"] == LEFT_FOR_LEGACY]

    @pytest.mark.parametrize("path,left", [
        ("/workspace/new", False),
        ("/pilot", False),
        ("/pilot/plans/plan-abc", False),
        ("/workspace/", True),
        ("/workspace/plans/7", True),
        ("/workspace/newsletter", True),
        ("/health/live", False),
    ])
    def test_which_paths_count(self, path, left):
        """`/workspace/newsletter` is the one that matters: a prefix test
        against `/workspace/new` would call it a pilot page forever."""
        from src.workspace.pilot_session import is_a_departure

        assert is_a_departure(path) is left

    def test_the_destination_recorded_is_a_route_not_a_url(self,
                                                           counting_only):
        """A path a user composed can carry their own words in a query string
        or an id, and nothing in the events table may."""
        counting_only.get(NEW, params={"describe": SENTENCE})
        counting_only.get("/workspace/", params={"q": SENTENCE})

        from src.workspace.pilot_events import LEFT_FOR_LEGACY, every_event

        for event in every_event():
            if event["kind"] == LEFT_FOR_LEGACY:
                assert SENTENCE not in event["destination"]
                assert "?" not in event["destination"]


class TestTheUnnecessaryClarificationProxy:
    def test_it_fires_when_the_answer_was_already_in_the_sentence(self):
        from src.workspace.pilot_events import answers_already_in_the_prompt

        assert answers_already_in_the_prompt(
            "invest $500 monthly into VTI", {"assets": "VTI"}) == ("assets",)

    def test_it_does_not_fire_on_a_genuine_addition(self):
        from src.workspace.pilot_events import answers_already_in_the_prompt

        assert answers_already_in_the_prompt(
            "invest $500 monthly", {"assets": "VTI"}) == ()

    def test_an_answer_does_not_match_inside_a_longer_number(self):
        """`50` is inside `$500`, and a plain substring test calls that a
        repetition.

        Two characters, not one, because the length guard already stops `5` —
        a single-digit case passes with the word boundary removed, so it proves
        the guard and says nothing about the boundary. A counter that fires on
        nothing is worse than an absent one: the transcripts it points at will
        all read fine, and the time goes into finding out why.
        """
        from src.workspace.pilot_events import answers_already_in_the_prompt

        assert answers_already_in_the_prompt(
            "invest $500 monthly", {"contribution": "50"}) == ()

    def test_and_a_one_character_answer_is_ignored_outright(self):
        from src.workspace.pilot_events import answers_already_in_the_prompt

        assert answers_already_in_the_prompt(
            "invest $500 monthly", {"horizon": "5"}) == ()

    def test_it_is_named_as_a_proxy_where_the_numbers_are_read(self):
        """The counter sits next to real findings and looks like one.

        Whoever reads `clarifications_answered_from_the_prompt` needs to be
        told in the same output that it says which transcripts to read, not
        whether a question was unnecessary.
        """
        from src.workspace.pilot_events import summary

        caveats = " ".join(summary()["not_measured_here"])
        assert "unnecessary" in caveats
        assert "proxy" in caveats
