"""Consent is about a person, not about a deployment.

The notice says *transcript recording is disabled unless you explicitly agree*.
That is a promise made to whoever is sitting there, so one deployment switch
covering ten people cannot keep it: the one who declined would be recorded
along with the nine who did not.

Everything here is a property of that promise. Fails closed, is not
retroactive, is versioned against the words actually shown, and can be
withdrawn in one call that also deletes what was already kept.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

SENTENCE = "invest $500 monthly"
REVISED = "contribute $500 monthly, rebalanced annually"
NEW = "/workspace/new"


@pytest.fixture
def study(monkeypatch, tmp_path):
    """A deployment running the study. Nobody has agreed to anything yet."""
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/c.db")
    monkeypatch.setenv("QUANTIFY_PILOT_TRANSCRIPTS", "yes")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.api import app

    return TestClient(app)


def _token(client):
    from src.workspace.pilot_session import COOKIE

    return client.cookies.get(COOKIE)


def _arrives(client):
    """Open the page, as a participant does before typing anything."""
    client.get(NEW)
    return _token(client)


class TestTheTokenExistsBeforeTheFirstSentence:
    def test_opening_the_page_issues_one(self, study):
        """Consent is recorded against a token.

        If the token only appeared on submission there would be nothing to
        record consent against until the participant's first sentence had
        already been discarded — and their unprompted first phrasing is the
        most informative thing they produce all session.
        """
        assert _arrives(study), "no token was issued before anything was typed"

    def test_so_the_first_sentence_can_be_kept(self, study):
        from src.workspace.pilot_consent import grant
        from src.workspace.pilot_session import transcript

        who = _arrives(study)
        grant(who)
        study.get(NEW, params={"describe": SENTENCE})

        assert [e["text"] for e in transcript(who)] == [SENTENCE]


class TestItFailsClosed:
    def test_a_participant_nobody_asked_is_not_recorded(self, study):
        """The deployment switch is on. That is not agreement."""
        who = _arrives(study)
        study.get(NEW, params={"describe": SENTENCE})

        from src.workspace.pilot_consent import UNKNOWN, state_of
        from src.workspace.pilot_session import transcript

        assert state_of(who) == UNKNOWN
        assert transcript(who) == []

    def test_a_participant_who_declined_is_not_recorded(self, study):
        who = _arrives(study)

        from src.workspace.pilot_consent import decline
        from src.workspace.pilot_session import transcript

        decline(who)
        study.get(NEW, params={"describe": SENTENCE})
        assert transcript(who) == []

    def test_one_decline_does_not_silence_the_others(self, monkeypatch,
                                                     tmp_path, study):
        """The failure a deployment-wide switch produces, in reverse.

        Two people in one study, one agrees and one does not, and each gets
        what they asked for. A single switch cannot express this at all — which
        is why it was the wrong mechanism for a promise made to a person.
        """
        from src.workspace.pilot_consent import decline, grant
        from src.workspace.pilot_session import transcript

        agreeing = _arrives(study)
        grant(agreeing)
        study.get(NEW, params={"describe": SENTENCE})

        study.cookies.clear()
        declining = _arrives(study)
        decline(declining)
        study.get(NEW, params={"describe": SENTENCE})

        assert [e["text"] for e in transcript(agreeing)] == [SENTENCE]
        assert transcript(declining) == []

    def test_the_deployment_switch_still_overrides_a_grant(self, monkeypatch,
                                                           tmp_path):
        """Both gates, and the deployment's comes first.

        Otherwise a developer checkout with a copied database would accumulate
        transcripts because consent rows happened to be in it.
        """
        from src.deploy import context as deploy_context

        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
        monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
        monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
        monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
        monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/d.db")
        monkeypatch.delenv("QUANTIFY_PILOT_TRANSCRIPTS", raising=False)

        resolved = deploy_context.resolve(dict(os.environ))
        monkeypatch.setattr(deploy_context, "current", lambda: resolved)

        from src.api import app
        from src.workspace.pilot_consent import GRANTED, grant, state_of
        from src.workspace.pilot_session import transcript

        client = TestClient(app)
        who = _arrives(client)
        grant(who)
        client.get(NEW, params={"describe": SENTENCE})

        assert state_of(who) == GRANTED, "the agreement itself still stands"
        assert transcript(who) == [], (
            "a deployment not running the study kept prose because a consent "
            "row existed")


class TestConsentIsNotRetroactive:
    def test_agreeing_later_does_not_reach_back(self, study):
        """Sentences typed before agreement stay unkept.

        Agreeing at the end of a session is agreeing to what happens next.
        Sweeping up the earlier prompts would be keeping words the person had
        not been asked about at the time they typed them — and there is no code
        path here that could, which is the property rather than an oversight.
        """
        who = _arrives(study)
        study.get(NEW, params={"describe": SENTENCE})       # before agreeing

        from src.workspace.pilot_consent import grant
        from src.workspace.pilot_session import transcript

        grant(who)
        study.get(NEW, params={"describe": REVISED})        # after

        kept = [e["text"] for e in transcript(who)]
        assert kept == [REVISED], (
            f"a sentence typed before consent was retained: {kept}")

    def test_and_the_attempt_number_still_counts_from_the_start(self, study):
        """The kept sentence is attempt 2, not attempt 1.

        The counts never stopped — only the prose did. A transcript renumbered
        from the first *kept* entry would read as this person's opening
        sentence, which is the one thing it is not.
        """
        who = _arrives(study)
        study.get(NEW, params={"describe": SENTENCE})

        from src.workspace.pilot_consent import grant
        from src.workspace.pilot_session import transcript

        grant(who)
        study.get(NEW, params={"describe": REVISED})

        assert [e["attempt"] for e in transcript(who)] == [2]


class TestTheWordingIsVersioned:
    def test_the_notice_shown_is_the_notice_recorded(self, study):
        """A consent record naming a version whose text nobody can produce is
        not evidence of consent."""
        from src.workspace.pilot_consent import (NOTICE, NOTICE_VERSION, grant,
                                                 record_of)

        who = _arrives(study)
        grant(who)

        assert NOTICE.strip()
        assert record_of(who)["notice_version"] == NOTICE_VERSION

    def test_a_grant_against_superseded_wording_stops_counting(self,
                                                               monkeypatch,
                                                               study):
        """Somebody who agreed to one notice has not agreed to a later one.

        Treating an old yes as a current one is how a study ends up holding
        words under terms nobody accepted.
        """
        from src.workspace import pilot_consent
        from src.workspace.pilot_consent import UNKNOWN, grant, state_of
        from src.workspace.pilot_session import transcript

        who = _arrives(study)
        grant(who)

        monkeypatch.setattr(pilot_consent, "NOTICE_VERSION", "2099-01-01.1")
        assert state_of(who) == UNKNOWN

        study.get(NEW, params={"describe": SENTENCE})
        assert transcript(who) == [], (
            "prose was kept under a notice this participant never saw")

    def test_the_notice_states_what_the_code_actually_does(self):
        """The promises in the text, checked against the mechanism.

        A notice is the one document in this project that a non-engineer reads
        and relies on, so its claims are asserted rather than trusted.
        """
        from src.deploy.context import StudyTarget
        from src.workspace import pilot_consent
        from src.workspace.pilot_consent import NOTICE

        # Whitespace-normalised: where the notice wraps is a formatting choice
        # and must not be something a test pins down.
        said = " ".join(NOTICE.split())

        assert "30 days" in said
        assert StudyTarget().retention_days == 30, (
            "the notice promises 30 days and the default says otherwise")

        assert "deleted on request" in said
        assert callable(pilot_consent.withdraw)

        assert "unless you explicitly agree" in said
        assert StudyTarget().retain_transcripts is False


class TestWithdrawalIsOnePromise:
    def test_it_revokes_and_deletes_together(self, study):
        """"You can have it deleted at any time" is a single promise, and
        splitting it across two calls is how half of it gets kept."""
        from src.workspace.pilot_consent import (DECLINED, grant, state_of,
                                                 withdraw)
        from src.workspace.pilot_session import transcript

        who = _arrives(study)
        grant(who)
        study.get(NEW, params={"describe": SENTENCE})
        assert transcript(who)

        assert withdraw(who) == 1
        assert transcript(who) == []
        assert state_of(who) == DECLINED

    def test_and_nothing_is_kept_afterwards(self, study):
        """Withdrawal that only deleted the backlog would refill on the next
        sentence."""
        from src.workspace.pilot_consent import grant, withdraw
        from src.workspace.pilot_session import transcript

        who = _arrives(study)
        grant(who)
        study.get(NEW, params={"describe": SENTENCE})
        withdraw(who)

        study.get(NEW, params={"describe": REVISED})
        assert transcript(who) == []


class TestAnEmptyStoreIsNeverAmbiguous:
    def test_the_operator_can_tell_declines_from_nobody_asked(self, study):
        """Zero transcripts reads as eight refusals or as eight people nobody
        asked, and those say opposite things about whether the protocol was
        followed. The same ambiguity as zero events versus zero usage.
        """
        from src.workspace.pilot_consent import (DECLINED, GRANTED, UNKNOWN,
                                                 by_state, decline, grant)

        first = _arrives(study)
        grant(first)
        study.get(NEW, params={"describe": SENTENCE})

        study.cookies.clear()
        second = _arrives(study)
        decline(second)
        study.get(NEW, params={"describe": SENTENCE})

        study.cookies.clear()
        third = _arrives(study)
        study.get(NEW, params={"describe": SENTENCE})

        grouped = by_state()
        assert grouped[GRANTED] == [first]
        assert grouped[DECLINED] == [second]
        assert grouped[UNKNOWN] == [third]
