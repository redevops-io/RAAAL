"""What Discovery may send to a hosted model, checked rather than intended.

The boundary Phase 3 rests on:

    hosted provider    sees  the user's own words, and a schema
    Mission/execution  sees  verified structured intent, and financial data

A policy nobody enforces is a paragraph. These tests read
`data/licensing/discovery-egress@1.yaml` and assert the shape a Discovery
implementation has to satisfy, so the record cannot quietly become decorative —
which is the failure mode `agentic-os` already has with its free-text
`constraints`, enforced by substring matching or not at all.

There is no Discovery Runtime yet. That is deliberate and these tests still
earn their place: the record is written before the code, so the code is
written against it rather than the record being back-filled to describe
whatever got built.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

RECORD = Path(__file__).resolve().parent.parent / "data" / "licensing" / \
    "discovery-egress@1.yaml"


@pytest.fixture(scope="module")
def policy():
    assert RECORD.exists(), (
        "the egress record is missing; Phase 3 must not ship without it")
    return yaml.safe_load(RECORD.read_text())


class TestTheRecordIsComplete:
    def test_it_names_a_reviewer_and_a_date(self, policy):
        """A decision with no reviewer and no date is one nobody can revisit."""
        assert policy["reviewer"] and policy["decided_on"]

    def test_it_names_the_model_and_says_it_is_hosted(self, policy):
        reader = policy["reader"]
        assert reader["model"] and reader["provider"]
        assert reader["hosted"] is True, (
            "a hosted reader is the thing being authorised; recording it as "
            "local would authorise nothing while looking like it did")

    def test_every_permission_gives_a_reason(self, policy):
        for section in ("may_send", "may_not_send"):
            for name, entry in policy[section].items():
                assert entry.get("detail"), f"{section}.{name} has no reason"


class TestFinancialDataMayNotLeave:
    """The distinction from the market-data record. That one governs prices;
    this governs the user's language. Neither answer implies the other."""

    @pytest.mark.parametrize("forbidden", [
        "price_series", "simulation_output", "portfolio_state",
        "retrieved_financial_history", "other_users_data"])
    def test_it_is_refused(self, policy, forbidden):
        assert policy["may_not_send"][forbidden]["permitted"] is False

    def test_only_language_and_schema_may_be_sent(self, policy):
        """The discriminating half: a record that forbade everything would
        also forbid reading, and Discovery would be unimplementable."""
        permitted = {k for k, v in policy["may_send"].items()
                     if v["permitted"] is True}
        assert permitted == {"user_utterance", "intent_schema",
                             "interpretation_instructions"}


class TestTheManifestMayNotReachTheReader:
    """Not a privacy rule — a correctness one, which is why it sits in the
    same list. Telling the reader which values the engine executes does not
    make it refuse the others; it makes it render them as the nearest thing it
    can say, so "by inverse volatility" comes back as an equal split."""

    def test_it_is_refused_alongside_the_data(self, policy):
        assert policy["may_not_send"]["capability_manifest"]["permitted"] is False

    def test_and_the_reason_given_is_the_correctness_one(self, policy):
        detail = policy["may_not_send"]["capability_manifest"]["detail"].lower()
        assert "refuse" in detail and "nearest" in detail


class TestTheConditionsAreStated:
    def test_shadow_first(self, policy):
        joined = " ".join(policy["conditions"]).lower()
        assert "shadow" in joined

    def test_the_user_is_told_before_it_happens(self, policy):
        """Authorising the egress is not authorising doing it silently."""
        joined = " ".join(policy["conditions"]).lower()
        assert "told" in joined or "inform" in joined

    def test_revisiting_is_required_before_the_obvious_next_steps(self, policy):
        joined = " ".join(policy["conditions"]).lower()
        for trigger in ("local model", "context runtime", "second provider"):
            assert trigger in joined, f"no revisit condition for {trigger}"


class TestTheReaderIsNotPermanentlySpecial:
    def test_the_record_says_the_adapter_is_provider_neutral(self, policy):
        detail = " ".join(str(v) for v in policy["reader"].values()).lower()
        assert "provider-neutral" in detail or "not permanent" in detail
