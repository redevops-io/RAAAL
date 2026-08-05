"""A stored plan states what it offered, not what today's catalogue would.

Reopening a plan and re-resolving "SP500 ETF" against the current registry
answers with the current catalogue. The chosen ticker survives — it was stored
— and everything around it quietly changes: the alternatives the user saw, the
order they were in, and the reasons they were ranked that way.

That is a subtler failure than a wrong number, because the plan still looks
right. It just no longer describes the decision that was actually made.
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

from src.mission import registry as reg
from src.mission.compiler import ParsedUtterance, compile_scenario
from src.mission.registry import compile_registry
from src.mission.spec import ScenarioAmendment

PRICEABLE = ("SPY", "VOO", "IVV", "QQQ", "VTI", "BND", "AGG")
TEXT = "if i buy 1000 usd of SP500 etf every time it crosses below its 200 DMA"
OBSERVED = "SP500 etf (specific ticker not given)"


def compiled(amendments=()):
    parsed = ParsedUtterance(text=TEXT, unclear=(OBSERVED,))
    return compile_scenario(TEXT, parsed=parsed, priceable=PRICEABLE,
                            amendments=amendments).scenario


def answered(symbol="SPY"):
    return (ScenarioAmendment(question_id=f"asset_identity:{OBSERVED}",
                              answer=symbol, recorded_at="t"),)


class TestThePlanRecordsHowItResolved:
    def test_the_registry_digest_is_stored(self):
        record = compiled().provenance.asset_resolutions[0]
        assert record.registry_digest.startswith("reg1:")

    def test_the_observed_phrase_is_stored(self):
        """What the user wrote, beside what it became."""
        assert compiled().provenance.asset_resolutions[0].observed_phrase == OBSERVED

    def test_the_concept_is_stored(self):
        assert compiled().provenance.asset_resolutions[0].resolved_concept_id \
            == "INDEX:SP500"

    def test_the_alternatives_are_stored(self):
        """A plan that keeps only the outcome cannot say what the choice was
        between."""
        shown = compiled().provenance.asset_resolutions[0].candidates_shown
        assert shown[0] == "SPY"
        assert set(shown) == {"SPY", "VOO", "IVV"}

    def test_the_ranking_reasons_are_stored(self):
        reasons = compiled().provenance.asset_resolutions[0].ranking_reasons
        assert any("default" in one for one in reasons)
        assert any("tracks" in one for one in reasons)

    def test_the_choice_is_stored_once_made(self):
        record = compiled(answered("VOO")).provenance.asset_resolutions[0]
        assert record.chosen_instrument_id == "VOO"

    def test_it_is_recorded_before_the_user_answers(self):
        """The alternatives shown are part of what happened whether or not a
        choice has been made yet."""
        record = compiled().provenance.asset_resolutions[0]
        assert record.chosen_instrument_id == ""
        assert record.candidates_shown

    def test_the_record_serialises(self):
        payload = compiled(answered()).provenance.asset_resolutions[0].to_json()
        assert payload["registry_digest"].startswith("reg1:")
        assert payload["candidates_shown"][0] == "SPY"


class TestAChangedRegistryDoesNotRewriteHistory:
    """The drift case: registry A stored, registry B current."""

    @pytest.fixture
    def registry_b(self, tmp_path):
        """A registry where the S&P default is VOO rather than SPY."""
        source = reg.SOURCE_DIR
        for name in ("concepts.yaml", "instruments.yaml", "aliases.yaml"):
            (tmp_path / name).write_text(
                (source / name).read_text(encoding="utf-8"), encoding="utf-8")
        loaded = yaml.safe_load((tmp_path / "concepts.yaml").read_text())
        for concept in loaded["concepts"]:
            if concept["concept_id"] == "INDEX:SP500":
                concept["default_instrument"] = "US:NYSEARCA:VOO"
        (tmp_path / "concepts.yaml").write_text(yaml.safe_dump(loaded))
        yield compile_registry(tmp_path)

    def test_the_two_registries_really_differ(self, registry_b):
        """The premise. If B resolved identically, every assertion below would
        hold for a system that ignored the registry entirely."""
        from src.mission.resolver import resolve

        under_b = resolve("SP500 etf", priceable=PRICEABLE, registry=registry_b)
        assert under_b.candidates[0].symbol == "VOO"
        assert registry_b.digest != compile_registry().digest

    def test_the_stored_record_keeps_registry_a(self, registry_b):
        """Written under A and read after B exists: the offered order, the
        reasons and the digest are all still A's."""
        stored = compiled(answered("SPY")).provenance.asset_resolutions[0]

        assert stored.candidates_shown[0] == "SPY"
        assert stored.registry_digest != registry_b.digest
        assert any("default" in one and one.startswith("SPY")
                   for one in stored.ranking_reasons)

    def test_reading_a_stored_record_consults_no_registry(self, registry_b):
        """Reopening must read the stored facts.

        Enforced by making resolution impossible: if reading the record
        touched the registry, this would raise rather than return.
        """
        stored = compiled(answered("SPY")).provenance.asset_resolutions[0]

        import src.mission.resolver as resolver_module

        def refuse(*args, **kwargs):
            raise AssertionError("reopening re-resolved against the registry")

        original = resolver_module.resolve
        resolver_module.resolve = refuse
        try:
            payload = stored.to_json()
        finally:
            resolver_module.resolve = original

        assert payload["chosen_instrument_id"] == "SPY"
        assert payload["candidates_shown"][0] == "SPY"
        assert payload["registry_digest"] == stored.registry_digest
