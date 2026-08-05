"""The registry separates what a flat alias map had to conflate.

`phrase -> ticker` was answering seven questions at once: what was written,
what kind of thing was meant, which securities satisfy it, which are available
here, which comes first, and what the user chose. Each has a different
lifetime and a different owner.

    concepts.yaml     what a user can mean that is not purchasable
    instruments.yaml  canonical identity, one record per tradeable thing
    aliases.yaml      an observation, with the facets a map cannot hold
"""
from __future__ import annotations

import pathlib

import pytest
import yaml

from src.mission import registry as reg
from src.mission.registry import RegistryError, compile_registry, normalize
from src.mission.resolver import resolve

PRICEABLE = ("SPY", "VOO", "IVV", "QQQ", "VTI", "BND", "AGG", "BIL", "IEF",
             "VXUS", "GLD", "IWM", "DIA")


@pytest.fixture(scope="module")
def compiled():
    return compile_registry()


class TestTheRegistryCompilesAndHoldsTogether:
    def test_it_compiles(self, compiled):
        assert compiled.instruments and compiled.concepts

    def test_identity_is_not_the_ticker(self, compiled):
        """Tickers collide across venues and are reissued after a delisting."""
        one = compiled.instrument_by_symbol("SPY")
        assert one.instrument_id == "US:NYSEARCA:SPY"
        assert one.symbol == "SPY"
        assert one.provider_symbols["yahoo"] == "SPY"

    def test_the_digest_is_stable_across_compilations(self, compiled):
        assert compile_registry().digest == compiled.digest

    def test_the_digest_changes_when_the_content_does(self, compiled, tmp_path):
        """A digest that never moves cannot pin anything."""
        for name in ("concepts.yaml", "instruments.yaml", "aliases.yaml"):
            (tmp_path / name).write_text(
                (reg.SOURCE_DIR / name).read_text(encoding="utf-8"),
                encoding="utf-8")
        loaded = yaml.safe_load((tmp_path / "concepts.yaml").read_text())
        loaded["concepts"].append({
            "concept_id": "INDEX:TEST", "kind": "INDEX",
            "canonical_name": "Test", "aliases": ["a test index"]})
        (tmp_path / "concepts.yaml").write_text(yaml.safe_dump(loaded))
        try:
            assert compile_registry(tmp_path).digest != compiled.digest
        finally:
            reg.SOURCE_DIR = (pathlib.Path(reg.__file__).resolve().parents[2]
                              / "data" / "instruments")


class TestTheCompilerRefusesWhatWouldMisresolve:
    def written(self, tmp_path, concepts=None, instruments=None, aliases=None):
        source = reg.SOURCE_DIR
        base = {
            "concepts.yaml": concepts if concepts is not None else {
                "version": 1, "concepts": [{
                    "concept_id": "INDEX:X", "kind": "INDEX",
                    "canonical_name": "X", "aliases": ["x index"]}]},
            "instruments.yaml": instruments if instruments is not None else {
                "version": 1, "instruments": [{
                    "instrument_id": "US:V:AAA", "symbol": "AAA", "name": "A",
                    "instrument_type": "ETF", "exchange": "V",
                    "currency": "USD", "issuer": "i",
                    "provider_symbols": {"yahoo": "AAA"},
                    "tracks_index": "INDEX:X"}]},
            "aliases.yaml": aliases if aliases is not None else {
                "version": 1, "aliases": []},
        }
        for name, payload in base.items():
            (tmp_path / name).write_text(yaml.safe_dump(payload))
        try:
            return compile_registry(tmp_path)
        finally:
            reg.SOURCE_DIR = source

    def test_a_dangling_relationship(self, tmp_path):
        """An ETF pointing at a concept nobody defined is a candidate list
        that silently comes back empty."""
        with pytest.raises(RegistryError, match="unknown concept"):
            self.written(tmp_path, instruments={
                "version": 1, "instruments": [{
                    "instrument_id": "US:V:AAA", "symbol": "AAA", "name": "A",
                    "instrument_type": "ETF", "exchange": "V",
                    "currency": "USD", "issuer": "i",
                    "provider_symbols": {},
                    "tracks_index": "INDEX:NOBODY"}]})

    def test_a_duplicate_identity(self, tmp_path):
        with pytest.raises(RegistryError, match="duplicate instrument"):
            one = {"instrument_id": "US:V:AAA", "symbol": "AAA", "name": "A",
                   "instrument_type": "ETF", "exchange": "V",
                   "currency": "USD", "issuer": "i", "provider_symbols": {}}
            self.written(tmp_path,
                         instruments={"version": 1, "instruments": [one, one]})

    def test_two_instruments_claiming_one_provider_symbol(self, tmp_path):
        """A price lookup would be ambiguous, and would pick by dict order."""
        with pytest.raises(RegistryError, match="claimed by"):
            self.written(tmp_path, instruments={"version": 1, "instruments": [
                {"instrument_id": "US:V:AAA", "symbol": "AAA", "name": "A",
                 "instrument_type": "ETF", "exchange": "V", "currency": "USD",
                 "issuer": "i", "provider_symbols": {"yahoo": "Z"}},
                {"instrument_id": "US:V:BBB", "symbol": "BBB", "name": "B",
                 "instrument_type": "ETF", "exchange": "V", "currency": "USD",
                 "issuer": "i", "provider_symbols": {"yahoo": "Z"}}]})

    def test_a_phrase_claimed_by_two_targets(self, tmp_path):
        with pytest.raises(RegistryError, match="claimed by"):
            self.written(tmp_path, concepts={"version": 1, "concepts": [
                {"concept_id": "INDEX:X", "kind": "INDEX",
                 "canonical_name": "X", "aliases": ["shared"]},
                {"concept_id": "INDEX:Y", "kind": "INDEX",
                 "canonical_name": "Y", "aliases": ["shared"]}]})

    def test_a_default_pointing_nowhere(self, tmp_path):
        with pytest.raises(RegistryError, match="unknown instrument"):
            self.written(tmp_path, concepts={"version": 1, "concepts": [{
                "concept_id": "INDEX:X", "kind": "INDEX",
                "canonical_name": "X", "aliases": ["x index"],
                "default_instrument": "US:V:NOBODY"}]})

    def test_the_same_spelling_twice_for_one_concept_is_fine(self, tmp_path):
        """"spx" and "^spx" normalise together and name the same thing."""
        built = self.written(tmp_path, concepts={"version": 1, "concepts": [{
            "concept_id": "INDEX:X", "kind": "INDEX", "canonical_name": "X",
            "aliases": ["spx", "^spx"]}]})
        assert built.phrase_index["spx"] == ("CONCEPT", "INDEX:X")


class TestResolutionIsTypedRatherThanSubstituted:
    def test_an_index_plus_a_vehicle_request(self):
        found = resolve("SP500 etf", priceable=PRICEABLE)
        assert found.concept_id == "INDEX:SP500"
        assert found.vehicle_requested.value == "ETF"
        assert "is an index" in found.mismatch

    def test_the_issuer_in_the_phrase_decides_the_order(self):
        found = resolve("vanguard s&p fund", priceable=PRICEABLE)
        assert found.candidates[0].symbol == "VOO"
        assert any("Vanguard" in reason
                   for reason in found.candidates[0].reasons)

    def test_order_is_declared_rather_than_alphabetical(self):
        """All three S&P funds tie on every other term. Without a declared
        default the sort fell to the alphabet and offered IVV first, which is
        an accident presented as a recommendation."""
        found = resolve("SP500 etf", priceable=PRICEABLE)
        assert found.candidates[0].symbol == "SPY"
        assert any("default" in reason
                   for reason in found.candidates[0].reasons)

    def test_every_candidate_explains_itself(self):
        for candidate in resolve("SP500 etf", priceable=PRICEABLE).candidates:
            assert candidate.reasons, f"{candidate.symbol} ranked with no reason"

    def test_an_equal_weight_fund_is_not_offered_as_the_index(self):
        """RSP holds the same constituents with a different strategy."""
        symbols = [one.symbol
                   for one in resolve("SP500 etf", priceable=PRICEABLE + ("RSP",)).candidates]
        assert "RSP" not in symbols

    def test_candidates_are_filtered_to_what_can_be_priced(self):
        found = resolve("SP500 etf", priceable=("VOO",))
        assert [one.symbol for one in found.candidates] == ["VOO"]

    def test_an_unknown_phrase_resolves_to_nothing(self):
        assert resolve("technology ETF", priceable=PRICEABLE).unresolved

    def test_a_directly_named_instrument_is_not_a_question(self):
        found = resolve("the spy", priceable=PRICEABLE)
        assert found.certain
        assert found.candidates[0].symbol == "SPY"

    def test_the_resolution_carries_the_registry_digest(self):
        """A stored plan must be able to say which catalogue read it."""
        assert resolve("SP500 etf", priceable=PRICEABLE).registry_digest.startswith("reg1:")


class TestNormalisation:
    @pytest.mark.parametrize("written,expected", [
        ("SP500 etf (no literal ticker given)", "sp500 etf"),
        ("  The   S&P  ", "the s&p"),
        ("^SPX", "spx"),
    ])
    def test_it_strips_what_is_not_the_phrase(self, written, expected):
        assert normalize(written) == expected


class TestProposedAliasesNeverResolve:
    """One user's clarification must not change everyone's reading."""

    def test_a_proposed_alias_is_not_indexed(self, tmp_path):
        source = reg.SOURCE_DIR
        (tmp_path / "concepts.yaml").write_text(yaml.safe_dump(
            {"version": 1, "concepts": [{
                "concept_id": "INDEX:X", "kind": "INDEX",
                "canonical_name": "X", "aliases": ["x index"]}]}))
        (tmp_path / "instruments.yaml").write_text(yaml.safe_dump(
            {"version": 1, "instruments": []}))
        (tmp_path / "aliases.yaml").write_text(yaml.safe_dump(
            {"version": 1, "aliases": [{
                "phrase": "s and p tracker", "target_kind": "CONCEPT",
                "target_id": "INDEX:X", "source": "PROPOSED"}]}))
        try:
            built = compile_registry(tmp_path)
            assert "s and p tracker" not in built.phrase_index
        finally:
            reg.SOURCE_DIR = source
