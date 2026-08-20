"""Discovery's side of the boundary, checked offline.

No test here calls a provider. The hosted reader's *prompt* is inspected
directly, which is the part that has to be right — whether the model answers
well is a measurement (`corpus/shadow_run.py`), not an assertion.
"""
from __future__ import annotations

import pytest

from src.discovery import QUANTIFY_SCHEMA, compare
from src.discovery.reader import Reading, ReadingSet
from src.discovery.readers_quantify import CompilerReader, HostedReader
from src.discovery.shadow import AGREED, CONTESTED, ONE_SIDED, UNREAD, _same


class TestTheSchemaIsWiderThanTheManifest:
    """Discovery may understand more than Mission can execute. A reader taught
    only what the engine runs does not refuse the rest — it says the nearest
    runnable thing, which is how "by inverse volatility" became an equal
    split."""

    def test_it_offers_allocation_methods_the_engine_refuses(self):
        from src.mission.capability import MANIFEST

        sayable = set(QUANTIFY_SCHEMA.dimension("allocation_method").values)
        runnable = set(MANIFEST["allocation_method"].values)
        # These converged deliberately: the computed strategies joined the
        # executable set, so every allocation method the reader can say, Mission
        # now runs. The schema is no wider than the manifest here — the width
        # that mattered (`inverse_volatility` said and substituted) closed when
        # the engine gained a way to run it. `inverse_volatility` is sayable and
        # runnable both, and nothing sayable is left that the engine refuses.
        assert runnable == sayable, sorted(sayable ^ runnable)
        assert "inverse_volatility" in sayable

    def test_it_offers_dimensions_the_engine_refuses_entirely(self):
        from src.mission.capability import MANIFEST, REFUSED

        refused = {n for n, d in MANIFEST.items() if d.support == REFUSED}
        assert refused & QUANTIFY_SCHEMA.names, (
            "no refused dimension is even sayable, so Mission can never name "
            "one in a refusal")

    def test_the_manifest_never_reaches_the_reader(self):
        """The egress policy's correctness clause, checked against the actual
        prompt rather than trusted."""
        from src.mission.schedule import REFUSED_CADENCES

        prompt = HostedReader()._schema_prompt(QUANTIFY_SCHEMA)
        assert "EXECUTED" not in prompt and "REFUSED" not in prompt
        assert "capability" not in prompt.lower()
        # and a value the engine refuses is still offered to the reader
        assert "inverse_volatility" in prompt
        # while nothing tells it which cadences are runnable
        assert "not execute" not in prompt.lower()
        assert all(why not in prompt for why in REFUSED_CADENCES.values())


class TestSilenceIsNotAgreement:
    def test_one_surviving_reader_agrees_with_nothing(self):
        """The bug this file exists for. With `len(values) < len(ok)` as the
        test, a shadow run against a dead endpoint reported every dimension it
        read as AGREED — perfect agreement with nobody."""
        alone = ReadingSet("solo", (Reading("cadence", "annual"),))
        result = compare("x", [alone], QUANTIFY_SCHEMA)
        assert result.by_state(AGREED) == ()
        assert [f.dimension for f in result.by_state(ONE_SIDED)] == ["cadence"]

    def test_a_comparison_with_one_contributor_is_not_usable(self):
        alone = ReadingSet("solo", (Reading("cadence", "annual"),))
        assert not compare("x", [alone], QUANTIFY_SCHEMA).usable

    def test_two_contributors_make_it_usable(self):
        """The discriminating half."""
        a = ReadingSet("a", (Reading("cadence", "annual"),))
        b = ReadingSet("b", (Reading("cadence", "annual"),))
        result = compare("x", [a, b], QUANTIFY_SCHEMA)
        assert result.usable
        assert [f.dimension for f in result.by_state(AGREED)] == ["cadence"]

    def test_a_dimension_neither_read_is_unread_not_agreed(self):
        a = ReadingSet("a", (Reading("cadence", "annual"),))
        b = ReadingSet("b", (Reading("cadence", "annual"),))
        result = compare("x", [a, b], QUANTIFY_SCHEMA)
        assert "sell_action" in {f.dimension for f in result.by_state(UNREAD)}


class TestATransportFailureIsNotAReading:
    def test_a_failed_reader_is_excluded_from_every_count(self):
        """An evaluator that scores its own outage as a product disagreement
        manufactures evidence."""
        ok = ReadingSet("a", (Reading("cadence", "annual"),))
        dead = ReadingSet("b", failed="Timeout")
        result = compare("x", [ok, dead], QUANTIFY_SCHEMA)
        assert result.failed_readers == {"b": "Timeout"}
        assert result.by_state(CONTESTED) == ()
        assert not result.usable

    def test_a_missing_key_fails_rather_than_returning_nothing(self):
        reader = HostedReader(api_key_env="DEFINITELY_NOT_SET")
        result = reader.read("anything", QUANTIFY_SCHEMA)
        assert not result.ok and "DEFINITELY_NOT_SET" in result.failed


class TestComparisonIsTypedNotClever:
    """The first live run reported 26 contested fields of which roughly three
    were about meaning; the rest were `"VTI, BND"` against `"VTI and BND"`."""

    def test_numbers_compare_as_numbers(self):
        assert _same("1000", "$1,000", "NUMBER")
        assert _same("200", "200-day", "NUMBER")
        assert not _same("200", "50", "NUMBER")

    def test_sets_ignore_the_conjunction(self):
        assert _same("VTI, BND", "VTI and BND", "SET")
        assert not _same("VTI, BND", "VTI and GLD", "SET")

    def test_text_does_not_get_a_synonym_table(self):
        """Resolving "annual" against "yearly" needs a table nobody can audit,
        and a dimension that needs one has the wrong vocabulary."""
        assert not _same("annual", "yearly", "TEXT")

    def test_a_value_that_tokenises_to_nothing_matches_nothing_real(self):
        """Two empty readings are the same reading and that is fine. What must
        not happen is punctuation matching an actual list."""
        assert not _same("and", "VTI, BND", "SET")
        assert not _same(",", "VTI", "SET")


class TestNoReaderIsPrivileged:
    def test_swapping_the_readers_does_not_change_the_verdict(self):
        a = ReadingSet("quantify-compiler@1", (Reading("cadence", "annual"),))
        b = ReadingSet("claude-sonnet-5@1", (Reading("cadence", "monthly"),))
        forward = compare("x", [a, b], QUANTIFY_SCHEMA)
        backward = compare("x", [b, a], QUANTIFY_SCHEMA)
        assert {f.dimension for f in forward.by_state(CONTESTED)} == \
            {f.dimension for f in backward.by_state(CONTESTED)}

    def test_the_module_holds_no_precedence_table(self):
        import src.discovery.shadow as shadow

        source = open(shadow.__file__).read()
        for smell in ("PRECEDENCE", "priority =", "wins", "prefer_model",
                      "prefer_rule"):
            assert smell not in source, f"{smell} looks like a winner rule"


class TestTheCompilerReaderIsAFairComparator:
    """Its first version omitted what the compiler knows but does not put in
    `recognitions`, which made the model look better for free."""

    def test_it_reports_assets_the_compiler_resolved(self):
        text = ("I buy $1,000 of SPY whenever it crosses below its 200-day "
                "moving average, over the past 5 years.")
        read = CompilerReader().read(text, QUANTIFY_SCHEMA)
        assert read.value_of("assets") is not None
        assert read.value_of("moving_average_window") is not None
        assert read.value_of("evaluation_period") is not None

    def test_the_window_is_the_users_words_not_an_enum(self):
        """`WindowKind.TRAILING` is this module's vocabulary, not evidence."""
        read = CompilerReader().read("I invest $100 monthly into VTI over the "
                                     "past 5 years.", QUANTIFY_SCHEMA)
        window = read.value_of("evaluation_period")
        assert window is not None and "WindowKind" not in str(window.value)

    def test_it_reports_what_it_did_not_read(self):
        read = CompilerReader().read("I invest $100 monthly into VTI.",
                                     QUANTIFY_SCHEMA)
        assert read.unread, "a reader that says nothing looks like agreement"
