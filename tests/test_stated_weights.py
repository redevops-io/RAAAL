"""A stated split, attached to the holdings it belongs to.

`stated_weights` was refused for the life of the project, and the reason on the
manifest was wrong for most of it: it said this build divides each purchase
equally and cannot honour 60/40. The engine has always been able to — `weighted`
takes the split and `_execute` fills to it. What was missing was the *relation*:
`60/40` beside two instruments could mean either assignment, and the corpus
sentences that state a bare ratio name no holdings whatever.

So the split is honoured when the sentence attaches it — "60% in VTI and 40% in
BND" — and refused by name when it does not. The refusal is the interesting half:
guessing would run 40/60 under the name 60/40, which is a wrong executable
meaning, and no check downstream could catch it because both readings produce a
perfectly ordinary figure.
"""
from __future__ import annotations

import pytest

from src.discovery.derived_readers import WEIGHTS_READER_ID, weight_binding


class TestTheBindingIsReadFromTheSentence:
    @pytest.mark.parametrize("text,expected", [
        ("invest 60% in VTI and 40% in BND", "VTI=60,BND=40"),
        ("put 70% into VTI, 20% into BND and 10% into GLD",
         "VTI=70,BND=20,GLD=10"),
        ("50% VTI 50% BND", "VTI=50,BND=50"),
    ])
    def test_an_attached_split_is_bound(self, text, expected):
        found = weight_binding([], None, text)
        assert found is not None and found.value == expected
        assert found.reader_id == WEIGHTS_READER_ID

    @pytest.mark.parametrize("text", [
        "a 60/40 portfolio",
        "rebalance to 70/30",
        "invest $500 a month into VTI and BND, 60/40",
        "60% VTI",
    ])
    def test_a_bare_ratio_is_not_bound(self, text):
        """The refusal these produce is the point. A ratio with nothing saying
        which side is which could mean either assignment, and the sentences in
        the corpus that state one name no holdings at all."""
        assert weight_binding([], None, text) is None

    def test_the_same_holding_twice_is_declined(self):
        """"50% in VTI and 50% in VTI" cannot be executed without deciding
        which mention wins, which is a decision nobody stated."""
        assert weight_binding([], None, "50% in VTI and 50% in VTI") is None

    def test_it_needs_no_parse(self):
        """Read from the sentence, not the parse. The deployment that serves
        users has no deterministic parser installed, so a reader that needed
        one would be correct in this suite and absent in production — the shape
        of every gap this project has found in its own deployment."""
        assert weight_binding([], None, "60% in VTI and 40% in BND") is not None


class TestTheCompilerHonoursIt:
    def _reading(self, text):
        from src.discovery.hosted_recording import RecordedHostedReader
        from src.discovery.schema import QUANTIFY_SCHEMA
        from src.discovery.syntax_stanza import RecordedReader
        from src.discovery.witnesses import BOTH
        from src.workspace.pilot import read

        return read(text, RecordedHostedReader(), schema=QUANTIFY_SCHEMA,
                    profile=BOTH, syntax_reader=RecordedReader())

    def test_an_attached_split_reaches_the_allocation_rule(self):
        reading = self._reading(
            "invest $500 a month, 60% in VTI and 40% in BND")
        assert reading.compiled is not None, "the plan did not compile"
        weights = dict(reading.compiled.scenario.allocation_rule.weights)
        assert weights, "the split was read and did not reach the scenario"
        assert weights["VTI"] > weights["BND"], (
            f"{weights} — the larger share went to the wrong holding, which is "
            "the failure this binding exists to prevent")

    def test_a_bare_ratio_is_refused_by_name(self):
        """Not dropped, and not guessed. The dimension is named so the request
        can be counted and the reason tells the person what would work."""
        reading = self._reading("invest $500 a month into VTI and BND, 60/40")
        refused = {getattr(r, "dimension", ""): getattr(r, "detail", "")
                   for r in reading.refusals}
        assert "stated_weights" in refused, (
            f"a bare ratio compiled without refusal: {refused}")
        assert "60% in VTI" in refused["stated_weights"], (
            "the refusal does not show what phrasing would run")


class TestTheManifestMatchesTheEngine:
    def test_stated_weights_is_no_longer_refused_wholesale(self):
        """The manifest said this build divides each purchase equally and
        cannot honour a split. It could; nothing passed it one."""
        from src.mission.capability import EXECUTED, MANIFEST

        assert MANIFEST["stated_weights"].support == EXECUTED
        assert not MANIFEST["stated_weights"].closed

    def test_the_reason_names_the_phrasing_that_works(self):
        from src.mission.capability import MANIFEST

        assert "60% in VTI" in MANIFEST["stated_weights"].why
