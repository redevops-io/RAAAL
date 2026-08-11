"""The harvested corpus, and the properties that keep its answer key honest.

The corpus is attested language with attribution. The annotations are mine,
written by reading the sentences. Most of this file is about the second half,
because an answer key derived from the system under test is the one failure
that cannot be seen from the results — everything looks consistent, and the
consistency is the artefact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

HARVEST = Path(__file__).resolve().parent.parent / "corpus" / "harvested"
ANNOTATIONS = HARVEST / "annotations.json"
SURVIVAL = HARVEST / "survival.json"


@pytest.fixture(scope="module")
def annotations():
    if not ANNOTATIONS.exists():
        pytest.skip("annotations.json is absent; run corpus/harvested/annotate.py")
    return json.loads(ANNOTATIONS.read_text())


@pytest.fixture(scope="module")
def survival():
    if not SURVIVAL.exists():
        pytest.skip("survival.json is absent; run corpus/harvested/survival.py")
    return json.loads(SURVIVAL.read_text())


class TestTheCorpusIsAttested:
    def test_every_annotated_sentence_carries_its_source(self, annotations):
        """CC-BY-SA requires attribution, and provenance is what separates this
        pack from sentences somebody made up that sound plausible."""
        for entry in annotations["annotations"]:
            assert entry["source"].startswith("http"), entry["text"][:40]
            assert entry["licence"]

    def test_the_sentences_are_not_variations_of_each_other(self, annotations):
        """Templating is the failure this corpus exists to avoid. Thirty
        rewrites of one sentence measure one sentence thirty times and report
        it as coverage."""
        texts = [e["text"] for e in annotations["annotations"]]
        assert len(set(texts)) == len(texts)
        openings = [" ".join(t.split()[:3]).lower() for t in texts]
        # No opening may account for more than a fifth of the pack.
        worst = max(openings.count(o) for o in set(openings))
        assert worst <= len(texts) // 5, (
            f"{worst} of {len(texts)} sentences open the same way")

    def test_the_yield_is_recorded_rather_than_the_harvest_size(self,
                                                                annotations):
        """220 harvested and 29 usable are different numbers and the first one
        flatters. Both are in the artifact."""
        assert annotations["harvested"] > annotations["strategy_statements"]
        assert annotations["not_a_strategy_statement"] > 0
        assert "situation" in annotations["yield_note"]


class TestTheAnswerKeyIsNotTheSystemsOwnOutput:
    def test_the_concept_vocabulary_is_not_the_schema_vocabulary(self,
                                                                 annotations):
        """The check that catches the tempting shortcut. If the material
        concepts were named `amount`, `cadence`, `assets`, the annotation would
        almost certainly have been read off Discovery rather than off the
        sentence, and the comparison would be the runtime against itself."""
        from src.discovery.schema import QUANTIFY_SCHEMA

        dimensions = {d.name for d in QUANTIFY_SCHEMA.dimensions}
        concepts = set(annotations["maps_to"])
        overlap = concepts & dimensions
        assert not overlap, (
            f"{sorted(overlap)} are named identically to schema dimensions; "
            "the answer key is using the vocabulary of the thing it grades")

    def test_every_material_concept_is_mapped(self, annotations):
        """Including to `None`. A concept with no entry is one nobody decided
        about, and it would silently score as unrepresentable."""
        mapped = annotations["maps_to"]
        for entry in annotations["annotations"]:
            for concept in entry["material"]:
                assert concept in mapped, (
                    f"{concept!r} is asserted by a sentence and has no entry "
                    "in MAPS_TO, so nothing has decided whether this build "
                    "could carry it")

    def test_a_concept_mapped_to_a_dimension_names_a_real_one(self,
                                                              annotations):
        """The mapping was wrong once, in the direction that manufactures
        findings: `which account it sits in` pointed at `asset_location` when
        the reader settles `account_type`, and ten sentences were reported as
        dropping a concept that had in fact survived."""
        from src.discovery.schema import QUANTIFY_SCHEMA

        dimensions = {d.name for d in QUANTIFY_SCHEMA.dimensions}
        relations = {r.kind for r in getattr(QUANTIFY_SCHEMA, "relations", ())}
        for concept, dimension in annotations["maps_to"].items():
            if dimension is None:
                continue
            assert dimension in dimensions | relations, (
                f"{concept!r} maps to {dimension!r}, which the schema does not "
                "have; a mapping to a name that does not exist scores every "
                "sentence asserting it as dropped")


class TestTheMetricCannotFlatter:
    def test_a_high_rate_with_nothing_running_is_labelled_as_such(self,
                                                                  survival):
        """The number this run produced is 100%, and it is not good news: no
        attested sentence reached a plan, so nothing could be reduced. An
        artifact that reported the percentage without saying so would be
        quotable as the opposite of what it means."""
        if survival["reached_a_plan"] == 0:
            assert "not a good result" in survival["caution"]
            assert "nothing ran" in survival["caution"]

    def test_recognition_is_not_counted_as_survival(self, survival):
        """The rotation defect in metric form: Discovery reading a concept
        correctly and the plan dropping it later must not score as honoured.
        `HONOURED` requires a plan to exist."""
        from corpus.harvested.survival import HONOURED, fate

        settled, refusals, questions = {"cadence"}, set(), set()
        assert fate("how often money goes in", "cadence", settled, refusals,
                    questions, executed=False) != HONOURED
        assert fate("how often money goes in", "cadence", settled, refusals,
                    questions, executed=True) == HONOURED

    def test_an_unrepresentable_concept_is_never_honoured(self, survival):
        """A concept this build has no dimension for cannot be carried by a
        plan, however well the plan compiled."""
        from corpus.harvested.survival import DROPPED, fate

        assert fate("a volatility level to aim at", None, set(), set(), set(),
                    executed=True) == DROPPED

    def test_unreadable_sentences_stay_in_the_denominator(self, survival):
        """A sentence that failed to read is recorded, not skipped. Dropping
        them raises the rate by exactly the cases that went worst."""
        assert "unrecorded" in survival
        assert isinstance(survival["unrecorded"], list)


class TestWhatTheHarvestFound:
    def test_the_dominant_stopper_is_the_unnamed_holding(self, survival):
        """The finding worth acting on. Real strategy statements routinely do
        not say what to buy — "I contribute $750/month to my 401k" is a
        complete thought to the person writing it. The authored corpus always
        names an asset, because whoever wrote it knew the runtime needed one.
        """
        stopped = survival["stopped_by"]
        assert stopped, "nothing was stopped; this run measured nothing"
        top = max(stopped, key=lambda k: stopped[k])
        assert top == "assets", (
            f"the dominant stopper is now {top!r} rather than the unnamed "
            "holding; the harvest's headline finding has changed and the "
            "write-up in docs/Harvested-Corpus.md no longer describes it")

    def test_no_attested_sentence_is_silently_reduced(self, survival):
        """The property that matters more than the rate. `DROPPED` is the
        narrow dangerous case: a plan ran and a concept the person asserted is
        neither in it nor mentioned."""
        assert survival["tally"]["DROPPED"] == 0, (
            f"{survival['dropped_by_concept']} were silently dropped from "
            "plans that executed")


class TestTheRateCannotBecomeAProductClaim:
    """Rule 3 in `docs/Evidence-Rules.md`. The caution lives in the artifact
    because the artifact is what gets quoted, and this is the same discipline
    applied to the prose that surrounds it."""

    DOCS = Path(__file__).resolve().parent.parent / "docs"

    def test_any_document_quoting_the_rate_also_carries_the_caution(self):
        """A percentage in a document with no denominator beside it is a claim,
        whatever the surrounding sentence intended. 18/18 reads as a triumph
        and means that nothing ran."""
        import re

        pattern = re.compile(r"survival[^.\n]{0,40}?(\d{1,3})\s*/\s*(\d{1,3})"
                             r"|survival[^.\n]{0,40}?(\d{2,3}(?:\.\d+)?)\s*%",
                             re.I)
        for path in sorted(self.DOCS.glob("*.md")):
            text = path.read_text()
            if not pattern.search(text):
                continue
            lowered = text.lower()
            assert ("should not be quoted" in lowered
                    or "must not be quoted" in lowered
                    or "not a product claim" in lowered), (
                f"{path.name} quotes a material-semantic survival figure with "
                "no caution beside it. The number is high because no attested "
                "sentence reached a plan; unqualified it says the opposite of "
                "what it means")

    def test_the_artifact_states_its_own_denominator(self, survival):
        """`adjudicated` is what the rate is over, and it is a fraction of the
        semantics asserted. A rate published without it cannot be read."""
        assert survival["adjudicated"] < survival["material_semantics"]
        assert survival["adjudicated"] == (
            survival["tally"]["HONOURED"] + survival["tally"]["NAMED"]
            + survival["tally"]["DROPPED"])
