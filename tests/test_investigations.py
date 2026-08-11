"""Investigations: the questions, including the ones that produced nothing.

The type exists for a narrow reason. Every other artifact records something that
happened; only an investigation can record something that was tried and did not
work out. Without it the library reports a filtered history — and the filter runs
in the direction that flatters the platform.
"""
from __future__ import annotations

import pytest

from src.knowledge import (
    Investigation,
    InvestigationOutcome as Outcome,
    InvestigationRegistry,
    KnowledgeGraph,
)
from src.knowledge.registry import (
    AssumptionRegistry,
    ClaimRegistry,
    EvidenceRegistry,
    FindingRegistry,
)


def make(**kw):
    defaults = dict(name="q", version=1, question="Does it?",
                    outcome=Outcome.PENDING)
    defaults.update(kw)
    return Investigation(**defaults)


@pytest.fixture
def graph():
    return KnowledgeGraph(
        methodologies=[], claims=ClaimRegistry().load_all(),
        assumptions=AssumptionRegistry().load_all(),
        evidence=EvidenceRegistry().load_all(),
        findings=FindingRegistry().load_all(),
        investigations=InvestigationRegistry().load_all(),
    )


class TestTheOutcomeMustMatchTheEvidence:
    def test_claiming_a_finding_requires_citing_one(self):
        with pytest.raises(ValueError, match="cites none"):
            make(outcome=Outcome.FINDING_RECORDED, closed_at="2026-01-01")

    def test_a_null_result_may_not_cite_a_finding(self):
        """If it concluded something, its outcome is mislabelled."""
        with pytest.raises(ValueError, match="mislabelled"):
            make(outcome=Outcome.NO_EFFECT_FOUND, closed_at="2026-01-01",
                 examined=("methodology/hrp@1",), findings=("finding/x@1",))

    def test_a_null_result_must_say_what_it_examined(self):
        """'We looked and found nothing' names nothing that was looked at."""
        with pytest.raises(ValueError, match="must say what it examined"):
            make(outcome=Outcome.INCONCLUSIVE, closed_at="2026-01-01")

    def test_a_closed_outcome_needs_a_closing_date(self):
        with pytest.raises(ValueError, match="closed state"):
            make(outcome=Outcome.ABANDONED)

    def test_an_open_inquiry_needs_none_of_this(self):
        i = make(outcome=Outcome.PENDING)
        assert i.is_open and not i.produced_nothing


class TestNullResultsAreFirstClass:
    def test_a_null_result_is_closed_and_produced_nothing(self):
        i = make(outcome=Outcome.NO_EFFECT_FOUND, closed_at="2026-01-01",
                 examined=("methodology/hrp@1",), trials_examined=6)

        assert i.outcome.is_closed
        assert i.outcome.is_null_result
        assert i.produced_nothing

    def test_inconclusive_is_not_the_same_as_no_effect(self):
        """'We could not tell' is not 'there is nothing there'."""
        assert Outcome.INCONCLUSIVE is not Outcome.NO_EFFECT_FOUND
        assert Outcome.INCONCLUSIVE.is_null_result
        assert Outcome.NO_EFFECT_FOUND.is_null_result

    def test_abandonment_is_distinguishable_from_a_null_result(self):
        """Stopping work is not a result, and must not read as one."""
        assert not Outcome.ABANDONED.is_null_result
        assert Outcome.ABANDONED.is_closed

    def test_the_library_records_at_least_one_of_each(self, graph):
        outcomes = {i.outcome for i in graph.investigations}
        assert Outcome.NO_EFFECT_FOUND in outcomes
        assert Outcome.INCONCLUSIVE in outcomes
        assert Outcome.PENDING in outcomes, (
            "a research record with no open questions is not a research record"
        )


class TestTrialsSurviveTheirInvestigation:
    """Deflation counts configurations tried, not configurations that worked."""

    def test_a_null_result_still_carries_its_trials(self, graph):
        nulls = graph.null_results()
        assert sum(i.trials_examined for i in nulls) > 0, (
            "an inquiry that spent trials and concluded nothing must still say so"
        )

    def test_trials_are_attributed_to_the_methodologies_examined(self, graph):
        counts = graph.recorded_trials()
        assert counts.get("methodology/xsmom@1", 0) >= 6, (
            "the lookback sweep spent six trials on xsmom; they belong to it"
        )

    def test_trial_attribution_ignores_non_methodology_references(self, graph):
        assert all(k.startswith("methodology/") for k in graph.recorded_trials())


class TestProvenanceRunsBothWays:
    def test_every_finding_names_the_inquiry_behind_it(self, graph):
        assert graph.unattributed_findings() == [], (
            "a conclusion with no recorded inquiry asserts work without naming it"
        )

    def test_a_finding_resolves_back_to_its_investigation(self, graph):
        finding = graph.findings[0]
        i = graph.investigation_for_finding(finding.artifact_id)

        assert i is not None
        assert finding.artifact_id in i.findings

    def test_provenance_resolves_findings_and_examined_artifacts(self, graph):
        i = next(i for i in graph.investigations if i.findings)
        p = graph.investigation_provenance(i)

        assert len(p["findings"]) == len(i.findings)
        assert p["examined_methodologies"], "an inquiry examined something"

    def test_an_open_inquiry_has_no_findings_to_resolve(self, graph):
        for i in graph.open_inquiries():
            assert graph.investigation_provenance(i)["findings"] == []


class TestIdentityAndHashing:
    def test_the_hash_covers_the_outcome(self):
        base = dict(name="q", version=1, question="Does it?",
                    examined=("methodology/hrp@1",), closed_at="2026-01-01")
        a = Investigation(outcome=Outcome.NO_EFFECT_FOUND, **base)
        b = Investigation(outcome=Outcome.INCONCLUSIVE, **base)

        assert a.content_hash != b.content_hash

    def test_the_hash_covers_trials_spent(self):
        base = dict(name="q", version=1, question="Does it?",
                    outcome=Outcome.NO_EFFECT_FOUND,
                    examined=("methodology/hrp@1",), closed_at="2026-01-01")
        assert (Investigation(trials_examined=6, **base).content_hash
                != Investigation(trials_examined=0, **base).content_hash), (
            "trials spent change what the record means and must change its hash"
        )

    def test_prose_does_not_change_identity(self):
        """Rewording a resolution is a metadata edit, not a new inquiry."""
        base = dict(name="q", version=1, question="Does it?",
                    outcome=Outcome.NO_EFFECT_FOUND,
                    examined=("methodology/hrp@1",), closed_at="2026-01-01")
        assert (Investigation(resolution="a", **base).content_hash
                == Investigation(resolution="b", **base).content_hash)


class TestRegistry:
    def test_every_artifact_on_disk_loads(self):
        assert len(InvestigationRegistry().load_all()) >= 5

    def test_filename_and_content_must_agree(self, tmp_path):
        (tmp_path / "wrong-name@1.yaml").write_text(
            "name: other-name\nversion: 1\nquestion: Does it?\noutcome: PENDING\n"
        )
        with pytest.raises(ValueError, match="disagree about identity"):
            InvestigationRegistry(tmp_path).load_all()

    def test_an_invalid_outcome_fails_at_load_not_at_render(self, tmp_path):
        (tmp_path / "bad@1.yaml").write_text(
            "name: bad\nversion: 1\nquestion: Does it?\n"
            "outcome: FINDING_RECORDED\nclosed_at: '2026-01-01'\n"
        )
        with pytest.raises(ValueError, match="cites none"):
            InvestigationRegistry(tmp_path).load_all()
