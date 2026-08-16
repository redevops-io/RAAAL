"""The same plan, both paths, offline, over the canonical fixtures.

Step 5 of the migration, and the one thing that must be clean before any of the
deployment steps mean anything.

    Given the same plan and the same relevant market observations, old and new
    must produce the same evaluator semantics even though they intentionally
    produce different snapshot identities.

Offline over fixtures rather than shadowing live traffic. Shadowing production
means resolving market data twice per request, so the first thing a live
harness would prove is that it doubled the load — and it would prove it against
users. The fixtures come first; live shadowing is a decision to take once this
is clean.

**Differences are classified, not merely counted.** A tolerated difference whose
justification lives in a whitelist records that somebody once decided it was
fine. A verdict with its reason attached records why, travels with the
comparison, and can be re-read by whoever wonders whether the reason still
holds.
"""
from __future__ import annotations

import os

import pytest
from runtime_contracts import Author, IntentField, VerifiedIntent

from src.discovery.canonical import canonicalise
from src.evaluation.service import evaluate, evaluate_by_hash
from src.evaluation.shadow import (EXPECTED_DIFFERENCES, MUST_MATCH, TOLERATED,
                                   Verdict, compare)
from src.mission.evaluation_policy import declared_policy
from src.mission.from_intent import compile_intent
from src.mission.strategy_spec import from_scenario, to_scenario

POLICY = declared_policy(data_policy="SYNTHETIC_ONLY", as_of="2026-08-15")
ENGINE = "quantify-engine@shadow"

#: The canonical fixtures. One of each shape the engine has a different path
#: for, and each names instruments the symbol table can resolve — the new path
#: resolves names, so a plan naming something unresolvable is a market-data
#: refusal rather than a parity question.
FIXTURES = {
    "monthly-contributions": {"assets": "VTI", "amount": "1000",
                              "cadence": "monthly"},
    "annual-contributions": {"assets": "VTI", "amount": "1000",
                             "cadence": "annual"},
    "two-holdings": {"assets": "VTI,BND", "amount": "1000",
                     "cadence": "monthly"},
    "one-off": {"assets": "VTI", "amount": "10000", "cadence": "once"},
    "never-sells": {"assets": "VTI", "amount": "500", "cadence": "monthly",
                    "sell_action": "never sold any of it"},
}


@pytest.fixture(autouse=True)
def workspace(monkeypatch, tmp_path):
    from src.db import migrate
    from src.db.engine import Database
    from src.deploy import context as deploy_context

    url = f"sqlite:///{tmp_path}/w.db"
    for name, value in (("PILOT_DATA_POLICY", "SYNTHETIC_ONLY"),
                        ("QUANTIFY_PILOT_READER", "recorded"),
                        ("QUANTIFY_PARSER_MODE", "RUNTIME"),
                        ("QUANTIFY_PARSER_MODEL", "claude-sonnet-5"),
                        ("ANTHROPIC_API_KEY", "unused"),
                        ("QUANTIFY_DATABASE_URL", url)):
        monkeypatch.setenv(name, value)
    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)
    migrate.upgrade(Database(url))


@pytest.fixture
def market(tmp_path):
    from fastapi.testclient import TestClient

    from src.market_data.adapters import LocalParquetAdapter
    from src.market_data.client import HttpMarketData
    from src.market_data.object_store import ObjectStore
    from src.market_data.server import create_app

    store = ObjectStore(root=tmp_path / "objects")
    with TestClient(create_app(adapter=LocalParquetAdapter(),
                               store=store)) as http:
        def post(url, json):
            answer = (http.post(url, json=json) if json is not None
                      else http.get(url))
            try:
                return answer.status_code, answer.json()
            except ValueError:
                return answer.status_code, None

        def fetch(url, params):
            answer = http.get(url, params=params)
            return answer.status_code, answer.content, dict(answer.headers)

        yield HttpMarketData(post=post, fetch=fetch)


def spec_for(stated):
    canonical = canonicalise(stated)
    intent = VerifiedIntent(
        objective="evaluate_investment_strategy", produced_by="shadow",
        utterance_ref="u",
        fields={n: IntentField(value=v, author=Author.MODEL)
                for n, (v, _a) in canonical.fields.items()},
        unresolved=()).seal()
    out = compile_intent(intent, benchmark_rule="a-rule")
    assert out.scenario is not None, [r.detail for r in out.refusals]
    return from_scenario(out.scenario)


def both_paths(stated, client):
    """One plan, two data boundaries, one evaluator."""
    from src.workspace.run_boundary import execute_compiled_plan, market_data_for

    spec = spec_for(stated)

    old = evaluate(spec, "", evaluation_policy=POLICY, engine_version=ENGINE,
                   access=market_data_for(to_scenario(spec), context="shadow"),
                   run_plan=execute_compiled_plan)

    reinvested = spec.dividend_policy == "reinvested"
    snapshot_hash, descriptor_hash = client.create(list(spec.assets),
                                                   reinvested=reinvested)
    new = evaluate_by_hash(
        spec, client.get(snapshot_hash, descriptor_hash),
        evaluation_policy=POLICY, engine_version=ENGINE,
        run_plan=execute_compiled_plan)
    return old, new


@pytest.mark.parametrize("case", sorted(FIXTURES), ids=sorted(FIXTURES))
class TestEvaluationParityHolds:
    def test_the_two_paths_agree_on_evaluator_semantics(self, case, market):
        comparison = compare(*both_paths(FIXTURES[case], market))
        assert comparison.parity, (
            f"{case}: the paths disagree about what the plan does\n"
            + comparison.report())

    def test_the_streams_agree_stage_by_stage(self, case, market):
        """Named separately from the verdict, so a failure says which stage."""
        comparison = compare(*both_paths(FIXTURES[case], market))
        stages = [one.field for one in comparison.mismatches
                  if one.field.startswith("stream:")]
        assert stages == [], f"{case}: {stages}"

    def test_the_disposition_agrees(self, case, market):
        """A figure withheld by one path and shown by the other is the most
        consequential difference there is, and it appears in no stream."""
        old, new = both_paths(FIXTURES[case], market)
        assert tuple(old.refusals) == tuple(new.refusals)


class TestIdentityDiffersAndIsRecordedAsSuch:
    def test_the_snapshot_identity_differs_by_design(self, market):
        """Not a parity failure. The new boundary is more precise: its address
        depends only on the evaluation's own inputs."""
        comparison = compare(*both_paths(FIXTURES["monthly-contributions"],
                                         market))
        expected = comparison.by_verdict(Verdict.EXPECTED_IDENTITY_DIFFERENCE)
        assert {one.field for one in expected} >= {"market_snapshot_hash"}
        assert comparison.parity

    def test_every_tolerated_difference_carries_its_reason(self, market):
        """A whitelist records that somebody decided; a reason records why."""
        comparison = compare(*both_paths(FIXTURES["monthly-contributions"],
                                         market))
        for one in comparison.differences:
            if one.verdict in (Verdict.EXPECTED_IDENTITY_DIFFERENCE,
                               Verdict.APPLICATION_ONLY_DIFFERENCE):
                assert len(one.reason) > 40, (
                    f"{one.field} is tolerated with no explanation, which is a "
                    "whitelist entry wearing a verdict's name")

    def test_what_computed_the_figure_must_still_match(self, market):
        """The identity fields that say *what ran* rather than *on what*."""
        old, new = both_paths(FIXTURES["monthly-contributions"], market)
        for name in MUST_MATCH:
            assert getattr(old, name) == getattr(new, name), name


class TestTheComparisonWouldNoticeAMismatch:
    """Parity that cannot fail is not parity."""

    def test_a_changed_stream_is_an_evaluator_mismatch(self, market):
        import dataclasses

        old, new = both_paths(FIXTURES["monthly-contributions"], market)
        broken = dataclasses.replace(
            new, streams={**new.streams,
                          "fills": dataclasses.replace(
                              new.streams["fills"], rows=())})

        comparison = compare(old, broken)
        assert not comparison.parity
        assert any(one.field == "stream:fills" for one in comparison.mismatches)

    def test_a_reordered_stream_is_a_mismatch(self, market):
        """Ordering and counts are part of the comparison rather than a
        separate check somebody has to remember."""
        import dataclasses

        old, new = both_paths(FIXTURES["monthly-contributions"], market)
        rows = new.streams["fills"].rows
        broken = dataclasses.replace(
            new, streams={**new.streams,
                          "fills": dataclasses.replace(
                              new.streams["fills"],
                              rows=tuple(reversed(rows)))})
        assert not compare(old, broken).parity

    def test_a_different_strategy_hash_is_a_mismatch(self, market):
        import dataclasses

        old, new = both_paths(FIXTURES["monthly-contributions"], market)
        broken = dataclasses.replace(new, strategy_hash="spec1:something-else")
        assert not compare(old, broken).parity

    def test_a_different_disposition_is_a_mismatch(self, market):
        import dataclasses

        old, new = both_paths(FIXTURES["monthly-contributions"], market)
        broken = dataclasses.replace(new, refusals=("withheld",))
        assert not compare(old, broken).parity

    def test_evaluator_mismatch_is_not_tolerated(self):
        assert Verdict.EVALUATOR_MISMATCH not in TOLERATED


class TestTheReportSaysWhatWasChecked:
    def test_equivalent_fields_are_recorded_too(self, market):
        """A report listing only differences cannot distinguish "these agreed"
        from "this was never checked", and the second is how a comparison
        quietly stops covering something."""
        comparison = compare(*both_paths(FIXTURES["monthly-contributions"],
                                         market))
        equivalent = comparison.by_verdict(Verdict.EQUIVALENT)
        assert len(equivalent) >= len(MUST_MATCH)
        assert any(one.field.startswith("stream:") for one in equivalent)

    def test_every_stream_appears_in_the_comparison(self, market):
        from src.evaluation.service import STREAMS

        comparison = compare(*both_paths(FIXTURES["monthly-contributions"],
                                         market))
        compared = {one.field[len("stream:"):] for one in comparison.differences
                    if one.field.startswith("stream:")}
        assert compared == set(STREAMS)
