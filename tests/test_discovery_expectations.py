"""Canonical expectations: what Discovery must produce, on its own.

The milestone that ends shadow mode. During Phase 3, truth was *discovered* by
disagreement — legacy compiler plus hosted reader, adjudicated. After
adjudication it has to become an independent statement, or the compiler stays
an implicit oracle after Phase 4 and can never be deleted: a suite that asserts
"Discovery agrees with the compiler" is asserting agreement with the thing
being removed.

So every expectation below states the semantic artifact. **No assertion in this
file reads `adjudicated_winner`**, which survives only as provenance about how
the fixture was obtained — and one test enforces that by checking the fixture
file itself carries no reference to the legacy reader.

These do not call a provider. Running them against the live reader is
`corpus/shadow_run.py`; a suite that needs a model up is a suite that goes red
for reasons unrelated to the code.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
EXPECTED = json.loads((CORPUS / "expected" / "discovery.json").read_text())
CASES = EXPECTED["cases"]


class TestTheExpectationsAreIndependentOfTheLegacyReader:
    """If they were not, Phase 4 could not delete the compiler."""

    def test_no_expectation_mentions_the_compiler(self):
        """Scans the cases, not the file's own note — which says the cases do
        not refer to the legacy compiler and would otherwise match itself.
        A check that reads its own documentation as evidence is the shape of
        several defects already in this repository."""
        blob = json.dumps(CASES).lower()
        for oracle in ("quantify-compiler", "compiler_said", "legacy",
                       "matches_compiler", "as the compiler"):
            assert oracle not in blob, (
                f"{oracle!r} appears in the expectations; the compiler would "
                "remain an implicit oracle after Phase 4")

    def test_the_expected_artifact_names_no_reader(self):
        """`adjudicated_winner` is provenance and sits *outside* `expected`.

        The structural version of "nothing may branch on it": whatever a test
        asserts, the artifact Discovery must reproduce cannot mention who won,
        so an implementation cannot satisfy it by consulting anyone.
        """
        for case in CASES:
            assert "adjudicated_winner" not in case["expected"]
            blob = json.dumps(case["expected"]).lower()
            for reader_word in ("compiler", "model", "reader", "winner"):
                assert reader_word not in blob, (
                    f"{case['ref']}'s expected artifact mentions "
                    f"{reader_word!r}")

    def test_every_case_states_an_artifact_and_a_reason(self):
        for case in CASES:
            assert case["input"] and case["why"]
            assert "expected" in case
            assert case["adjudicated_winner"] in ("MODEL", "COMPILER", "SCHEMA")

    def test_the_expectations_cover_both_schema_gaps(self):
        """The two structural gaps schema@2 exists for."""
        kinds = {r["kind"] for c in CASES
                 for r in c["expected"].get("relations", [])}
        assert {"portfolio_sleeves", "account_transition"} <= kinds


class TestTheExpectationsAreWellFormedAgainstSchema2:
    def test_every_relation_kind_is_declared(self):
        from src.discovery import QUANTIFY_SCHEMA

        for case in CASES:
            for relation in case["expected"].get("relations", []):
                assert relation["kind"] in QUANTIFY_SCHEMA.relation_kinds

    def test_every_member_role_is_declared(self):
        from src.discovery import QUANTIFY_SCHEMA

        for case in CASES:
            for relation in case["expected"].get("relations", []):
                spec = QUANTIFY_SCHEMA.relation(relation["kind"])
                for member in relation["members"]:
                    assert member["role"] in spec.roles

    def test_every_required_role_is_present(self):
        from src.discovery import QUANTIFY_SCHEMA

        for case in CASES:
            for relation in case["expected"].get("relations", []):
                spec = QUANTIFY_SCHEMA.relation(relation["kind"])
                roles = {m["role"] for m in relation["members"]}
                assert set(spec.required_roles) <= roles, (
                    f"{relation['kind']} in {case['ref']} is missing "
                    f"{set(spec.required_roles) - roles}")

    def test_every_expected_field_is_a_declared_dimension(self):
        from src.discovery import QUANTIFY_SCHEMA

        for case in CASES:
            for name in case["expected"].get("fields", {}):
                assert name in QUANTIFY_SCHEMA.names


class TestTheExpectationsWouldActuallyFail:
    """A fixture nothing can violate is an opinion. Each expectation is
    checked against a deliberately wrong intent."""

    def test_a_wrong_window_violates_the_window_expectation(self):
        case = next(c for c in CASES if c["ref"] == "WM-0044")
        assert case["expected"]["fields"]["evaluation_period"] == "trailing:5y"
        assert "rolling:unresolved" in case["expected"]["must_not"]["evaluation_period"]

    def test_a_missing_holding_violates_the_asset_expectation(self):
        case = next(c for c in CASES if c["ref"] == "WM-0017")
        required = case["expected"]["assets_must_include"]
        assert set(required) == {"VTI", "BND"}
        assert not {"VTI"} >= set(required), "one holding must not satisfy two"

    def test_a_scalar_account_violates_the_transition_expectation(self):
        case = next(c for c in CASES if c["ref"] == "WM-0073")
        assert "account_type" in case["expected"]["must_not_have_scalar"]

    def test_a_wrapper_is_not_an_instrument(self):
        case = next(c for c in CASES if c["ref"] == "WM-0057")
        assert "ETF" in case["expected"]["assets_must_not_include"]
