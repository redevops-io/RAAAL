"""Whether Quantify refuses what it cannot model, across twenty families.

A web sweep of planning literature found roughly twenty strategy families
people bring to a planner. The phrasing pack contained examples of three of
them, all accumulation — because the sources it could lawfully read are places
where people ask how to put money in, not where they state a decumulation plan.

**The property is refusal, not understanding.** Quantify runs a buy-and-hold
engine and the manifest is candid about it: `sell_action` is REFUSED —
*selling, withdrawing and harvesting are not modelled* — and `tax_treatment` is
NOT_MODELLED. For those families the rule is:

    if the engine cannot model the semantic, Discovery must preserve enough of
    it for Mission to refuse it BY NAME

so each case is scored by asking `refusals_for` what was read, not by checking
whether one nominated dimension appeared. An earlier version did the latter and
was wrong in both directions — it called asset location understood because
`account_type` happened to appear, and called a withdrawal unhandled when the
model had read `objective=assess_withdrawal` that Mission would refuse on.

`SILENTLY_REDUCED` is ranked worse than `NOTHING_READ`. A sentence that reads as
nothing sends somebody back to rephrase; a fragment sends them away with a
figure computed for a strategy they did not describe.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parent.parent / "corpus" / "parser"


@pytest.fixture(scope="module")
def cases():
    return json.loads((CORPUS / "strategy_families.json").read_text())


@pytest.fixture(scope="module")
def compiler():
    """The legacy reader. Deterministic, so its numbers can be pinned."""
    import sys

    sys.path.insert(0, str(CORPUS))
    from strategy_closure import measure

    return measure("compiler")


@pytest.fixture(scope="module")
def serving():
    """The reader a deployment actually serves — one recorded draw from it."""
    import sys

    sys.path.insert(0, str(CORPUS))
    from strategy_closure import measure

    return measure()


class TestTheCorpusIsWellFormed:
    def test_every_carrier_is_something_the_schema_can_say(self, cases):
        """A carrier the schema cannot express can never be read, so every case
        using it would score as a defect forever — a measurement that
        manufactures its own findings.

        Dimensions *or* relation kinds. `asset_location` is a relation, because
        the mapping is the meaning, and requiring carriers to be dimensions
        would have made the relation unusable as evidence that the request was
        read at all."""
        from src.discovery.schema import QUANTIFY_SCHEMA

        names = (set(QUANTIFY_SCHEMA.names)
                 | {r.kind for r in QUANTIFY_SCHEMA.relations})
        for case in cases["cases"]:
            unknown = [c for c in case["carriers"] if c not in names]
            assert not unknown, f"{case['id']} names {unknown}, not in the schema"

    def test_every_case_cites_where_the_family_came_from(self, cases):
        """The sentences are authored; the family list is not. The citation is
        what separates 'twenty families people ask about' from 'twenty things
        I thought of'."""
        for case in cases["cases"]:
            assert case["source"].startswith("http"), case["id"]

    def test_the_pack_declares_that_its_sentences_are_authored(self, cases):
        """`real_phrasings.json` is the attested pack and this is not it."""
        assert all(c["provenance"] == "authored_from_cited_definition"
                   for c in cases["cases"])
        assert "NOT attested" in cases["provenance_note"]


class TestTheExpectationsMatchTheManifest:
    """A family expected to be refused must be refusable, or the corpus asserts
    a boundary the engine never claimed."""

    def test_each_refused_family_can_actually_be_refused(self, cases):
        from src.mission.capability import MANIFEST

        manifest = dict(MANIFEST)
        for case in cases["cases"]:
            if case["must_be"] != "REFUSED_BY_NAME":
                continue
            for carrier in case["carriers"]:
                entry = manifest.get(carrier)
                assert entry is not None, (
                    f"{case['id']}: {carrier} has no manifest entry")
                # Either the whole dimension is refused, or it executes and
                # refuses particular values — `objective=assess_withdrawal` is
                # the second kind.
                assert entry.support in ("REFUSED", "NOT_MODELLED") \
                    or entry.refuses, (
                        f"{case['id']}: {carrier} is {entry.support} and "
                        "refuses nothing, so this family is not refusable")

    @pytest.mark.parametrize("objective,refused", [
        ("assess_withdrawal", True),
        ("assess_conversion", True),
        ("assess_debt_repayment", True),
        ("plan_contributions", False),
        ("evaluate_investment_strategy", False),
        ("other", False),
    ])
    def test_the_objective_dimension_is_classified(self, objective, refused):
        """`objective` had no manifest entry, so `decide` returned None — an
        unclassified dimension is not forbidden, which is the correct default
        and was the wrong answer. `other` stays executable deliberately:
        refusing it would convert absence of classification into a refusal.
        """
        from src.mission.capability import decide

        assert (decide("objective", objective) is not None) is refused

    def test_every_refusable_objective_is_in_the_schema(self):
        """Mission may only refuse what Discovery can say. A manifest refusing
        a value no reader can produce is a boundary nothing reaches."""
        from src.discovery.schema import QUANTIFY_SCHEMA
        from src.mission.capability import MANIFEST

        vocabulary = {d.name: d.values for d in QUANTIFY_SCHEMA.dimensions}
        for value in dict(MANIFEST)["objective"].refuses:
            assert value in vocabulary["objective"], (
                f"the manifest refuses objective={value}, which no reader can "
                "propose because the schema has no such value")


class TestTheLegacyReaderIsOffTheServingPath:
    def test_nothing_in_src_constructs_it(self):
        """The architecture decision, checked structurally rather than by
        reading the routes. `quantify-compiler@2` is the source of the
        accumulation bias; it survives as a corpus comparator and must not
        return to serving.
        """
        import ast

        offenders = []
        for path in (Path(__file__).resolve().parent.parent / "src").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and \
                        getattr(node.func, "id", "") == "CompilerReader":
                    offenders.append(path.name)
        assert not offenders, (
            f"CompilerReader is constructed in {offenders}; the legacy reader "
            "is back on a serving path")

    def test_the_serving_reader_is_the_hosted_one(self):
        """What a runtime deployment hands to Discovery."""
        import inspect

        from src.workspace import pilot_routes

        source = inspect.getsource(pilot_routes.configured_reader)
        assert "HostedReader" in source
        assert "CompilerReader" not in source


class TestTheCompilerBaseline:
    """Pinned, because this reader is deterministic. It is a defect report."""

    #: Back to 11, and neither 11 nor the intermediate 9 was a change in the
    #: compiler. The instrument moved twice: scoring by carrier presence gave
    #: 11, scoring by refusal gave 9, and folding relations into what Mission
    #: is asked about gave 11 again — because the relation-shaped families the
    #: compiler cannot read at all now count as reductions rather than as
    #: schema gaps. The reader has been constant throughout.
    SILENTLY_REDUCED = 11

    def test_the_legacy_reader_still_reduces_this_many(self, compiler):
        assert compiler["by_state"].get("SILENTLY_REDUCED", 0) == \
            self.SILENTLY_REDUCED, (
                "the legacy baseline moved; it is deterministic, so something "
                "changed in the compiler or the corpus")

    @pytest.mark.parametrize("text,how", [
        ("sell VTI and buy BND", "the sell becomes a purchase of both"),
        ("convert $30,000 from the traditional IRA to the Roth each year",
         "indistinguishable from contributing $30,000 to a Roth annually"),
        ("withdraw 4% of the portfolio each year, adjusted for inflation",
         "reads as an annual cadence with no withdrawal"),
    ])
    def test_the_named_legacy_defects(self, compiler, text, how):
        states = {c["text"]: c["state"] for c in compiler["cases"]}
        assert states.get(text) in ("SILENTLY_REDUCED", "NOTHING_READ"), how


class TestTheServingReaderRefusesFarMore:
    """Direction, not magnitude. The magnitude is not stable — see below."""

    def test_it_refuses_where_the_compiler_stayed_silent(self, compiler,
                                                         serving):
        assert serving["by_state"].get("REFUSED", 0) > \
            compiler["by_state"].get("REFUSED", 0) * 3

    def test_and_reduces_far_fewer(self, compiler, serving):
        assert serving["by_state"].get("SILENTLY_REDUCED", 0) < \
            compiler["by_state"].get("SILENTLY_REDUCED", 0)


class TestTheServingReaderIsNotStable:
    """The finding that decides what this corpus may claim.

    Two recordings of the same model, same prompt version, differed on 24 of
    36 sentences — only one attributable to a schema change. Two sentences lost
    a `sell_action` they previously had, and one inverted its trigger
    semantics from `persistent_condition` to `crossing_event`, which is the
    exact defect class this project already names.

    So a gate of the form `SILENTLY_REDUCED == 0` is not well defined against
    this witness alone: the number moves when nothing changed. That is an
    argument for a second, deterministic witness on the serving path rather
    than for pinning a number that will drift.
    """

    ARCHIVE = CORPUS / "strategy_closure_hosted.json"

    def test_two_recordings_of_one_model_disagree(self, serving):
        if not self.ARCHIVE.exists():
            pytest.skip("no earlier recording retained to compare against")

        earlier = {c["text"]: c["read"]
                   for c in json.loads(self.ARCHIVE.read_text())["cases"]}
        now = {c["text"]: c["read"] for c in serving["cases"]}
        moved = [t for t in earlier if earlier[t] != now.get(t)]
        assert len(moved) > 10, (
            "the two archived recordings now agree closely. If the model or "
            "prompt was stabilised, say so here — this test exists to stop a "
            "single stochastic draw being quoted as a fixed measurement")

    def test_so_no_hosted_count_is_pinned_in_this_file(self):
        """Checked structurally, because the temptation is to add one.

        Pinning a hosted count would produce a test that fails on
        re-recording for reasons unrelated to any change in the code, and the
        usual repair is to update the constant — which quietly converts a
        measurement into whatever the last draw happened to say.
        """
        import ast

        tree = ast.parse(Path(__file__).read_text())
        pinned = [n.targets[0].id for n in ast.walk(tree)
                  if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name)
                  and n.targets[0].id.isupper()
                  and isinstance(n.value, ast.Constant)
                  and isinstance(n.value.value, int)]
        assert pinned == ["SILENTLY_REDUCED"], (
            f"{pinned} are pinned integers; only the deterministic compiler "
            "baseline may be one")


class TestTheAssetLocationGapIsClosed:
    """It was the one schema gap left standing, and it is now a relation.

    `account_type` *was* read from "hold the bonds in the IRA and the stocks in
    the taxable account" — it returned TAXABLE — so the family scored as
    understood while the mapping, which is the entire request, was gone. A
    single-valued dimension cannot carry a mapping.
    """

    def test_it_is_refused_by_name_rather_than_scored_as_a_gap(self, serving):
        states = {c["text"]: c["state"] for c in serving["cases"]}
        assert states["keep the REITs in the Roth"] == "REFUSED"
        assert states["hold the bonds in the IRA and the stocks in the "
                      "taxable account"] == "REFUSED"

    def test_the_reader_returns_the_mapping_not_two_lists(self):
        """What makes the refusal honest. A relation naming one account and
        two holdings would be refused too, and would have lost the pairing on
        the way."""
        from src.discovery.hosted_recording import RecordedHostedReader
        from src.discovery.schema import QUANTIFY_SCHEMA

        reading = RecordedHostedReader().read(
            "hold the bonds in the IRA and the stocks in the taxable account",
            QUANTIFY_SCHEMA)
        pairs = [dict((role, subject) for role, subject, *_ in r.members)
                 for r in reading.relations if r.kind == "asset_location"]
        assert len(pairs) == 2, f"expected two placements, got {pairs}"
        assert {p["holding"] for p in pairs} == {"bonds", "stocks"}

    def test_no_case_is_a_schema_gap_any_more(self, serving):
        assert serving["by_state"].get("SCHEMA_GAP", 0) == 0


class TestTheReportSeparatesTheFailures:
    def test_a_sentence_read_as_nothing_is_not_a_silent_reduction(self,
                                                                  serving):
        for case in serving["cases"]:
            if case["state"] == "NOTHING_READ":
                assert case["read"] == {}, (
                    f"{case['id']} was called NOTHING_READ with "
                    f"{case['read']} in hand")

    def test_a_silent_reduction_read_something_and_earned_no_refusal(
            self, serving):
        for case in serving["cases"]:
            if case["state"] == "SILENTLY_REDUCED":
                assert case["read"], f"{case['id']} reduced to nothing at all"
                assert not case["refused"], (
                    f"{case['id']} was refused and still called reduced")

    def test_the_report_names_its_witness(self, serving, compiler):
        """A report that did not name its reader would read as a fact about
        Quantify rather than about one witness."""
        assert serving["witness"] != compiler["witness"]
        assert "MODEL_ONLY" in serving["witness_note"]
