"""Whether Quantify can *name* what people ask for, across twenty families.

A web sweep of planning literature found roughly twenty strategy families
people bring to a planner. The phrasing pack contained examples of three of
them, all accumulation — because the sources it could lawfully read are places
where people ask how to put money in, not where they state a decumulation plan.

**The property here is not "the parser understands this".** Quantify runs a
buy-and-hold engine and the manifest is candid about it: `sell_action` is
REFUSED — *selling, withdrawing and harvesting are not modelled* — and
`tax_treatment` is NOT_MODELLED. For those families, being understood is not
the goal. Being *named* is, because a refusal that never fires is not a
boundary, and the fragment that survives an unfired refusal is always
accumulation-shaped.

That is what makes `SILENTLY_REDUCED` worse than `NOTHING_READ`. A sentence
that reads as nothing sends somebody back to rephrase. A sentence that reads as
a fragment sends them away with a figure computed for a strategy they did not
describe.
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
def report():
    import sys

    sys.path.insert(0, str(CORPUS))
    from strategy_closure import measure

    return measure()


class TestTheCorpusIsWellFormed:
    def test_every_carrier_is_a_real_schema_dimension(self, cases):
        """A carrier naming a dimension that does not exist can never be read,
        so every case using it would score as a defect forever — a measurement
        that manufactures its own findings."""
        from src.discovery.schema import QUANTIFY_SCHEMA

        names = set(QUANTIFY_SCHEMA.names)
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
        """`real_phrasings.json` is the attested pack and this is not it.

        Counting these as evidence about phrasing coverage would be the
        self-authored-evidence defect: the sentences would be measuring how
        well the parser handles sentences written by someone who knew what the
        parser does.
        """
        assert all(c["provenance"] == "authored_from_cited_definition"
                   for c in cases["cases"])
        assert "NOT attested" in cases["provenance_note"]


class TestTheExpectationsMatchTheManifest:
    """A family expected to be refused must be refusable.

    Otherwise the corpus asserts a boundary the engine never claimed, and the
    report counts a defect against behaviour nothing promised.
    """

    #: Carriers with no manifest entry, and why that is itself a finding.
    #: Declared rather than skipped: an exception needs a stated reason, and
    #: these two are the reason the list exists.
    NO_MANIFEST_ENTRY: dict = {
        # `objective` was here. The hosted-reader measurement showed the model
        # reads "draw down 3% a year in retirement" correctly as
        # `assess_withdrawal`, and `decide` then executed it, because an
        # unclassified dimension is treated as not forbidden. It now has a
        # manifest entry and the check below is what removed it from this list.
    }

    def test_each_refused_family_can_actually_be_refused(self, cases):
        from src.mission.capability import MANIFEST

        manifest = dict(MANIFEST)
        for case in cases["cases"]:
            if case["must_be"] != "REFUSED_BY_NAME":
                continue
            for carrier in case["carriers"]:
                if carrier in self.NO_MANIFEST_ENTRY:
                    continue
                entry = manifest.get(carrier)
                assert entry is not None, (
                    f"{case['id']}: {carrier} is neither in the manifest nor "
                    "declared as a gap")
                # Either the whole dimension is refused, or it executes but
                # refuses particular values — 'allocate by inverse volatility'
                # is the second kind.
                assert entry.support in ("REFUSED", "NOT_MODELLED") \
                    or entry.refuses, (
                        f"{case['id']}: {carrier} is {entry.support} and "
                        "refuses nothing, so this family is not refusable")

    def test_the_declared_gaps_are_still_gaps(self):
        """If a manifest entry appears later, this list must shrink.

        A declared exception that has quietly become unnecessary is how an
        escape hatch outlives its reason.
        """
        from src.mission.capability import MANIFEST

        manifest = dict(MANIFEST)
        for carrier in self.NO_MANIFEST_ENTRY:
            assert carrier not in manifest, (
                f"{carrier} is now in the manifest; remove it from "
                "NO_MANIFEST_ENTRY and let the real check cover it")


class TestWhatTheEngineCurrentlyDoes:
    """The measurement, pinned. These numbers are a defect report, not a goal.

    They are asserted so that the gap cannot widen unnoticed and so that
    closing any part of it shows up as a failing test asking to be updated.
    """

    #: Measured against `quantify-compiler@2` on 2026-08-09.
    SILENTLY_REDUCED_TODAY = 11

    def test_the_silent_reduction_count_has_not_grown(self, report):
        reduced = report["by_state"].get("SILENTLY_REDUCED", 0)
        assert reduced <= self.SILENTLY_REDUCED_TODAY, (
            f"{reduced} sentences now produce a plan for a strategy nobody "
            f"described, up from {self.SILENTLY_REDUCED_TODAY}")

    def test_and_if_it_shrank_this_number_should_follow(self, report):
        reduced = report["by_state"].get("SILENTLY_REDUCED", 0)
        assert reduced == self.SILENTLY_REDUCED_TODAY, (
            f"{reduced} silent reductions, fewer than the recorded "
            f"{self.SILENTLY_REDUCED_TODAY}. Something was fixed — lower the "
            "constant so the improvement is locked in")

    @pytest.mark.parametrize("text,wrong_reading", [
        ("sell VTI and buy BND", "the sell becomes a purchase of both"),
        ("convert $30,000 from the traditional IRA to the Roth each year",
         "indistinguishable from contributing $30,000 to a Roth annually"),
        ("withdraw 4% of the portfolio each year, adjusted for inflation",
         "reads as an annual cadence with no withdrawal"),
    ])
    def test_the_named_defects_are_still_there(self, report, text,
                                               wrong_reading):
        """Named individually so that fixing one is visible as one test going
        green, rather than a count moving by an amount nobody can attribute."""
        states = {c["text"]: c["state"] for c in report["cases"]}
        assert states.get(text) in ("SILENTLY_REDUCED", "NOTHING_READ"), (
            f"{text!r} no longer misreads ({wrong_reading}) — update this list")


class TestAsymmetryBetweenSchemaAndRecognition:
    """Asset location is a schema gap, and saying so sends the work somewhere.

    `account_type` *is* read from "hold the bonds in the IRA and the stocks in
    the taxable account" — it returns TAXABLE. Scored against that carrier the
    family looks understood, while the thing asked for, a mapping of holdings
    onto accounts, is gone. A single-valued dimension cannot carry a mapping,
    so no amount of recognition work fixes this one and counting it as a
    recognition defect would send the effort to the wrong layer.
    """

    def test_asset_location_is_scored_as_a_schema_gap(self, report):
        states = {c["text"]: c["state"] for c in report["cases"]}
        assert states["hold the bonds in the IRA and the stocks in the "
                      "taxable account"] == "SCHEMA_GAP"

    def test_and_the_reading_that_made_it_look_fine_is_still_there(self,
                                                                   report):
        """The trap, kept visible. If `account_type` stopped being read this
        would pass for the wrong reason — the gap would look closed because
        recognition got worse."""
        case = [c for c in report["cases"]
                if c["family"] == "asset_location"][0]
        assert case["read"].get("account_type"), (
            "account_type no longer reads here; this family stopped being a "
            "trap and the test above stopped proving anything")

    def test_a_schema_gap_declares_no_carrier(self, cases):
        for case in cases["cases"]:
            if case["must_be"] == "NO_DIMENSION":
                assert case["carriers"] == [], (
                    f"{case['id']} claims no dimension exists but names one")


class TestTheReportSeparatesTheTwoFailures:
    """The distinction the whole report rests on.

    If these two states collapsed into one, the number that matters — how often
    somebody gets a figure for a strategy they did not describe — would be
    hidden inside a general "did not work" count.
    """

    def test_a_sentence_read_as_nothing_is_not_a_silent_reduction(self, report):
        for case in report["cases"]:
            if case["state"] == "NOTHING_READ":
                assert case["read"] == {}, (
                    f"{case['id']} was called NOTHING_READ with "
                    f"{case['read']} in hand")

    def test_a_silent_reduction_always_has_something_it_read_instead(
            self, report):
        for case in report["cases"]:
            if case["state"] == "SILENTLY_REDUCED":
                assert case["read"], f"{case['id']} reduced to nothing at all"
                assert not any(c in case["read"] for c in case["carriers"]), (
                    f"{case['id']} read its own carrier and was still called "
                    "reduced")

    def test_the_report_says_which_witness_it_used(self, report):
        """One reader, and the pilot profile has only one too.

        A report that did not name its witness would read as a fact about
        Quantify rather than about `quantify-compiler@2`, and under MODEL_ONLY
        there is no second reader to catch the model missing the same
        dimension.
        """
        assert report["witness"] == "quantify-compiler@2"
        assert "MODEL_ONLY" in report["witness_note"]


@pytest.fixture(scope="module")
def hosted():
    import sys

    sys.path.insert(0, str(CORPUS))
    from strategy_closure import measure

    return measure("hosted")


class TestWhichReaderIsActuallyAtFault:
    """The measurement that decides where the work goes.

    Eleven silent reductions against `quantify-compiler@2` is a finding about
    the reader being deleted. Whether it is also a finding about Discovery is a
    different question, and running the same 36 cases through the hosted reader
    answers it: the model carries 25 where the compiler carries 4, and emits
    `sell_action` on twelve sentences the compiler read as purchases or as
    nothing.

    So this is overwhelmingly an argument for finishing the legacy-reader
    deletion rather than teaching it to recognise decumulation.
    """

    def test_the_hosted_reader_carries_far_more(self, report, hosted):
        assert hosted["by_state"].get("CARRIED", 0) > \
            report["by_state"].get("CARRIED", 0) * 4

    def test_and_the_sharpest_inversion_is_gone(self, hosted):
        """`sell VTI and buy BND` reads as a purchase of both under the
        compiler. The plan holds what the person said they were disposing of,
        which is not an approximation of their request but its opposite."""
        case = [c for c in hosted["cases"] if c["text"] == "sell VTI and buy BND"][0]
        assert case["state"] == "CARRIED"
        assert "sell" in str(case["read"].get("sell_action", "")).lower()

    def test_a_sell_action_reading_earns_a_refusal_by_name(self):
        """Recognition is only worth something if the manifest then fires.

        Asserted through `decide` rather than by reading the manifest table,
        because the table says what is refused and this says that asking
        produces the refusal.
        """
        from src.mission.capability import decide

        refusal = decide("sell_action", "sell VTI")
        assert refusal is not None
        assert refusal.dimension == "sell_action"

    def test_a_withdrawal_objective_is_refused_rather_than_executed(self):
        """The gap the hosted measurement exposed.

        The model read "draw down 3% a year in retirement" correctly. Mission
        executed it, because `decide` treats an unclassified dimension as not
        forbidden — the right default and the wrong answer here.
        """
        from src.mission.capability import decide

        assert decide("objective", "assess_withdrawal") is not None
        assert decide("objective", "plan_contributions") is None, (
            "classifying the objective dimension must not refuse the "
            "objectives this build does execute")

    def test_four_cases_got_worse_and_that_is_recorded(self, report, hosted):
        """Honest counterweight: the model reads a fragment where the compiler
        read nothing, and by this report's own ranking that is a regression —
        NOTHING_READ sends somebody back to rephrase, SILENTLY_REDUCED does
        not. Named so the hosted reader is not reported as a pure improvement.
        """
        before = {c["id"]: c["state"] for c in report["cases"]}
        after = {c["id"]: c["state"] for c in hosted["cases"]}
        worse = [i for i in before
                 if before[i] == "NOTHING_READ" and after[i] == "SILENTLY_REDUCED"]
        assert sorted(worse) == ["bucket_strategy-01", "cash_reserve-01",
                                 "glidepath-02", "leverage-01"]
