"""The result-completeness contract.

    Anything that can materially qualify, delay, invalidate or reinterpret an
    RSU result travels on the result itself. It cannot remain only in an
    engine-local diagnostic.

Every stage of this pipeline produces one — unpriced arrivals, unsettled
dispositions, unfilled allocation targets, missing concentration prices,
benchmark verdicts — and each was, until this envelope existed, available at the
point of computation and nowhere afterwards.

The mutation test at the bottom draws its inventory by **introspecting the
engine types**, not from the destination registry it checks. Parametrizing a
guard over the same declaration it guards is how a list can shrink without
anything failing; that happened in step 8 and is not repeated here.
"""
from __future__ import annotations

import dataclasses
import re

import pytest

from src.mission.rsu_result import (
    DESTINATIONS,
    POST_WITHHOLDING_BASIS,
    AllocationContext,
    ComparisonContext,
    ConcentrationContext,
    DiagnosticKind,
    DispositionContext,
    Presentability,
    RSUResultContext,
    ScopeStatus,
    VestAccountingContext,
    from_json,
)

SCOPE = {"modelled": ["share delivery"], "out_of_scope": ["capital-gains tax"]}


def clean(**overrides) -> RSUResultContext:
    """A run where everything completed."""
    base = dict(
        vest_accounting=VestAccountingContext(
            gross_vest_value=5_000.0, withheld_value=1_100.0,
            delivered_value=3_900.0, cash_remainder=0.0),
        disposition=DispositionContext(status="EXECUTED"),
        allocation=AllocationContext(requested_targets={"VTI": 1.0},
                                     executed_targets={"VTI": 3_892.0},
                                     residual_cash=0.0, unallocated_weight=0.0),
        concentration=ConcentrationContext(
            current=0.5, target=0.2, projected=0.199, realized=0.198,
            denominator_scope=("settled holdings", "settled cash")),
        comparisons=ComparisonContext(verdict_rows=(
            {"benchmark_id": "hold", "status": "COMPARABLE",
             "unchecked_dimensions": []},)),
        modelling_scope=SCOPE)
    base.update(overrides)
    return RSUResultContext(**base)


class TestThreeConceptsStayApart:

    def test_a_limitation_is_not_a_data_gap(self):
        context = clean()
        kinds = {one.kind for one in context.diagnostics()}
        assert DiagnosticKind.LIMITATION in kinds
        assert DiagnosticKind.DATA_GAP not in kinds

    def test_an_unpriced_arrival_is_a_data_gap(self):
        context = clean(vest_accounting=VestAccountingContext(
            unpriced_arrivals=({"asset": "ACME", "source_ref": "vest:g1"},)))
        assert any(one.kind is DiagnosticKind.DATA_GAP
                   for one in context.diagnostics())

    def test_a_pending_sale_is_unsettled_not_a_failure(self):
        context = clean(disposition=DispositionContext(
            status="PENDING",
            pending_instructions=({"instruction_id": "d1"},)))
        kinds = {one.kind for one in context.diagnostics()}
        assert DiagnosticKind.UNSETTLED in kinds
        assert DiagnosticKind.EXECUTION_FAILURE not in kinds

    def test_a_failed_sale_is_an_execution_failure(self):
        context = clean(disposition=DispositionContext(
            status="FAILED",
            failed_instructions=({"instruction_id": "d1",
                                  "why": "never filled"},)))
        assert any(one.kind is DiagnosticKind.EXECUTION_FAILURE
                   for one in context.diagnostics())

    def test_an_unfilled_target_is_a_partial_result(self):
        context = clean(allocation=AllocationContext(
            unfilled_targets=({"asset": "BND", "why": "no price"},)))
        assert any(one.kind is DiagnosticKind.PARTIAL_RESULT
                   for one in context.diagnostics())


class TestAbsenceIsNotCompleteness:

    def test_an_undeclared_context_is_not_complete(self):
        """A result nobody examined must not render like a clean one."""
        empty = RSUResultContext()
        assert empty.scope_status is ScopeStatus.NOT_DECLARED
        assert empty.presentability is not Presentability.COMPLETE

    def test_an_undeclared_context_is_blocked(self):
        assert RSUResultContext().presentability is Presentability.BLOCKED

    def test_a_declared_clean_context_is_complete(self):
        assert clean().presentability is Presentability.COMPLETE


class TestPresentability:

    def test_a_data_gap_blocks(self):
        context = clean(vest_accounting=VestAccountingContext(
            unpriced_arrivals=({"asset": "ACME"},)))
        assert context.presentability is Presentability.BLOCKED

    def test_a_failed_sale_blocks(self):
        context = clean(disposition=DispositionContext(
            failed_instructions=({"instruction_id": "d1", "why": "unfilled"},)))
        assert context.presentability is Presentability.BLOCKED

    def test_a_targeted_cap_with_no_realized_measurement_blocks(self):
        """A cap claimed but never measured is the claim this refuses."""
        context = clean(concentration=ConcentrationContext(
            target=0.2, projected=0.199, realized=None))
        assert context.presentability is Presentability.BLOCKED

    def test_an_unfilled_target_is_partial_not_blocked(self):
        context = clean(allocation=AllocationContext(
            unfilled_targets=({"asset": "BND", "why": "no price"},),
            residual_cash=389.0))
        assert context.presentability is Presentability.PARTIAL

    def test_a_comparison_with_unchecked_dimensions_is_partial(self):
        context = clean(comparisons=ComparisonContext(verdict_rows=(
            {"benchmark_id": "hold",
             "status": "COMPARABLE_WITH_UNCHECKED_DIMENSIONS",
             "unchecked_dimensions": ["tax_runtime"]},)))
        assert context.presentability is Presentability.PARTIAL

    def test_limitations_alone_do_not_reduce_presentability(self):
        """Every run has limitations. If they downgraded the result, nothing
        would ever be complete and the status would stop meaning anything."""
        assert clean().presentability is Presentability.COMPLETE


class TestProjectedAndRealizedStayApart:

    def test_a_projection_alone_cannot_claim_the_cap(self):
        context = clean(concentration=ConcentrationContext(
            target=0.2, projected=0.199, realized=None))
        assert context.concentration.cap_achieved is None

    def test_a_realized_measurement_can(self):
        assert clean().concentration.cap_achieved is True

    def test_a_realized_miss_says_so(self):
        context = clean(concentration=ConcentrationContext(
            target=0.2, projected=0.199, realized=0.24))
        assert context.concentration.cap_achieved is False

    def test_a_projection_is_labelled_projected(self):
        context = clean(concentration=ConcentrationContext(
            target=0.2, projected=0.199, realized=None))
        [note] = [one for one in context.diagnostics()
                  if one.code == "concentration_projected"]
        assert "PROJECTED" in note.detail


class TestTheBasisTravels:

    def test_every_result_carries_the_post_withholding_basis(self):
        assert clean().to_json()["vest_accounting"]["basis_note"] == \
            POST_WITHHOLDING_BASIS

    def test_it_says_it_is_not_gross_compensation(self):
        assert "not represent gross compensation" in POST_WITHHOLDING_BASIS

    def test_it_says_it_is_not_final_tax_liability(self):
        assert "final tax liability" in POST_WITHHOLDING_BASIS

    def test_the_household_scope_travels_with_concentration(self):
        assert "external assets" in clean().to_json()["concentration"]["scope_note"]


class TestNothingIsDroppedInStorage:
    """engine -> stored -> reopened, verbatim."""

    def full(self) -> RSUResultContext:
        return RSUResultContext(
            vest_accounting=VestAccountingContext(
                gross_vest_value=5_000.0, withheld_value=1_100.0,
                delivered_value=3_900.0, cash_remainder=0.25,
                unpriced_arrivals=({"asset": "XYZ", "quantity": 3.0,
                                    "source_ref": "vest:g2",
                                    "why": "no usable price"},)),
            disposition=DispositionContext(
                status="PARTIAL",
                pending_instructions=({"instruction_id": "d1",
                                       "status": "PENDING"},),
                failed_instructions=({"instruction_id": "d2",
                                      "why": "never filled"},),
                unsettled_report=({"instruction_id": "d2",
                                   "quantity": 40.0},)),
            allocation=AllocationContext(
                requested_targets={"VTI": 0.6, "BND": 0.4},
                executed_targets={"VTI": 2_300.0},
                unfilled_targets=({"asset": "BND", "requested_weight": 0.4,
                                   "why": "could not be priced"},),
                residual_cash=1_500.0, unallocated_weight=0.4),
            concentration=ConcentrationContext(
                current=0.5, target=0.2, projected=0.199, realized=None,
                missing_prices=("BND",), unresolved_inputs=("BND",),
                denominator_scope=("settled holdings", "settled cash"),
                excluded_components=("unvested grants",)),
            comparisons=ComparisonContext(verdict_rows=(
                {"benchmark_id": "hold", "status": "COMPARABLE",
                 "unchecked_dimensions": []},
                {"benchmark_id": "value_matched", "status": "INCOMPARABLE",
                 "reason": "cost model differs", "unchecked_dimensions": []},
                {"benchmark_id": "never_ran", "status": "NOT_EVALUATED",
                 "reason": "not built", "unchecked_dimensions": []})),
            modelling_scope=SCOPE)

    def test_a_round_trip_preserves_everything(self):
        original = self.full()
        assert from_json(original.to_json()).to_json() == original.to_json()

    @pytest.mark.parametrize("path", [
        ("vest_accounting", "unpriced_arrivals"),
        ("disposition", "unsettled_report"),
        ("disposition", "failed_instructions"),
        ("allocation", "unfilled_targets"),
        ("allocation", "residual_cash"),
        ("concentration", "missing_prices"),
        ("comparisons", "verdict_rows"),
    ])
    def test_each_diagnostic_survives_verbatim(self, path):
        original = self.full()
        reopened = from_json(original.to_json())

        section, name = path
        assert getattr(getattr(reopened, section), name) == \
            getattr(getattr(original, section), name)

    def test_every_requested_benchmark_row_survives(self):
        reopened = from_json(self.full().to_json())
        assert [row["benchmark_id"] for row in reopened.comparisons.verdict_rows] \
            == ["hold", "value_matched", "never_ran"]

    def test_an_incomparable_row_is_not_filtered_on_the_way_out(self):
        rendered = from_json(self.full().to_json()).to_json()
        statuses = {row["status"] for row in rendered["comparisons"]["verdict_rows"]}
        assert "INCOMPARABLE" in statuses
        assert "NOT_EVALUATED" in statuses

    def test_the_context_version_survives(self):
        assert from_json(self.full().to_json()).context_version == \
            self.full().context_version


class TestEveryDiagnosticHasADestination:
    """The mutation test.

    The inventory is discovered by introspecting the engine's own result types,
    deliberately *not* by reading `DESTINATIONS`. A guard parametrized over the
    declaration it guards can only catch changed entries, never removed ones —
    which is exactly how a field was dropped from the comparability check in
    step 8 while every test stayed green.
    """

    #: Names that denote something a reader must be told about.
    DIAGNOSTIC_SHAPES = (
        r"^unpriced_", r"^unsettled", r"^unfilled_", r"^missing_",
        r"^residual_", r"^failed_", r"^pending_", r"^unresolved_",
        r"^unchecked_", r"^unallocated_", r"^excluded_", r"_remainder$",
    )

    #: Declarations, not diagnostics. `missing_price_policy` says what the
    #: engine *will do* about a missing price; it is not a report that one
    #: occurred, and requiring it to land on a result would be noise.
    DECLARATION_SUFFIXES = ("_policy", "_model")

    def discovered(self):
        """Material diagnostics, found in the engine rather than declared.

        Three shapes, because a diagnostic is not always a dataclass field:
        `unsettled_report` is a method, and `unpriced_in_kind_arrivals` and
        `cash_remainder` are dict keys. Scanning only fields would miss exactly
        the ones most easily dropped — the ones with no type to anchor them.
        """
        import ast
        import inspect

        from src.comparison import rsu_profile
        from src.mission import simulate as engine
        from src.runtime import allocation, concentration, disposition, rsu

        modules = (rsu, disposition, allocation, concentration, rsu_profile,
                   engine)
        candidates = set()

        for module in modules:
            for name in dir(module):
                member = getattr(module, name)
                if dataclasses.is_dataclass(member):
                    candidates.update(f.name for f in dataclasses.fields(member))
                if inspect.isclass(member):
                    candidates.update(attribute for attribute in dir(member)
                                      if not attribute.startswith("_"))

            # Dict keys and other string literals in the module's own source.
            try:
                tree = ast.parse(inspect.getsource(module))
            except (OSError, TypeError):                        # pragma: no cover
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    candidates.add(node.value)

        return {name for name in candidates
                if any(re.search(shape, name)
                       for shape in self.DIAGNOSTIC_SHAPES)
                and not name.endswith(self.DECLARATION_SUFFIXES)}

    def test_the_inventory_is_not_empty(self):
        """Guards the guard: a discovery that finds nothing proves nothing."""
        found = self.discovered()
        assert len(found) >= 10, found
        # The three that only exist as a method name or a dict key, and would
        # be invisible to a field-only scan.
        for name in ("unsettled_report", "unpriced_in_kind_arrivals",
                     "cash_remainder"):
            assert name in found, name

    def test_every_discovered_diagnostic_has_a_declared_destination(self):
        missing = sorted(self.discovered() - set(DESTINATIONS))
        assert not missing, (
            "these engine diagnostics have nowhere to land on a result: "
            f"{missing}. Add a destination in rsu_result.DESTINATIONS")

    def test_every_destination_names_a_real_context_field(self):
        sections = {"vest_accounting": VestAccountingContext,
                    "disposition": DispositionContext,
                    "allocation": AllocationContext,
                    "concentration": ConcentrationContext,
                    "comparisons": ComparisonContext}
        for source, destination in DESTINATIONS.items():
            section, _, attribute = destination.partition(".")
            assert section in sections, (source, destination)
            names = {f.name for f in dataclasses.fields(sections[section])}
            names |= {name for name in dir(sections[section])
                      if not name.startswith("_")}
            assert attribute in names, (source, destination)
