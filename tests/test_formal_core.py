"""The milestone sentence, checked against the tree it describes.

A milestone statement is the artifact most likely to outlive what it describes.
This one names twelve areas and a presentation property, and each is asserted
here — so the claim fails with the code rather than after somebody quotes it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FORMAL = ROOT / "formal" / "Quantify"
DOC = ROOT / "docs" / "FormalCore.md"

#: Claim in the milestone sentence -> a theorem or definition that carries it.
CLAIMS = {
    "ledger conservation": ("Ledger.lean", "cash_conservation"),
    "positions": ("Ledger.lean", "position_conservation"),
    "valuation": ("Ledger.lean", "portfolioValue"),
    "cadence": ("Cadence.lean", "contributions_are_one_per_period"),
    "trigger semantics": ("Triggers.lean", "crossing_and_persistent_differ_materially"),
    "evaluation windows": ("Window.lean", "warm_up_is_loaded_and_not_reported"),
    "ordering": ("Ordering.lean", "close_signal_fills_strictly_later"),
    "cash flows": ("Summary.lean", "net_is_the_flow_total"),
    "headline accounting": ("Summary.lean", "gain_is_value_less_flows"),
    "TWR": ("Returns/TimeWeighted.lean", "a_boundary_flow_does_not_change_the_return"),
    "MWR": ("Returns/MoneyWeighted.lean", "a_reported_rate_is_the_unique_root"),
    "drawdown": ("Returns/Drawdown.lean", "recovery_does_not_erase_the_maximum"),
}


class TestEveryClaimHasATheorem:
    @pytest.mark.parametrize("claim", sorted(CLAIMS))
    def test_the_named_result_exists(self, claim):
        module, name = CLAIMS[claim]
        path = FORMAL / module
        if not path.exists():
            pytest.skip(f"{module} is absent")
        assert re.search(rf"^(theorem|def) {re.escape(name)}\b",
                         path.read_text(), re.M), (
            f"the milestone claims {claim!r} and {module} has no {name}")

    def test_the_document_states_the_sentence(self):
        if not DOC.exists():
            pytest.skip("docs/FormalCore.md is absent")
        # Whitespace-normalised: the sentence wraps, so "ledger conservation"
        # appears as "ledger\n> conservation" in the source. Where a claim sits
        # relative to a line break is a formatting choice and must not be
        # something a test pins down — the same fix the consent notice needed.
        text = " ".join(DOC.read_text().replace(">", " ").split()).lower()
        for claim in CLAIMS:
            assert claim.lower() in text, claim

    def test_no_module_is_claimed_that_does_not_exist(self):
        for module, _ in CLAIMS.values():
            assert (FORMAL / module).exists(), module


class TestTheUndefinedClaimHolds:
    """"undefined returns are surfaced explicitly rather than rendered as
    zero" — the part of the sentence about presentation, which is where the
    defect actually was."""

    TEMPLATES = ROOT / "src" / "workspace" / "templates"

    @pytest.mark.parametrize("name", ["plan.html", "new.html",
                                      "_comparison.html"])
    def test_neither_return_basis_renders_undefined_as_zero(self, name):
        text = (self.TEMPLATES / name).read_text()
        for field in ("time_weighted_annualized", "money_weighted"):
            assert f"{field} or 0" not in text, f"{name}: {field}"

    def test_the_engine_still_reports_undefined_rather_than_a_number(self):
        """The other half. Templates that handled `None` would prove nothing if
        the engine had started substituting zero upstream."""
        import inspect

        from src.mission.simulate import MissionResult

        body = inspect.getsource(MissionResult.time_weighted_annualized.fget)
        assert "return None" in body


class TestTheCoreStaysLight:
    def test_no_core_module_imports_mathlib(self):
        offenders = [p.name for p in sorted(FORMAL.glob("*.lean"))
                     if re.search(r"^import Mathlib", p.read_text(), re.M)]
        assert not offenders, offenders

    def test_the_boundary_the_document_declares_is_real(self):
        """`coverage` and `modelling_scope` stay out of Lean. A milestone that
        described a boundary the tree did not have would be describing an
        intention."""
        for path in list(FORMAL.glob("*.lean")) + list(
                (FORMAL / "Returns").glob("*.lean")):
            text = path.read_text()
            assert "modelling_scope" not in text, path.name
            assert "def coverage" not in text, path.name
