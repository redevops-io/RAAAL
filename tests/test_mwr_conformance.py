"""Whether the production solver obeys the frozen MWR contract.

    docs/MWR.md                      what a money-weighted return means
    Quantify/Returns/MoneyWeighted   the reporting contract, proven in Lean
    this file                        whether the engine's solver conforms

The contract's whole purpose is that the solver may search but does not get to
define what a valid answer is. So the interesting test is not that the solver
returns a number for a well-behaved series — it does — but what it does with a
series the contract says has no reportable answer.

**One non-conformance is recorded here rather than hidden.** It was found by
running the solver, not by reading it, and it is named so the gap is
attributable instead of absorbed.
"""
from __future__ import annotations

import pytest


def _flows(amounts, terminal, sessions=None, horizon=None):
    sessions = sessions if sessions is not None else list(range(len(amounts)))
    horizon = horizon if horizon is not None else float(len(amounts) - 1)
    return amounts, [float(s) for s in sessions], float(horizon), float(terminal)


def _solver(amounts, terminal, periods_per_year=1):
    import pandas as pd

    from src.mission.accounting import money_weighted_return

    return money_weighted_return(pd.Series([float(a) for a in amounts]),
                                 terminal_value=float(terminal),
                                 periods_per_year=periods_per_year)


class TestTheCertificationNamesWhatItChecked:
    def test_it_carries_the_contract_version_and_the_solver(self):
        """"The solver was verified" is not a fact unless it says verified
        against what. A certification with no contract version outlives the
        contract it was granted under."""
        from src.mission.mwr_contract import CONTRACT_VERSION, SOLVER, certify

        certification = certify(*_flows([100.0, 0.0], 110.0), 0.1)
        assert certification.contract_version == CONTRACT_VERSION
        assert certification.solver == SOLVER
        assert certification.tolerance > 0

    def test_the_tolerance_lives_here_and_not_in_the_definition(self):
        """A financial definition carrying an epsilon is a definition that
        changes when somebody tunes a solver."""
        from pathlib import Path

        doc = Path(__file__).resolve().parent.parent / "docs" / "MWR.md"
        if not doc.exists():
            pytest.skip("docs/MWR.md is absent")
        text = doc.read_text().lower()
        for word in ("tolerance", "epsilon", "1e-", "converge"):
            assert word not in text, (
                f"docs/MWR.md mentions {word!r}; numerical concerns belong to "
                "the implementation boundary, not to what MWR means")

    def test_uniqueness_is_never_claimed_from_sampling(self):
        """Sampling can prove a second root exists and cannot prove one does
        not. The verdict name is deliberately clumsy so nobody quotes it as
        `UNIQUE`."""
        from src.mission import mwr_contract

        assert not hasattr(mwr_contract, "UNIQUE")
        assert mwr_contract.NO_EVIDENCE_OF_NON_UNIQUENESS \
            == "NO_EVIDENCE_OF_NON_UNIQUENESS"


class TestAWellBehavedSeriesIsReportable:
    def test_contributions_then_a_terminal_value(self):
        """The shape the solver's Descartes argument actually covers.

        Contributions are positive in the engine's convention and the terminal
        value is subtracted, so the coefficient sequence is `+ … -`: one sign
        change, one positive root, and the argument holds."""
        from src.mission.mwr_contract import RATE, certify

        amounts, sessions, horizon, terminal = _flows([100.0, 0.0], 110.0)
        returned = _solver([100.0, 0.0], 110.0)
        certification = certify(amounts, sessions, horizon, terminal, returned)

        assert certification.verdict == RATE
        assert certification.reportable
        assert len(certification.roots_found) == 1

    def test_and_the_rate_is_the_one_the_equation_gives(self):
        from src.mission.mwr_contract import certify, npv

        amounts, sessions, horizon, terminal = _flows([100.0, 0.0], 110.0)
        returned = _solver([100.0, 0.0], 110.0)
        certification = certify(amounts, sessions, horizon, terminal, returned)
        assert abs(npv(amounts, sessions, horizon, terminal,
                       certification.roots_found[0])) < 1e-6


class TestNoSolutionIsNotTurnedIntoANumber:
    def test_a_series_with_nothing_to_recover(self):
        from src.mission.mwr_contract import certify

        amounts, sessions, horizon, terminal = _flows([100.0, 50.0], 0.0)
        returned = _solver([100.0, 50.0], 0.0)
        certification = certify(amounts, sessions, horizon, terminal, returned)

        assert not certification.reportable
        assert returned is None, "the solver invented a rate for a series with "\
                                 "no terminal value"


class TestTheNonUniqueCase:
    """The finding, and it came from running the solver rather than reading it.

    A withdrawal of 100 at session 0, a contribution of 450 at session 1, and
    450 remaining at session 2. The coefficient sequence is `- + -`: two sign
    changes, and Descartes allows two positive roots.

        f(g) = -100g² + 450g - 450     roots g = 1.5 and g = 3.0
                                       rates 50% and 200%

    Both satisfy the series. The contract says report neither.

    The solver's docstring justifies uniqueness by Descartes' rule — one sign
    change, one positive root — which is true for contributions plus a terminal
    value and false as soon as a withdrawal appears. Nothing checks that the
    series has the shape the argument assumes, so it is a precondition the code
    relies on and never tests.
    """

    AMOUNTS = [-100.0, 450.0, 0.0]
    TERMINAL = 450.0

    def test_two_rates_satisfy_the_series(self):
        from src.mission.mwr_contract import npv

        amounts, sessions, horizon, terminal = _flows(self.AMOUNTS,
                                                      self.TERMINAL)
        for rate in (0.5, 2.0):
            assert abs(npv(amounts, sessions, horizon, terminal, rate)) < 1e-6

    def test_the_contract_refuses_to_report_either(self):
        from src.mission.mwr_contract import NON_UNIQUE, certify

        amounts, sessions, horizon, terminal = _flows(self.AMOUNTS,
                                                      self.TERMINAL)
        returned = _solver(self.AMOUNTS, self.TERMINAL)
        certification = certify(amounts, sessions, horizon, terminal, returned)

        assert certification.verdict == NON_UNIQUE
        assert not certification.reportable
        assert len(certification.roots_found) >= 2

    def test_and_the_solver_returns_one_of_them_anyway(self):
        """The non-conformance, asserted so it is a recorded fact rather than
        a remark. The solver returns the root its bracket happens to contain
        and says nothing about the other.

        Not fixed here. Changing the engine's return type is a product decision
        — every caller of `money_weighted_return` currently reads `Optional
        [float]` and would need to learn a fourth outcome — and doing it inside
        a verification slice would mean the lane that found the defect also
        chose the remedy.
        """
        returned = _solver(self.AMOUNTS, self.TERMINAL)
        assert returned is not None
        assert abs(returned - 0.5) < 1e-6, (
            "the solver no longer returns the lower root; if it now refuses "
            "this series, this test and the record in docs/MWR.md should say "
            "so")

    def test_the_gap_is_recorded_where_somebody_will_look(self):
        """A defect known only to a test is a defect the next person
        rediscovers."""
        from pathlib import Path

        doc = Path(__file__).resolve().parent.parent / "docs" / "MWR.md"
        if not doc.exists():
            pytest.skip("docs/MWR.md is absent")
        text = doc.read_text()
        assert "Known non-conformance" in text
        assert "money_weighted_return" in text


class TestTheConformanceRecordIsFitToShow:
    def test_it_says_what_the_solver_returned_and_what_was_permitted(self):
        """Both halves. A record carrying only the verdict cannot be audited,
        and one carrying only the number is what the contract exists to
        prevent."""
        from src.mission.mwr_contract import conformance_of

        record = conformance_of(*_flows([-100.0, 450.0, 0.0], 450.0),
                                _solver([-100.0, 450.0, 0.0], 450.0))
        assert record["verdict"] == "NON_UNIQUE"
        assert record["solver_returned"] is not None
        assert record["contract_version"]
        assert len(record["roots_found"]) >= 2

    def test_a_reportable_record_carries_its_single_root(self):
        from src.mission.mwr_contract import conformance_of

        record = conformance_of(*_flows([100.0, 0.0], 110.0),
                                _solver([100.0, 0.0], 110.0))
        assert record["verdict"] == "RATE"
        assert len(record["roots_found"]) == 1
