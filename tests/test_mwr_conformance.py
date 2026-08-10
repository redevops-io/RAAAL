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

    result = money_weighted_return(pd.Series([float(a) for a in amounts]),
                                   terminal_value=float(terminal),
                                   periods_per_year=periods_per_year)
    return result.rate


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

    def test_and_the_solver_now_refuses_it(self):
        """The non-conformance, closed.

        The solver used to return 0.4999999999 here — the root its opening
        bracket happened to straddle — and say nothing about the other. It now
        reports `NON_UNIQUE` and publishes no number.
        """
        import pandas as pd

        from src.mission.accounting import MWRStatus, money_weighted_return

        result = money_weighted_return(
            pd.Series([float(a) for a in self.AMOUNTS]),
            terminal_value=self.TERMINAL, periods_per_year=1)
        assert result.status is MWRStatus.NON_UNIQUE
        assert result.rate is None

    def test_the_gap_is_recorded_where_somebody_will_look(self):
        """A defect known only to a test is a defect the next person
        rediscovers."""
        from pathlib import Path

        doc = Path(__file__).resolve().parent.parent / "docs" / "MWR.md"
        if not doc.exists():
            pytest.skip("docs/MWR.md is absent")
        text = doc.read_text()
        assert "money_weighted_return" in text
        assert "Closed" in text, (
            "docs/MWR.md still records this as an open non-conformance")


class TestTheConformanceRecordIsFitToShow:
    def test_it_says_what_the_solver_returned_and_what_was_permitted(self):
        """Both halves. A record carrying only the verdict cannot be audited,
        and one carrying only the number is what the contract exists to
        prevent.

        `solver_returned` is `None` here now, and that is the conformance: the
        contract found two roots and the solver published neither. This
        assertion previously required a number, because when it was written the
        solver returned one.
        """
        from src.mission.mwr_contract import conformance_of

        record = conformance_of(*_flows([-100.0, 450.0, 0.0], 450.0),
                                _solver([-100.0, 450.0, 0.0], 450.0))
        assert record["verdict"] == "NON_UNIQUE"
        assert record["solver_returned"] is None
        assert record["contract_version"]
        assert len(record["roots_found"]) >= 2

    def test_a_reportable_record_carries_its_single_root(self):
        from src.mission.mwr_contract import conformance_of

        record = conformance_of(*_flows([100.0, 0.0], 110.0),
                                _solver([100.0, 0.0], 110.0))
        assert record["verdict"] == "RATE"
        assert len(record["roots_found"]) == 1


class TestEveryOutcomeIsReachable:
    """`None` let several meanings collapse into one. A result type that fixed
    that and then carried a state nothing produces would have moved the problem
    rather than solved it.

    So each status is reached here by a named series, and `INDETERMINATE` in
    particular is kept only because the implementation genuinely needs it.
    """

    import pandas as _pd

    CASES = {
        "rate": ([100.0, 0.0], 110.0, "contributions, then a terminal value"),
        "non_unique": ([-100.0, 450.0, 0.0], 450.0,
                       "a withdrawal makes the coefficients read - + -"),
        "insufficient_cash_flows": ([0.0, 0.0], 100.0, "no money ever moved"),
        "no_solution": ([-100.0, -50.0], 10.0,
                        "no sign change, so Descartes gives no positive root"),
        "indeterminate": ([-100.0, 400.0, 0.0], 400.0,
                          "a root that touches zero without crossing"),
    }

    @pytest.mark.parametrize("status", sorted(CASES))
    def test_each_status_has_a_series_that_produces_it(self, status):
        import pandas as pd

        from src.mission.accounting import money_weighted_return

        amounts, terminal, _why = self.CASES[status]
        result = money_weighted_return(pd.Series(amounts),
                                       terminal_value=terminal,
                                       periods_per_year=1)
        assert result.status.value == status

    def test_the_cases_cover_the_whole_enum(self):
        """A status added later with no series would be a state nobody can
        reach, which is how an outcome becomes decoration."""
        from src.mission.accounting import MWRStatus

        assert {s.value for s in MWRStatus} == set(self.CASES)

    def test_indeterminate_is_earned_not_assumed(self):
        """The tangent case, spelled out: `-100(g-2)²` has a root at `g = 2`
        and the present value never changes sign, so a crossing scan finds
        nothing while Descartes permits two roots. Reporting `NO_SOLUTION`
        would turn "could not establish" into "established"."""
        import pandas as pd

        from src.mission.accounting import MWRStatus, money_weighted_return

        result = money_weighted_return(pd.Series([-100.0, 400.0, 0.0]),
                                       terminal_value=400.0,
                                       periods_per_year=1)
        assert result.status is MWRStatus.INDETERMINATE
        assert result.rate is None

        def npv(rate):
            g = 1.0 + rate
            return -100 * g ** 2 + 400 * g - 400

        assert abs(npv(1.0)) < 1e-9, "there is a root"
        assert npv(0.9) < 0 and npv(1.1) < 0, "and it never crosses"


class TestTheInvariantIsEnforced:
    def test_a_rate_status_must_carry_a_number(self):
        from src.mission.accounting import MWRResult, MWRStatus

        with pytest.raises(ValueError):
            MWRResult(MWRStatus.RATE)

    @pytest.mark.parametrize("status", ["no_solution", "non_unique",
                                        "insufficient_cash_flows",
                                        "indeterminate"])
    def test_and_every_other_status_must_not(self, status):
        """The shape the old return type allowed: a reason and a number
        together, with nothing saying which to believe."""
        from src.mission.accounting import MWRResult, MWRStatus

        with pytest.raises(ValueError):
            MWRResult(MWRStatus(status), 0.1)
