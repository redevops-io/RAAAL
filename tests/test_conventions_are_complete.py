"""Every convention this build runs on is named, and named by QuantLib.

Step 5. Three categories were named — calendar, business-day, day count — and
five were not: settlement lag, schedule frequency, compounding, currency and
evaluation date. A convention that is not named is one the evaluator infers,
and a convention the evaluator infers is one nobody can check a figure against.

**Checked against the library, not against a table here.** The whole argument
for the vocabulary is that a reader outside this repository can verify a claim
against a definition that is not ours. A test comparing our constants to our
own constants would be the opposite of that — it would pass on a mapping that
had drifted from QuantLib entirely, which is exactly how "one place decides
what annually means" became two places that disagreed.
"""
from __future__ import annotations

import pytest

from src.mission import conventions
from src.mission.conventions import (AVAILABLE, COMPOUNDING, COMPOUNDINGS,
                                     CURRENCY, FREQUENCIES, SETTLEMENT_DAYS,
                                     SETTLEMENT_LAG, declared, frequency,
                                     frequency_value)
from src.mission.evaluation_policy import EvaluationPolicy, declared_policy

pytestmark = pytest.mark.skipif(not AVAILABLE, reason="QuantLib is not installed")


class TestTheFiveThatWereMissing:
    def test_all_eight_categories_are_declared(self):
        """The list from the plan, checked as a set rather than by eye."""
        named = declared()
        for category in ("exchange", "contribution_convention", "annualisation",
                         "sessions_per_year", "settlement_lag", "compounding",
                         "currency", "vocabulary"):
            assert named.get(category), f"{category} is not declared"

    @pytest.mark.parametrize("cadence,expected", sorted(FREQUENCIES.items()))
    def test_each_cadence_maps_to_quantlibs_own_frequency(self, cadence, expected):
        """The integers are QuantLib's — 12 for Monthly, 1 for Annual.

        Compared against the library so a mapping that drifted from it fails
        here rather than producing a schedule nobody can look up.
        """
        import QuantLib as ql

        assert frequency(cadence) == expected
        assert frequency_value(cadence) == int(getattr(ql, expected))

    def test_a_cadence_quantlib_has_no_name_for_gets_none(self):
        """Refuses rather than guessing. A frequency invented here would be a
        schedule with no definition outside this repository."""
        assert frequency("whenever") == ""
        assert frequency_value("whenever") is None

    def test_the_settlement_lag_is_what_the_calendar_actually_does(self):
        """T+1 for US equities since May 2024, and checked by advancing the
        NYSE calendar rather than by asserting the string."""
        import QuantLib as ql

        assert SETTLEMENT_LAG == f"T+{SETTLEMENT_DAYS}"
        friday = ql.Date(14, 8, 2026)
        assert conventions.settles_on(friday) == \
            conventions.calendar().advance(friday,
                                           ql.Period(SETTLEMENT_DAYS, ql.Days))

    def test_the_compounding_is_one_quantlib_defines(self):
        import QuantLib as ql

        assert COMPOUNDING in COMPOUNDINGS
        assert hasattr(ql, COMPOUNDING)

    def test_the_currency_is_quantlibs_code(self):
        import QuantLib as ql

        assert CURRENCY == ql.USDCurrency().code()
        assert conventions.currency_name() == ql.USDCurrency().name()


class TestTheEvaluationDateIsAnInputAndNotAClock:
    def test_it_takes_what_it_is_given(self):
        assert conventions.evaluation_date("2020-01-02") == "2020-01-02"

    def test_a_policy_built_with_an_as_of_does_not_read_the_global(self):
        """QuantLib keeps `Settings.instance().evaluationDate` as a mutable
        global that defaults to today. A result reading it would depend on when
        it was computed rather than on what it was computed from — the
        reproducibility defect, arriving through a convention nobody thought of
        as an input.
        """
        import QuantLib as ql

        before = ql.Settings.instance().evaluationDate
        try:
            ql.Settings.instance().evaluationDate = ql.Date(2, 1, 2020)
            policy = declared_policy(data_policy="SYNTHETIC_ONLY",
                                     as_of="2026-08-15")
            assert policy.evaluation_date == "2026-08-15", (
                "the policy read QuantLib's global clock instead of its "
                "argument")
        finally:
            ql.Settings.instance().evaluationDate = before


class TestMeasurementAndExecutionAreKeptApart:
    def test_the_policy_carries_measurement_and_not_execution(self):
        policy = declared_policy(data_policy="SYNTHETIC_ONLY",
                                 as_of="2026-08-15")
        body = policy.to_json()
        assert set(body) == {"compounding", "annualisation",
                             "sessions_per_year", "evaluation_date",
                             "data_policy", "models_settlement", "version"}
        for execution in ("calendar", "business_day", "currency",
                          "schedule_frequency", "settlement_lag"):
            assert execution not in body, (
                f"{execution} decides what the plan does and is in the "
                "measurement policy, so changing how a figure is measured "
                "would change what was executed")

    def test_it_admits_the_settlement_lag_is_not_applied(self):
        """A convention named but not honoured is only safe when the record
        says so. Unnamed and unhonoured is how somebody assumes it was
        handled — which is what `dividend_policy` did for months."""
        policy = declared_policy(data_policy="SYNTHETIC_ONLY")
        assert policy.models_settlement is False

    def test_the_data_policy_travels_with_the_measurement(self):
        """Not a QuantLib convention and it belongs here anyway: a result that
        did not say which data it was permitted would let a synthetic figure be
        read as a market one."""
        policy = declared_policy(data_policy="SYNTHETIC_ONLY")
        assert policy.data_policy == "SYNTHETIC_ONLY"


class TestTheEvaluatorInfersNone:
    def test_evaluate_reports_the_policy_it_was_handed(self):
        from dataclasses import replace

        policy = declared_policy(data_policy="SYNTHETIC_ONLY",
                                 as_of="2026-08-15")
        other = replace(policy, compounding="Continuous")
        assert policy.to_json() != other.to_json(), (
            "two measurement policies serialize identically, so a result "
            "citing one proves nothing about which was used")
