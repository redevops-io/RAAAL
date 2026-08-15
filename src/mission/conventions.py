"""Named market conventions, taken from QuantLib rather than reasoned out here.

Every rule in this file already had a name in finance before this project
existed, and this project kept inventing prose for them.

The clearest case: a contribution on the 15th lands on the first session on or
after the 15th, unless that would cross into the next month, in which case it
takes the last session of the month it belongs to. That paragraph is
`ModifiedFollowing`, and it was written from scratch, tested from scratch, and
described in three different comments before anybody checked. Verified against
`ql.UnitedStates(ql.UnitedStates.NYSE)` for all twelve months of 2024 on both
the 15th and the 31st: identical.

The point is not that QuantLib computes it faster. It is that a reader who
knows the vocabulary can check a claim in this codebase against a definition
outside it — and a plan that says "ModifiedFollowing" says the same thing to
somebody who has never read this repository.

**Sessions still come from the data, not from the calendar.** QuantLib names
the convention and computes the intended date; whether a market bar exists on
that date is a fact about the snapshot. A calendar that says the market was
open on a day the data has no price for would produce a purchase at a price
nobody has, which is a worse error than a contribution landing a day late.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

#: Whether the vocabulary is available.
#:
#: Guarded rather than assumed. QuantLib is in `requirements-core.txt` and so
#: reaches the image, but a check that assumed it would be is the same
#: assumption that left Stanza in `requirements.txt` and the deterministic
#: reader absent from every deployment for months — a fact the pilot page
#: prints about itself to this day.
try:
    import QuantLib as ql

    AVAILABLE = True
except ImportError:                                          # pragma: no cover
    ql = None
    AVAILABLE = False


#: The exchange this build's sessions belong to.
#:
#: Named, because "trading day" is not a universal fact. The NYSE is shut on
#: days the Federal Reserve is open and vice versa, and a schedule built on the
#: wrong one is wrong on about nine days a year without ever looking wrong.
EXCHANGE = "NYSE"


def calendar():
    """The NYSE calendar, or None where the vocabulary is absent."""
    if not AVAILABLE:
        return None
    return ql.UnitedStates(ql.UnitedStates.NYSE)


@dataclass(frozen=True)
class BusinessDayConvention:
    """What to do when a nominated date is not a trading session."""

    name: str
    describes: str

    def quantlib(self):
        return getattr(ql, self.name) if AVAILABLE else None


#: The conventions this build can state. The names are QuantLib's, which are
#: the market's — `Following`, `ModifiedFollowing`, `Preceding`,
#: `ModifiedPreceding`, `Unadjusted`.
FOLLOWING = BusinessDayConvention(
    "Following", "roll to the next session, even into the next month")
MODIFIED_FOLLOWING = BusinessDayConvention(
    "ModifiedFollowing",
    "roll to the next session, unless that leaves the month, in which case "
    "take the last session of the month")
PRECEDING = BusinessDayConvention(
    "Preceding", "roll back to the previous session")
MODIFIED_PRECEDING = BusinessDayConvention(
    "ModifiedPreceding",
    "roll back to the previous session, unless that leaves the month")
UNADJUSTED = BusinessDayConvention(
    "Unadjusted", "leave the date alone even if the market is shut")

CONVENTIONS = (FOLLOWING, MODIFIED_FOLLOWING, PRECEDING, MODIFIED_PRECEDING,
               UNADJUSTED)

#: What a contribution on a nominated day of the month does.
#:
#: ModifiedFollowing, so a contribution stays in the month it was meant for.
#: `Following` would move a 31st into the next month and give one month two
#: contributions and another none, which is not what "monthly" means.
CONTRIBUTION_CONVENTION = MODIFIED_FOLLOWING


@dataclass(frozen=True)
class DayCount:
    """How a year is measured when a figure is annualised."""

    name: str
    quantlib_name: str
    per_year: Optional[float]
    describes: str

    def quantlib(self):
        if not AVAILABLE:
            return None
        if self.quantlib_name == "Business252":
            return ql.Business252(calendar())
        if self.quantlib_name == "Thirty360":
            return ql.Thirty360(ql.Thirty360.ISDA)
        if self.quantlib_name == "ActualActual":
            return ql.ActualActual(ql.ActualActual.ISDA)
        return getattr(ql, self.quantlib_name)()


BUSINESS_252 = DayCount(
    "Business/252", "Business252", 252.0,
    "sessions, counted. What a daily-return series is denominated in, and the "
    "only one of these that matches a figure compounded per trading day")
ACTUAL_365_FIXED = DayCount(
    "Actual/365 (Fixed)", "Actual365Fixed", 365.0,
    "calendar days over a fixed 365. The convention most equity performance "
    "is quoted on")
ACTUAL_360 = DayCount(
    "Actual/360", "Actual360", 360.0,
    "calendar days over 360. A money-market convention; it overstates an "
    "annual rate against Actual/365 by about 1.4%")
THIRTY_360 = DayCount(
    "30E/360 (ISDA)", "Thirty360", 360.0,
    "every month is 30 days and every year 360. A bond convention, and wrong "
    "for anything counted in sessions")
ACTUAL_ACTUAL = DayCount(
    "Actual/Actual (ISDA)", "ActualActual", None,
    "the real length of the real year, leap years included")

DAY_COUNTS = (BUSINESS_252, ACTUAL_365_FIXED, ACTUAL_360, THIRTY_360,
              ACTUAL_ACTUAL)

#: What this build annualises on, and why it is this one.
#:
#: The simulator compounds a *per-session* return series, so the exponent has
#: to be sessions per year — 252 — or the figure answers a different question
#: from the one the series asks. This was already the number; it simply had no
#: name, and a bare `periods_per_year: int = 252` is a convention nobody can
#: look up.
#:
#: It is not the convention most equity performance is *quoted* on, which is
#: Actual/365 Fixed. Reporting on one and computing on the other is how a
#: figure drifts a percent and nobody can say why, so this stays as it is and
#: says so.
ANNUALISATION = BUSINESS_252


def adjust(day: int, month: int, year: int,
           convention: BusinessDayConvention = CONTRIBUTION_CONVENTION):
    """The date a nominated day of the month resolves to, by name.

    Returns a `datetime.date`, or None where the vocabulary is absent — the
    caller then falls back to the session arithmetic it already has, which is
    equivalent and simply unnamed.
    """
    if not AVAILABLE:
        return None
    import datetime

    last = ql.Date.endOfMonth(ql.Date(1, month, year)).dayOfMonth()
    wanted = ql.Date(min(day, last), month, year)
    settled = calendar().adjust(wanted, convention.quantlib())
    return datetime.date(settled.year(), settled.month(), settled.dayOfMonth())


def period(text: str):
    """`1d`, `2w`, `1m`, `1y` as QuantLib says them.

    Here because the cadence vocabulary is heading the same way as the day
    rule: `monthly` is this build's word for `1m`, and a person who writes
    "every 2 weeks" is naming `2w`.
    """
    if not AVAILABLE:
        return None
    return ql.Period(text)


# --- the five that were missing ---------------------------------------------
#
# Calendar, business-day convention and day count were named here already. The
# rest of the vocabulary was not, which meant the evaluator either inferred them
# or did not model them, and neither is visible in a result. A convention the
# evaluator infers is a convention nobody can check a figure against.

#: The canonical cadence, as QuantLib's own frequency.
#:
#: `Once` is QuantLib's word for it too, which is worth stating: this mapping is
#: a translation, not a naming scheme invented here. The integers are QuantLib's
#: values — 12 for Monthly, 1 for Annual — and `test_conventions` compares them
#: against the library rather than against this table.
FREQUENCIES = {
    "once": "Once",
    "weekly": "Weekly",
    "biweekly": "Biweekly",
    "monthly": "Monthly",
    "quarterly": "Quarterly",
    "annual": "Annual",
}


def frequency(cadence: str) -> str:
    """QuantLib's name for a canonical cadence, or "" for one it has none for."""
    return FREQUENCIES.get(str(cadence).strip().lower(), "")


def frequency_value(cadence: str) -> Optional[int]:
    """The integer QuantLib assigns it — how many times a year."""
    name = frequency(cadence)
    if not name or not AVAILABLE:
        return None
    return int(getattr(ql, name))


#: How long after a trade the cash and the shares actually move.
#:
#: US equities settle T+1 since May 2024; they settled T+2 before. Stated rather
#: than assumed because it is the kind of fact that changes under a build
#: without the build noticing, and a run that does not say which it used cannot
#: be compared with one that does.
#:
#: This build simulates on session closes and does not model settlement, so the
#: value is declared and *not applied* — which the manifest and
#: `EvaluationPolicy.models_settlement` both say out loud. A convention named
#: but not honoured is only safe when the record says so; unnamed and unhonoured
#: is how somebody assumes it was handled.
SETTLEMENT_LAG = "T+1"
SETTLEMENT_DAYS = 1


def settles_on(trade_date):
    """When a trade struck on this session settles, by the NYSE calendar."""
    if not AVAILABLE:
        return None
    return calendar().advance(trade_date, ql.Period(SETTLEMENT_DAYS, ql.Days))


#: How a rate is compounded when one is quoted.
#:
#: QuantLib's four. `Compounded` is this build's, because the figures it reports
#: are period returns chained over sessions rather than a continuously
#: compounded rate — and reporting a continuously compounded number under a
#: discrete method is a different figure, not a rounding difference.
COMPOUNDINGS = ("Simple", "Compounded", "Continuous", "SimpleThenCompounded")
COMPOUNDING = "Compounded"

#: The currency amounts are in.
#:
#: One, and declared. The engine does no conversion, so a plan stating euros
#: and a snapshot priced in dollars would be arithmetic across two units with
#: nothing saying so — the manifest refuses a stated currency this build cannot
#: hold, and this is what it is compared against.
CURRENCY = "USD"


def currency_name() -> str:
    """QuantLib's own name for it, so the code is checkable outside this repo."""
    return ql.USDCurrency().name() if AVAILABLE else ""


def evaluation_date(as_of=None) -> str:
    """The date a valuation is made as of, ISO, or "" when unavailable.

    An argument rather than a constant. QuantLib keeps a global
    `Settings.instance().evaluationDate`, and reading that would make a result
    depend on when it was computed rather than on what it was computed from —
    which is the reproducibility defect this whole layer exists to prevent,
    hidden in a library default.
    """
    if as_of is not None:
        return str(as_of)
    if not AVAILABLE:
        return ""
    today = ql.Settings.instance().evaluationDate
    return f"{today.year():04d}-{today.month():02d}-{today.dayOfMonth():02d}"


def declared() -> dict:
    """What this build's time conventions are, for the deployment record.

    Reported rather than assumed, the same rule the parser identity follows. A
    figure annualised on 252 and a figure annualised on 365 are different
    claims, and a run that does not say which it used cannot be compared with
    one that does.
    """
    return {
        "exchange": EXCHANGE,
        "contribution_convention": CONTRIBUTION_CONVENTION.name,
        "annualisation": ANNUALISATION.name,
        "sessions_per_year": ANNUALISATION.per_year,
        "settlement_lag": SETTLEMENT_LAG,
        "compounding": COMPOUNDING,
        "currency": CURRENCY,
        "vocabulary": f"QuantLib {ql.__version__}" if AVAILABLE else "absent",
    }
