"""Trading calendars as versioned artifacts.

`calendar: business_days` was a placeholder that quietly meant "Monday to Friday,
holidays included as flat days". That is not any real exchange, and it left ~3.4%
of observations as padded zeros after Erratum 02.

The fix is not a better string. It is the same move the project has made at every
layer: a choice that materially changes a published number becomes an identified,
hashable, versioned artifact that a result can cite.

    calendar: nyse@1

not

    calendar: business_days

Holidays are expressed as **rules**, not a date list, because that is how
exchanges actually publish them and because a rule can be checked against a
source. Each calendar declares the range it covers, and evaluating outside that
range is an error rather than a silent assumption.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Set

import pandas as pd

from ..runtime.base import (
    Exclusion,
    RuntimeArtifact,
    RuntimeAssumption,
    RuntimeLimitation,
)

CALENDAR_SPEC_VERSION = "0.1"

WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def easter_sunday(year: int) -> date:
    """Anonymous Gregorian computus.

    Needed because several exchanges observe Good Friday, which is Easter-relative
    and therefore cannot be expressed as a fixed or nth-weekday rule.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


@dataclass(frozen=True)
class HolidayRule:
    """One recurring closure.

    kind:
      ``fixed``          — month/day, optionally shifted off a weekend
      ``nth_weekday``    — e.g. the third Monday of January
      ``easter_offset``  — days relative to Easter Sunday (Good Friday is −2)
    """

    name: str
    kind: str
    month: Optional[int] = None
    day: Optional[int] = None
    weekday: Optional[int] = None        # 0 = Monday
    nth: Optional[int] = None            # negative counts from the end
    offset: Optional[int] = None         # easter_offset only
    observed: str = "nearest_weekday"    # nearest_weekday | none
    first_year: Optional[int] = None     # rule not in force before this
    last_year: Optional[int] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "month": self.month,
            "day": self.day, "weekday": self.weekday, "nth": self.nth,
            "offset": self.offset, "observed": self.observed,
            "first_year": self.first_year, "last_year": self.last_year,
        }

    def applies_in(self, year: int) -> bool:
        if self.first_year is not None and year < self.first_year:
            return False
        if self.last_year is not None and year > self.last_year:
            return False
        return True

    def resolve(self, year: int) -> Optional[date]:
        if not self.applies_in(year):
            return None

        if self.kind == "fixed":
            base = date(year, self.month, self.day)
            return _observe(base, self.observed)

        if self.kind == "nth_weekday":
            return _nth_weekday(year, self.month, self.weekday, self.nth)

        if self.kind == "easter_offset":
            return easter_sunday(year) + timedelta(days=self.offset)

        raise ValueError(f"unsupported holiday rule kind {self.kind!r}")


def _observe(day: date, policy: str) -> date:
    """Shift a weekend-falling holiday to the nearest weekday, as exchanges do."""
    if policy == "none":
        return day
    if policy != "nearest_weekday":
        raise ValueError(f"unsupported observance policy {policy!r}")
    if day.weekday() == 5:      # Saturday -> Friday
        return day - timedelta(days=1)
    if day.weekday() == 6:      # Sunday -> Monday
        return day + timedelta(days=1)
    return day


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    if nth > 0:
        first = date(year, month, 1)
        shift = (weekday - first.weekday()) % 7
        return first + timedelta(days=shift + 7 * (nth - 1))
    # Count back from the last day of the month.
    if month == 12:
        last = date(year, 12, 31)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    shift = (last.weekday() - weekday) % 7
    return last - timedelta(days=shift + 7 * (-nth - 1))


@dataclass(frozen=True)
class TradingCalendar(RuntimeArtifact):
    """A named, versioned trading calendar.

    Retrofitted onto `RuntimeArtifact` because it was the last runtime-like
    object outside that lifecycle. While it stayed outside, it could not declare
    semantic preconditions, its comparison dependencies could not be derived, and
    it kept its own hashing — which quietly weakened the claim that every
    execution condition shares one lifecycle.

    `calendar_id` is retained as an alias of `artifact_id`; every existing caller
    keeps working and the two can no longer disagree.
    """

    kind: ClassVar[str] = "calendar"

    #: A calendar is interpretable alone: sessions are sessions whether or not
    #: anything is evaluated over them. Nothing else is required for its
    #: declarations to have a truth value.
    undefined_without: ClassVar[Sequence[str]] = ()

    name: str
    version: int
    title: str
    weekmask: Sequence[int] = (0, 1, 2, 3, 4)     # weekdays that are sessions
    holidays: Sequence[HolidayRule] = ()
    periods_per_year: int = 252
    timezone: str = "America/New_York"
    covers_from: str = "2000-01-01"
    covers_to: str = "2035-12-31"
    source: str = ""
    spec_version: str = CALENDAR_SPEC_VERSION

    @property
    def calendar_id(self) -> str:
        """Alias of `artifact_id`, kept so existing callers are unaffected."""
        return self.artifact_id

    def declared_form(self) -> Dict[str, Any]:
        """Everything declared. `title` and `source` are prose and excluded from
        both hashes — they were already outside `canonical_form`."""
        return self.canonical_form()

    def comparable_form(self) -> Dict[str, Any]:
        """What could move a number.

        Coverage bounds are deliberately excluded: extending a horizon from 2035
        to 2040 adds sessions nobody has evaluated over yet and changes no
        result already produced. That is the coverage-extension case the two-hash
        split exists for — a new version, still comparable.

        `timezone` stays in: it decides which calendar day a bar belongs to.
        """
        declared = dict(self.canonical_form())
        for horizon in ("covers_from", "covers_to", "spec_version", "name",
                        "version"):
            declared.pop(horizon, None)
        return declared

    @property
    def assumptions(self) -> Sequence[RuntimeAssumption]:
        return (
            RuntimeAssumption(
                name="declared-sessions",
                statement=(f"Sessions are {len(self.weekmask)} weekdays minus "
                           f"{len(self.holidays)} holiday rule(s), annualized on "
                           f"{self.periods_per_year} periods."),
                realized_by="sessions",
                risk=("Annualizing on the wrong session count is how weekend "
                      "padding inflated published figures by 31%."),
            ),
            RuntimeAssumption(
                name="declared-coverage",
                statement=(f"Sessions are only produced between {self.covers_from} "
                           f"and {self.covers_to}."),
                realized_by="sessions",
            ),
        )

    @property
    def limitations(self) -> Sequence[RuntimeLimitation]:
        return (
            RuntimeLimitation(
                name="no-extrapolation",
                statement=(f"Dates outside {self.covers_from}..{self.covers_to} "
                           f"are refused rather than guessed. A calendar that "
                           f"extrapolates invents trading days."),
                reason=Exclusion.NOT_APPLICABLE,
            ),
            RuntimeLimitation(
                name="no-intraday-sessions",
                statement=("Half-days and early closes are not modelled; every "
                           "session counts the same."),
                reason=Exclusion.OUT_OF_SCOPE,
            ),
        )

    def canonical_form(self) -> Dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "name": self.name,
            "version": self.version,
            "weekmask": list(self.weekmask),
            "holidays": sorted((h.to_json() for h in self.holidays), key=lambda h: h["name"]),
            "periods_per_year": self.periods_per_year,
            "timezone": self.timezone,
            "covers_from": self.covers_from,
            "covers_to": self.covers_to,
        }

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.canonical_form(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_json(self) -> Dict[str, Any]:
        return {
            **self.canonical_form(),
            "calendar_id": self.calendar_id,
            "content_hash": self.content_hash,
            "title": self.title,
            "source": self.source,
        }

    # ---- session logic ----------------------------------------------------

    def holiday_dates(self, first_year: int, last_year: int) -> Set[date]:
        out: Set[date] = set()
        for year in range(first_year, last_year + 1):
            for rule in self.holidays:
                resolved = rule.resolve(year)
                if resolved is not None:
                    out.add(resolved)
        return out

    def sessions(self, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """The subset of `index` that are trading sessions on this calendar."""
        self._assert_covers(index)
        weekmask = set(self.weekmask)
        holidays = self.holiday_dates(index.min().year, index.max().year)
        keep = [
            ts for ts in index
            if ts.dayofweek in weekmask and ts.date() not in holidays
        ]
        return pd.DatetimeIndex(keep)

    def filter(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[self.sessions(frame.index)]

    def _assert_covers(self, index: pd.DatetimeIndex) -> None:
        """Refuse to apply a calendar outside the range it declares.

        Silently extrapolating holiday rules past their published horizon is how a
        calendar becomes wrong without anyone noticing.
        """
        start, end = pd.Timestamp(self.covers_from), pd.Timestamp(self.covers_to)
        if index.min() < start or index.max() > end:
            raise ValueError(
                f"{self.calendar_id} covers {self.covers_from}..{self.covers_to} but "
                f"the data spans {index.min().date()}..{index.max().date()}"
            )
