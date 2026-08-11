"""On-disk registry for trading calendars.

Same shape as every other registry in the project: one YAML per version under
`calendars/`, versions coexisting. Adding a holiday to an exchange's schedule
produces `nyse@2`; results citing `nyse@1` remain interpretable under the
calendar they were actually measured with.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

from .calendar import HolidayRule, TradingCalendar

DEFAULT_ROOT = Path("calendars")
_FILENAME = re.compile(r"^(?P<name>[a-z0-9][a-z0-9_-]*)@(?P<version>\d+)\.ya?ml$")


class CalendarRegistry:
    def __init__(self, root: Path | str = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def _iter_files(self) -> Iterator[Path]:
        if not self.root.exists():
            return
        for path in sorted(self.root.iterdir()):
            if _FILENAME.match(path.name):
                yield path

    def _load_file(self, path: Path) -> TradingCalendar:
        payload = yaml.safe_load(path.read_text()) or {}
        calendar = TradingCalendar(
            name=payload["name"],
            version=int(payload["version"]),
            title=payload.get("title", payload["name"]),
            weekmask=tuple(payload.get("weekmask", (0, 1, 2, 3, 4))),
            holidays=tuple(HolidayRule(**h) for h in payload.get("holidays", [])),
            periods_per_year=int(payload.get("periods_per_year", 252)),
            timezone=payload.get("timezone", "America/New_York"),
            covers_from=str(payload.get("covers_from", "2000-01-01")),
            covers_to=str(payload.get("covers_to", "2035-12-31")),
            source=payload.get("source", ""),
        )
        match = _FILENAME.match(path.name)
        assert match
        if match.group("name") != calendar.name or int(match.group("version")) != calendar.version:
            raise ValueError(
                f"{path.name} declares {calendar.calendar_id} — filename and content "
                "disagree about identity"
            )
        return calendar

    def load_all(self) -> List[TradingCalendar]:
        return [self._load_file(p) for p in self._iter_files()]

    def versions(self, name: str) -> List[TradingCalendar]:
        return sorted((c for c in self.load_all() if c.name == name), key=lambda c: c.version)

    def get(self, name: str, version: Optional[int] = None) -> TradingCalendar:
        candidates = self.versions(name)
        if not candidates:
            raise KeyError(f"no trading calendar named {name!r} in {self.root}")
        if version is None:
            return candidates[-1]
        for c in candidates:
            if c.version == version:
                return c
        raise KeyError(
            f"{name}@{version} not found; available: {[c.version for c in candidates]}"
        )

    def resolve(self, reference: str) -> TradingCalendar:
        ref = reference.removeprefix("calendar/")
        if "@" in ref:
            name, _, version = ref.partition("@")
            return self.get(name, int(version))
        return self.get(ref)

    def names(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        for c in self.load_all():
            out.setdefault(c.name, []).append(c.version)
        return {k: sorted(v) for k, v in sorted(out.items())}
