"""Every key a view reads from a run result must be one the engine emits.

Two columns in the worksheet have been blank since they were written, for the
same reason and without anybody noticing:

    money_weighted      the engine emits `money_weighted_annualized`
    max_drawdown        the engine emits nothing of the sort

`dict.get` returns `None` for a key that was never there, and `None` renders as
an empty cell, so a misspelled or aspirational key looks exactly like a metric
that happened not to compute. Neither failed a test, because nothing compared
the names.

This compares the names. It is deliberately structural — read out of the
producer's own source rather than from a list somebody maintains, because a
list is a third place to get the spelling wrong.
"""
from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest


def _emitted() -> set:
    """Keys `MissionResult.to_json` actually puts in the dict."""
    from src.mission.simulate import MissionResult

    return set(re.findall(r'"([a-z_]+)":', inspect.getsource(MissionResult.to_json)))


def _read_by(path: Path, variable: str) -> set:
    """Keys fetched as `<variable>.get("...")` in a module."""
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "get":
            continue
        if not isinstance(func.value, ast.Name) or func.value.id != variable:
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            found.add(node.args[0].value)
    return found


#: Keys the worksheet reads that the engine does not emit, and why each is
#: still here. An entry is a defect with a reason, not a permission.
KNOWN_ABSENT = {
    "max_drawdown":
        "Mission computes no drawdown at all. The column has been blank since "
        "it was written. Removing the read would hide the gap; computing the "
        "metric is a product decision, so it stays declared and visible.",
}


class TestTheViewOnlyReadsKeysTheEngineEmits:
    VIEW = Path(__file__).resolve().parent.parent / "src" / "workspace" \
        / "worksheet_view.py"

    def test_no_undeclared_key_is_read(self):
        emitted = _emitted()
        read = _read_by(self.VIEW, "result")
        missing = sorted(read - emitted - set(KNOWN_ABSENT))
        assert not missing, (
            f"{missing} are read from a run result and never produced; "
            "`dict.get` makes that indistinguishable from a metric that did "
            "not compute")

    def test_the_declared_absences_are_still_absent(self):
        """A key that started being emitted should leave this list, or the
        list becomes a place where fixed things are still called broken."""
        emitted = _emitted()
        stale = sorted(k for k in KNOWN_ABSENT if k in emitted)
        assert not stale, (
            f"{stale} are emitted now; remove them from KNOWN_ABSENT")

    def test_and_they_are_still_read(self):
        """The other direction. An entry describing a read that no longer
        happens is a note about code that is gone."""
        read = _read_by(self.VIEW, "result")
        gone = sorted(k for k in KNOWN_ABSENT if k not in read)
        assert not gone, f"{gone} are no longer read; remove them"

    def test_the_repaired_key_is_the_one_the_engine_emits(self):
        """The half already fixed, pinned so it cannot drift back."""
        read = _read_by(self.VIEW, "result")
        assert "money_weighted_annualized" in read
        assert "money_weighted" not in read


class TestWhatMissionActuallyReports:
    """A metric nothing computes cannot be verified against anything.

    Recorded because it decides what the next formal slice can be: volatility
    has no implementation in this engine, so a proof of it would be a
    definition with nothing to conform to.
    """

    def test_mission_reports_neither_volatility_nor_drawdown(self):
        emitted = _emitted()
        assert not any("volatil" in key for key in emitted)
        assert not any("drawdown" in key for key in emitted)

    def test_it_does_report_both_return_bases(self):
        emitted = _emitted()
        assert "time_weighted_annualized" in emitted
        assert "money_weighted_annualized" in emitted
        assert "money_weighted_status" in emitted
