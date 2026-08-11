"""Every field a first-party view reads from a run result must be accounted for.

The same failure has now appeared four times:

    consumer names a field  →  producer never emits it  →  `dict.get` turns a
    contract violation into an empty cell

`money_weighted`, `max_drawdown` and `contributed` were blank columns in the
worksheet for months. None failed anything, because nothing compared the names,
and a missing key is indistinguishable from a metric that did not compute.

That is the declared-but-not-executable defect this project has been
eliminating elsewhere, applied to result schemas. So every read resolves to
exactly one of:

    PRODUCED              the producer emits this key
    EXPLICITLY_ABSENT     it does not, and the consumer says why
    DERIVED_BY_CONSUMER   the consumer computes it, and the site exists

Anything else fails. Both non-produced categories require evidence: a reason
for the first, a real derivation site for the second, so neither becomes a
place to park a stale field.

Read out of the producer's and consumers' own source. A central list would be a
third place to get a spelling wrong, which is the defect one level up.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

PRODUCED = "PRODUCED"
EXPLICITLY_ABSENT = "EXPLICITLY_ABSENT"
DERIVED_BY_CONSUMER = "DERIVED_BY_CONSUMER"

#: Modules that read fields off a `MissionResult.to_json` dict. Derived below
#: rather than listed: any module that reads `result.get("...")` is a consumer,
#: whether or not somebody remembered to add it here.
CONSUMER_ROOTS = ("workspace", "web")


def _produced() -> set:
    """Keys `MissionResult.to_json` puts in the dict."""
    from src.mission.simulate import MissionResult

    return set(re.findall(r'"([a-z_]+)":',
                          inspect.getsource(MissionResult.to_json)))


def _reads(path: Path) -> set:
    """Keys fetched as `result.get("...")` or `result["..."]`."""
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "result"
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            found.add(node.args[0].value)
        if (isinstance(node, ast.Compare) and len(node.comparators) == 1
                and isinstance(node.left, ast.Constant)
                and isinstance(node.left.value, str)
                and isinstance(node.comparators[0], ast.Name)
                and node.comparators[0].id == "result"
                and any(isinstance(op, ast.In) for op in node.ops)):
            found.add(node.left.value)
    return found


def consumers() -> dict:
    """Every first-party module that reads a result field, and what it reads."""
    found = {}
    for root in CONSUMER_ROOTS:
        for path in (SRC / root).rglob("*.py"):
            keys = _reads(path)
            if keys:
                found[path] = keys
    return found


def _notes(path: Path) -> dict:
    module = "src." + str(path.relative_to(SRC.parent)).removesuffix(".py") \
        .replace("/", ".").removeprefix("src.")
    module = "src." + module.removeprefix("src.")
    return getattr(importlib.import_module(module), "RESULT_FIELD_NOTES", {})


class TestEveryReadIsClassified:
    def test_at_least_one_consumer_was_found(self):
        """A scan that found nothing would pass every check below while
        checking nothing — the empty-room failure."""
        assert consumers(), "no result consumers found; the scan is broken"

    def test_no_field_is_read_without_being_accounted_for(self):
        produced = _produced()
        unaccounted = {}
        for path, keys in consumers().items():
            notes = _notes(path)
            missing = sorted(k for k in keys
                             if k not in produced and k not in notes)
            if missing:
                unaccounted[path.name] = missing
        assert not unaccounted, (
            f"{unaccounted} are read from a run result, not produced, and not "
            "declared. `dict.get` makes that indistinguishable from a metric "
            "that did not compute")

    def test_every_note_uses_a_known_classification(self):
        for path, _keys in consumers().items():
            for field, (kind, _why) in _notes(path).items():
                assert kind in (EXPLICITLY_ABSENT, DERIVED_BY_CONSUMER), (
                    f"{path.name}:{field} is classified {kind!r}")

    def test_an_absent_field_carries_a_reason(self):
        """A category with no evidence requirement is a place to park a stale
        field."""
        for path, _keys in consumers().items():
            for field, (kind, why) in _notes(path).items():
                if kind == EXPLICITLY_ABSENT:
                    assert len(why) > 40, f"{path.name}:{field} has no reason"

    def test_a_derived_field_names_a_site_that_exists(self):
        """The other evidence requirement. A derivation nobody wrote is the
        same defect wearing a different label."""
        for path, _keys in consumers().items():
            module = importlib.import_module(
                "src." + str(path.relative_to(SRC)).removesuffix(".py")
                .replace("/", "."))
            for field, (kind, site) in _notes(path).items():
                if kind == DERIVED_BY_CONSUMER:
                    assert hasattr(module, site), (
                        f"{path.name}:{field} claims to derive via {site!r}, "
                        "which does not exist")


class TestTheNotesStayHonest:
    def test_a_declared_absence_is_still_absent(self):
        """A field that started being emitted should leave the notes, or they
        become a place where fixed things are still called broken."""
        produced = _produced()
        for path, _keys in consumers().items():
            stale = sorted(f for f, (kind, _) in _notes(path).items()
                           if kind == EXPLICITLY_ABSENT and f in produced)
            assert not stale, f"{path.name}: {stale} are emitted now"

    def test_and_is_still_read(self):
        """A note describing a read that no longer happens documents code that
        is gone."""
        for path, keys in consumers().items():
            gone = sorted(f for f in _notes(path) if f not in keys)
            assert not gone, f"{path.name}: {gone} are no longer read"


class TestTheKnownRepairsHold:
    """The three columns this gate was built from, pinned so they cannot
    silently revert."""

    def test_the_worksheet_reads_the_names_the_engine_emits(self):
        view = SRC / "workspace" / "worksheet_view.py"
        keys = _reads(view)
        assert "money_weighted_annualized" in keys
        assert "contributed" in keys
        assert "money_weighted" not in keys

    def test_mission_still_reports_no_drawdown(self):
        """`max_drawdown` is declared absent, and the declaration is only true
        while Mission does not compute one."""
        assert "max_drawdown" not in _produced()
