"""Reads `cases.json`, and refuses to hand back a corpus that is not one.

Validation here rather than in the tests, because a corpus that silently loses
half its cases still produces a green run — the failure mode is a test file that
iterates over three cases believing it iterated over two hundred.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

CASES = Path(__file__).resolve().parent / "cases.json"
SCHEMA = "quantify-parser-corpus@1"

TIERS = ("normalization", "dependency", "semantics")
ORIGINS = ("constructed", "falsification", "observed", "web")


@dataclass(frozen=True)
class Case:
    id: str
    tier: str
    property: str
    text: str
    language: str
    asserts: Mapping[str, Any]
    origin: str
    note: str = ""

    @property
    def is_falsification(self) -> bool:
        return self.origin == "falsification"


class CorruptCorpus(ValueError):
    """The file on disk is not a usable corpus."""


def load(path: Path = CASES) -> Sequence[Case]:
    document = json.loads(path.read_text())
    if document.get("schema") != SCHEMA:
        raise CorruptCorpus(
            f"schema is {document.get('schema')!r}, expected {SCHEMA!r} — a "
            "corpus read under the wrong schema is measuring something else")

    cases = [Case(**one) for one in document["cases"]]

    if len(cases) != document["count"]:
        raise CorruptCorpus(
            f"the file says {document['count']} cases and carries {len(cases)}")

    seen = set()
    for one in cases:
        if one.id in seen:
            raise CorruptCorpus(
                f"duplicate id {one.id!r}; two cases with one name cannot be "
                "told apart in a result table")
        seen.add(one.id)
        if one.tier not in TIERS:
            raise CorruptCorpus(f"{one.id}: unknown tier {one.tier!r}")
        if one.origin not in ORIGINS:
            raise CorruptCorpus(f"{one.id}: unknown origin {one.origin!r}")
        if not one.text.strip():
            raise CorruptCorpus(f"{one.id}: empty text")
        if not one.asserts:
            raise CorruptCorpus(
                f"{one.id}: asserts nothing. A case that claims nothing passes "
                "forever and reads like coverage")

    _preflight(cases)
    return tuple(cases)


def _preflight(cases: Sequence[Case]) -> None:
    """Expectations must be representable by the contract before they may judge.

    The same class was caught at three layers in three passes: wrong field
    names, wrong field *value* vocabularies, wrong numeric and unit coercion.
    Each time the corpus had quietly become a second schema, and each time it
    was invisible until a second witness answered the same question and
    disagreed with it.

    So a fixture whose expected field or expected value the contract cannot
    hold is rejected at load. Two escapes, both explicit and both meaning
    something:

        schema_gap             the reading is right and the contract has no
                               value for it — that is the finding, not an error
        INTERMEDIATE_FIELDS    the case asserts semantics this pipeline
                               computes outside the contract boundary

    Anything else is a corpus inventing vocabulary, which is what this refuses.
    """
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.semantics import INTERMEDIATE_FIELDS

    dimensions = {d.name: d for d in QUANTIFY_SCHEMA.dimensions}
    problems = []

    for one in cases:
        if one.tier != "semantics":
            continue
        field = one.asserts.get("field")
        if field is None or field in INTERMEDIATE_FIELDS:
            continue
        if field not in dimensions:
            problems.append(
                f"{one.id}: asserts field {field!r}, which is not a contract "
                f"dimension. Use a contract name, or mark the case "
                f"INTERMEDIATE_SEMANTIC if it is outside the boundary")
            continue
        if "schema_gap" in one.asserts:
            continue
        value, allowed = one.asserts.get("value"), dimensions[field].values
        if allowed and value is not None and str(value) not in allowed:
            problems.append(
                f"{one.id}: asserts {field}={value!r}, which is not one of "
                f"{sorted(allowed)}. If the reading is right and the contract "
                f"cannot hold it, mark it `schema_gap` — do not rename it to "
                f"the nearest allowed value")

    if problems:
        raise CorruptCorpus(
            "corpus expectations are not representable by the contract:\n  "
            + "\n  ".join(problems))


def by_tier(tier: str, path: Path = CASES) -> Sequence[Case]:
    return tuple(one for one in load(path) if one.tier == tier)
