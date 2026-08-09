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

    return tuple(cases)


def by_tier(tier: str, path: Path = CASES) -> Sequence[Case]:
    return tuple(one for one in load(path) if one.tier == tier)
