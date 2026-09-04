"""What a custom basket may hold — the tradable universe, named.

`data/instruments/pilot_universe.yaml` is the committed source of record: every
ticker the snapshot refresh prices, and therefore what a plan may name. The basket
composer offers exactly that set — the S&P 500 constituents plus the curated fund
list — so a basket names only tickers the pilot is built to price. A ticker the
day's snapshot transiently could not fetch is refused *by name* at read time (the
evaluator's own rule: everything is offered, silence is the only failure), so
offering the declared universe never produces a silent no-figure.

Names come from `security_names.json` (company/fund names for the tickers), a
convenience only — a ticker the map does not cover shows as the ticker itself.

Both files are read once and cached; a render never re-parses them.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

_UNIVERSE_YAML = Path("data/instruments/pilot_universe.yaml")
_NAMES_JSON = Path("data/instruments/security_names.json")


@lru_cache(maxsize=1)
def _names() -> dict:
    try:
        return json.loads(_NAMES_JSON.read_text())
    except Exception:
        return {}


@lru_cache(maxsize=1)
def evaluable() -> dict:
    """`{equities: [...], funds: [...]}` — the tradable universe, each item
    `{ticker, name, kind}`. Equities are the S&P 500 constituents; funds are the
    curated ETF/macro set. Sorted by ticker."""
    names = _names()
    try:
        declared = yaml.safe_load(_UNIVERSE_YAML.read_text()) or {}
    except Exception:
        declared = {}

    def rows(tickers, kind):
        return [
            {"ticker": t, "name": names.get(t, t), "kind": kind}
            for t in sorted(set(map(str, tickers or [])))
        ]

    return {
        "equities": rows(declared.get("equities"), "stock"),
        "funds": rows(declared.get("funds_and_macro"), "fund"),
    }
