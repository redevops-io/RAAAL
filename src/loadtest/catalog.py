"""The strategy corpus, as committed data.

144 synthetic strategy flavors across 18 families. **Not a recommendation
database** — every row is marked `SYNTHETIC_TEST_FLAVOR`, and its value is
combinatorial coverage: different flows, account types, tax treatments, timing
rules and benchmark relationships, chosen to make the compiler reveal
assumptions it holds without stating them.

Committed as CSV rather than read from a spreadsheet. A corpus that lives in
someone's Downloads folder cannot be re-run from a clone, and a load result that
cannot be reproduced is an anecdote.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

CATALOG = Path(__file__).resolve().parents[2] / "data" / "catalog"


@dataclass(frozen=True)
class Strategy:
    """One catalog row, typed."""

    strategy_id: str
    family: str
    variant: str
    category: str
    base_description: str
    test_objective: str
    universe_or_assets: str
    rule_or_parameter: str
    cadence: str
    account_context: str
    default_benchmarks: str
    complexity: int
    source_name: str
    source_url: str

    @property
    def assets(self) -> List[str]:
        """Tickers, where the row names any.

        Several rows carry a phrase rather than a universe — "$24,500 2026 cap",
        "invest only above floor". Those are deliberately kept: a compiler that
        only ever sees clean ticker lists has not been tested on what users
        write.
        """
        raw = self.universe_or_assets.replace("+", " ").replace(",", " ")
        return [t for t in raw.split()
                if t.isupper() and t.isalpha() and 1 <= len(t) <= 5]

    @property
    def benchmarks(self) -> List[str]:
        return [b.strip() for b in self.default_benchmarks.split("|") if b.strip()]

    @property
    def accounts(self) -> List[str]:
        raw = self.account_context.replace("+", "|")
        return [a.strip() for a in raw.split("|") if a.strip()]


def load_strategies(path: Path = CATALOG / "strategy_catalog.csv") -> List[Strategy]:
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("include_in_load_test", "Yes").strip().lower() != "yes":
                continue
            rows.append(Strategy(
                strategy_id=row["strategy_id"], family=row["family"],
                variant=row["variant"], category=row["category"],
                base_description=row["base_description"],
                test_objective=row["test_objective"],
                universe_or_assets=row["universe_or_assets"],
                rule_or_parameter=row["rule_or_parameter"],
                cadence=row["cadence"], account_context=row["account_context"],
                default_benchmarks=row["default_benchmarks"],
                complexity=int(row["complexity_1_6"]),
                source_name=row["source_name"], source_url=row["source_url"]))
    return rows


@dataclass(frozen=True)
class LoadScenario:
    scenario_id: str
    virtual_users: int
    concurrent_requests: int
    variants_per_user: int
    purpose: str
    execution_backend: str
    target: str


def load_scenarios(path: Path = CATALOG / "load_scenarios.csv") -> List[LoadScenario]:
    with path.open() as handle:
        return [LoadScenario(
            scenario_id=r["scenario_id"],
            virtual_users=int(r["virtual_users"]),
            concurrent_requests=int(r["concurrent_requests"]),
            variants_per_user=int(r["variants_per_user"]),
            purpose=r["purpose"], execution_backend=r["execution_backend"],
            target=r["latency_or_exit_target"]) for r in csv.DictReader(handle)]


def families(strategies: Sequence[Strategy]) -> Dict[str, List[Strategy]]:
    out: Dict[str, List[Strategy]] = {}
    for s in strategies:
        out.setdefault(s.family, []).append(s)
    return out
