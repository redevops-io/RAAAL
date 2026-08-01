"""EvolutionBench: one plan, lived with over time.

Every other benchmark here measures throughput on independent work. This one
measures what happens to a *single* plan as the world moves underneath it — the
workload nobody fakes, because it needs versioned artifacts, pinned runtimes,
historical replay and comparability all at once.

The spine is a history, not a loop:

    create -> contribution change -> account change -> RSU event
           -> tax runtime version -> market-data restatement
           -> compiler upgrade -> counterfactual -> replay at each point

At every checkpoint three things are recorded and kept apart:

    replay          the Mission as it was saved, under its pinned versions
    reinterpret     the same words compiled by today's compiler
    migrate         a proposal, explained and never performed

The invariant: **a historical replay hash must not move when the compiler
changes.** A reinterpretation may, and when it does the difference is typed.

Deterministic and single-user on purpose. A bottleneck in artifact versioning or
historical replay is invisible under concurrency, and a benchmark that starts
with a thousand users measures the queue instead.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..mission.compiler import compile_scenario
from ..mission.evolution import (
    COMPILER_VERSION,
    SemanticDiff,
    as_compiled_by,
    diff_stored_against,
    propose_migration,
)

BENCHMARK_RULE = "benchmark-policy/public-default@1"


@dataclass(frozen=True)
class Edit:
    """One thing that happened to a user, in their own words."""

    at: str
    what: str
    text: str
    #: Runtime versions in force when this edit was made. Pinned into the
    #: revision, so a replay years later resolves the same semantics rather than
    #: today's.
    runtimes: Mapping[str, str] = field(default_factory=dict)


#: A five-year history. Each entry is a real life event from the catalog's
#: families, worded the way someone would say it.
SPINE: Sequence[Edit] = (
    Edit("2021-01-04", "opens a plan",
         "I put $500 into VTI every month in my taxable brokerage account, on "
         "the first trading day of the period, reinvesting the dividends, and "
         "I never sell.",
         {"tax": "tax/none@1", "calendar": "calendar/nyse@1",
          "market_data": "prices-2021-01-04", "compiler": "1"}),
    Edit("2021-07-01", "raises the contribution",
         "I put $900 into VTI every month in my taxable brokerage account, on "
         "the first trading day of the period, reinvesting the dividends, and "
         "I never sell.",
         {"tax": "tax/none@1", "calendar": "calendar/nyse@1",
          "market_data": "prices-2021-07-01", "compiler": "1"}),
    Edit("2022-03-15", "adds a second holding",
         "I put $900 into VTI and BND every month in my taxable brokerage "
         "account, on the first trading day of the period, reinvesting the "
         "dividends, and I never sell.",
         {"tax": "tax/none@1", "calendar": "calendar/nyse@1",
          "market_data": "prices-2022-03-15", "compiler": "1"}),
    Edit("2022-11-01", "moves to a retirement account",
         "I put $900 into VTI and BND every month in my Roth IRA, on the first "
         "trading day of the period, reinvesting the dividends, and I never "
         "sell.",
         {"tax": "tax/roth@1", "calendar": "calendar/nyse@1",
          "market_data": "prices-2022-11-01", "compiler": "1"}),
    Edit("2023-04-01", "starts holding dividends as cash",
         "I put $900 into VTI and BND every month in my Roth IRA, on the first "
         "trading day of the period, holding the dividends as cash, and I "
         "never sell.",
         {"tax": "tax/roth@1", "calendar": "calendar/nyse@1",
          "market_data": "prices-2023-04-01", "compiler": "1"}),
    Edit("2024-02-01", "adds a dip-buying rule",
         "I put $900 into VTI and BND every month in my Roth IRA, on the first "
         "trading day of the period, holding the dividends as cash, and I "
         "never sell. Whenever SPY is below its exponential 200 day moving "
         "average I buy more with additional cash.",
         {"tax": "tax/roth@2", "calendar": "calendar/nyse@1",
          "market_data": "prices-2024-02-01", "compiler": "1"}),
    Edit("2025-06-02", "restated market data",
         "I put $900 into VTI and BND every month in my Roth IRA, on the first "
         "trading day of the period, holding the dividends as cash, and I "
         "never sell. Whenever SPY is below its exponential 200 day moving "
         "average I buy more with additional cash.",
         {"tax": "tax/roth@2", "calendar": "calendar/nyse@2",
          "market_data": "prices-2025-06-02", "compiler": "1"}),
)


@dataclass
class Revision:
    """One saved state of the plan, as it was compiled at the time."""

    at: str
    what: str
    text: str
    runtimes: Mapping[str, str]
    compiler: str
    stored: Mapping[str, Any]
    unresolved: Sequence[str]
    compile_us: int

    @property
    def rule_hash(self) -> str:
        return self.stored["rule_hash"]

    @property
    def content_hash(self) -> str:
        return self.stored["content_hash"]


@dataclass
class Checkpoint:
    at: str
    what: str
    revisions: int
    replay_us: int
    replay_hash: str
    reinterpret_us: int
    reinterpret_hash: str
    diff: Optional[SemanticDiff]
    migration_recommended: bool
    pinned_runtimes: Mapping[str, str]

    @property
    def reinterpretation_differs(self) -> bool:
        return self.replay_hash != self.reinterpret_hash

    def as_row(self) -> Dict[str, Any]:
        return {
            "at": self.at, "what": self.what, "revisions": self.revisions,
            "replay_us": self.replay_us, "reinterpret_us": self.reinterpret_us,
            "replay_hash": self.replay_hash[:16],
            "reinterpret_hash": self.reinterpret_hash[:16],
            "reinterpretation_differs": self.reinterpretation_differs,
            "migration_recommended": self.migration_recommended,
            "pinned_runtimes": dict(self.pinned_runtimes),
            "changes": [str(c) for c in self.diff.changes] if self.diff else [],
        }


def build_history(spine: Sequence[Edit] = SPINE) -> List[Revision]:
    """Compile each edit as it happened, keeping the result rather than the text.

    The stored form is what a replay reads. Keeping only the text and
    recompiling later is precisely the mistake this benchmark exists to detect:
    it silently substitutes today's meaning for the one the user agreed to.
    """
    revisions: List[Revision] = []
    for edit in spine:
        started = time.perf_counter_ns()
        result = compile_scenario(edit.text, name="plan", version=1,
                                  benchmark_rule=BENCHMARK_RULE)
        elapsed = (time.perf_counter_ns() - started) // 1000
        # Stored as the compiler of the day would have written it. Compiling
        # both sides with today's compiler is how a version benchmark measures
        # nothing while appearing to pass.
        compiler = edit.runtimes.get("compiler", "1")
        revisions.append(Revision(
            at=edit.at, what=edit.what, text=edit.text, runtimes=edit.runtimes,
            compiler=compiler,
            stored=as_compiled_by(result.scenario.to_json(), compiler),
            unresolved=tuple(u.field for u in result.unresolved),
            compile_us=elapsed))
    return revisions


def replay(revision: Revision) -> tuple:
    """Read back what was saved. No compiler runs.

    This is the operation a plan page owes when someone opens an old plan, and
    the one the workspace was not performing: it recompiled the text and
    simulated the *new* interpretation while displaying the stored scenario.
    """
    from ..mission.scenario import _hash

    started = time.perf_counter_ns()
    stored = revision.stored
    # Verified, not trusted. Re-deriving the hash from the stored body is the
    # same rule the evidence ledger follows: a record that can be edited without
    # detection proves nothing about what was saved.
    body = {k: stored[k] for k in ("spec_version", "name", "version",
                                   "objective", "methodology", "protocol",
                                   "flows", "inferred") if k in stored}
    identity = _hash(body)
    return identity, (time.perf_counter_ns() - started) // 1000


def reinterpret(revision: Revision) -> tuple:
    """Compile the original words again, under today's compiler."""
    started = time.perf_counter_ns()
    result = compile_scenario(revision.text, name="plan", version=1,
                              benchmark_rule=BENCHMARK_RULE)
    elapsed = (time.perf_counter_ns() - started) // 1000
    return result, elapsed


def run(spine: Sequence[Edit] = SPINE) -> List[Checkpoint]:
    revisions = build_history(spine)
    checkpoints: List[Checkpoint] = []

    for index, revision in enumerate(revisions, 1):
        replay_hash, replay_us = replay(revision)
        current, reinterpret_us = reinterpret(revision)

        diff = diff_stored_against(
            revision.stored, current.scenario,
            stored_compiler=revision.compiler,
            current_compiler=COMPILER_VERSION,
            stored_unresolved=revision.unresolved,
            current_unresolved=[u.field for u in current.unresolved])
        proposal = propose_migration("plan", diff)

        checkpoints.append(Checkpoint(
            at=revision.at, what=revision.what, revisions=index,
            replay_us=replay_us, replay_hash=replay_hash,
            reinterpret_us=reinterpret_us,
            reinterpret_hash=current.scenario.content_hash,
            diff=diff if not diff.is_empty else None,
            migration_recommended=bool(proposal and proposal.recommended),
            pinned_runtimes=revision.runtimes))
    return checkpoints


def summarize(checkpoints: Sequence[Checkpoint]) -> Dict[str, Any]:
    return {
        "checkpoints": len(checkpoints),
        "reinterpretations_differing": sum(
            1 for c in checkpoints if c.reinterpretation_differs),
        "migrations_recommended": sum(
            1 for c in checkpoints if c.migration_recommended),
        "replay_us_max": max((c.replay_us for c in checkpoints), default=0),
        "reinterpret_us_max": max(
            (c.reinterpret_us for c in checkpoints), default=0),
    }
