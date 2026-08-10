"""Whether formal verification may start yet.

    from src.mission.prelean_gate import verdict
    verdict()          # -> Gate(open=False, blockers=[...])

Two conditions, and neither is "Discovery is deterministic":

    zero UNSTABLE_EXECUTABLE   in the live drift artifact
    zero SILENTLY_REDUCED      in the serving closure report

**Why that is the right pair.** Lean proves that deterministic operators obey
their contract. It cannot prove the contract describes what the person asked
for. So the thing that must hold first is that an unsupported or ambiguous
intent never arrives at the engine wearing an executable shape — because
formalising the calculator while Discovery can still turn *convert IRA to Roth*
into *annual Roth contribution* would let Lean prove the wrong strategy
perfectly.

Determinism is not required and would be the wrong target. A stochastic reader
that lands on REFUSAL one draw and CLARIFICATION the next is safe: nothing
executes either way, and the person is asked something. What must be zero is
instability that can change what executes.

**Stale evidence is not evidence.** The drift artifact records the schema,
prompt, pipeline and fusion versions it was produced against, and this gate
refuses it when any of them has moved or when it is more than a week old.
Without that, "we ran the drift lane once" becomes a permanent guarantee that
nobody re-earned — the same failure as a frozen fingerprint quietly re-pointed
at a schema it was never computed under.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

CORPUS = Path(__file__).resolve().parent.parent.parent / "corpus" / "parser"
DRIFT = CORPUS / "drift.json"
CLOSURE = CORPUS / "strategy_closure.json"

VALID_FOR_DAYS = 7

#: Prompts that have been UNSTABLE_SAFE — the draws disagreed and none of them
#: executed. They do not block, and they are named here because the transition
#: that would matter is invisible in a count.
#:
#: "Unstable but always safe" and "sometimes executable" differ by one draw. A
#: later model or prompt change could move any of these across that line, and
#: the totals would barely shift: UNSTABLE_SAFE 6 -> 5 and UNSTABLE_EXECUTABLE
#: 0 -> 1 reads as noise unless something says these six in particular were
#: being watched.
WATCHED = (
    "withdraw 4% of the portfolio each year, adjusted for inflation",
    "a 60/40 portfolio",
    "70% stocks, 20% bonds, 10% cash",
    "hold my age in bonds",
    "refill the cash bucket from stocks after a good year",
    "sell covered calls one strike out of the money each month",
)


@dataclass(frozen=True)
class Gate:
    """Whether Lean may start, and everything standing in the way."""

    open: bool
    blockers: Sequence[str] = field(default_factory=tuple)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"open": self.open, "blockers": list(self.blockers),
                "evidence": dict(self.evidence)}


def _current_versions() -> dict:
    import sys

    sys.path.insert(0, str(CORPUS.parent))
    from shadow_run import schema_fingerprint                   # noqa: E402

    from ..discovery.hosted_recording import PROMPT_VERSION
    from ..discovery.pipeline import PIPELINE_VERSION
    from ..discovery.schema import QUANTIFY_SCHEMA

    return {"schema_fingerprint": schema_fingerprint(QUANTIFY_SCHEMA),
            "prompt_version": PROMPT_VERSION,
            "pipeline_version": PIPELINE_VERSION}


def _staleness(drift: Mapping[str, Any], *, now=None) -> Sequence[str]:
    """Why this artifact may not be cited, if it may not."""
    from datetime import datetime, timedelta, timezone

    problems = []
    provenance = drift.get("provenance", {})
    current = _current_versions()
    for name, value in current.items():
        was = provenance.get(name)
        if was != value:
            problems.append(
                f"the drift artifact was produced against {name}={was!r} and "
                f"this build is {value!r}; re-run corpus/parser/drift_lane.py")

    recorded = provenance.get("recorded_at")
    if not recorded:
        problems.append("the drift artifact does not say when it was produced")
    else:
        now = now or datetime.now(timezone.utc)
        try:
            at = datetime.fromisoformat(recorded)
        except ValueError:
            problems.append(f"unreadable timestamp {recorded!r}")
        else:
            age = now - at
            if age > timedelta(days=VALID_FOR_DAYS):
                problems.append(
                    f"the drift artifact is {age.days} days old and evidence "
                    f"about stochastic behaviour is valid for {VALID_FOR_DAYS}; "
                    "the provider can change under a fixed model id")

    # A single draw measures nothing about stability. The longitudinal lane
    # writes its own file for exactly this reason, and pointing the gate at it
    # would turn a provider-drift check into a stability claim.
    if provenance.get("draws_per_prompt", 0) < 3:
        problems.append(
            f"produced with {provenance.get('draws_per_prompt')} draw(s) per "
            "prompt; stability needs at least 3")
    return tuple(problems)


def verdict(*, drift_path: Optional[Path] = None,
            closure_path: Optional[Path] = None, now=None,
            require_ci: bool = True) -> Gate:
    """The gate, from artifacts on disk rather than from a fresh run.

    Reading files rather than measuring here is deliberate: the drift lane
    costs provider calls and must not be triggered by something asking whether
    it may start Lean. A gate that ran its own evidence would be re-earning the
    guarantee at the moment of being asked, which is when the answer is least
    likely to be scrutinised.

    `require_ci` is the difference between "somebody ran this on a laptop" and
    "the scheduled lane spoke to the provider". Both are useful and only one is
    a guarantee: a seven-day-old local run would otherwise keep the gate open
    while CI had never successfully made a call. Development passes `False`
    deliberately and says so; nothing else should.
    """
    drift_path = drift_path or DRIFT
    closure_path = closure_path or CLOSURE

    blockers, evidence = [], {}

    if not drift_path.exists():
        blockers.append(
            "no drift artifact; run corpus/parser/drift_lane.py, which calls "
            "the provider live in the serving profile")
    else:
        drift = json.loads(drift_path.read_text())
        blockers.extend(_staleness(drift, now=now))

        producer = drift.get("provenance", {}).get("producer", "unknown")
        evidence["producer"] = producer
        evidence["mode"] = drift.get("provenance", {}).get("mode", "")
        if require_ci and producer != "github-actions":
            blockers.append(
                f"the drift artifact was produced by {producer!r}, not the "
                "scheduled lane. A local run is evidence for development and "
                "not a guarantee about the deployment that serves people; "
                "dispatch .github/workflows/drift-lane.yml")
        unsafe = drift.get("execution_unsafe", [])
        evidence["execution_unsafe"] = len(unsafe)
        evidence["by_classification"] = drift.get("by_classification", {})
        if unsafe:
            blockers.append(
                f"{len(unsafe)} prompt(s) are UNSTABLE_EXECUTABLE: a draw can "
                "change what executes. " + "; ".join(t[:60] for t in unsafe[:3]))

        # Across draws, not from one recording. On one draw `sell VTI and buy
        # BND` carries sell_action and is refused; on the next it does not and
        # executes. A gate reading a single-draw number would open on the luck
        # of which recording happened to be current.
        # A watched prompt crossing into executable. Reported separately from
        # the general count so the blocker names the transition rather than a
        # number that moved by one.
        crossed = [r["text"] for r in drift.get("results", [])
                   if r["text"] in WATCHED
                   and r["classification"] == "UNSTABLE_EXECUTABLE"]
        evidence["watched_crossed"] = len(crossed)
        if crossed:
            blockers.append(
                f"{len(crossed)} watched prompt(s) moved from unstable-but-safe "
                "to sometimes-executable: " + "; ".join(t[:60] for t in crossed))

        reduced_any = drift.get("silently_reduced_any_draw")
        if reduced_any is not None:
            evidence["silently_reduced_any_draw"] = len(reduced_any)
            if reduced_any:
                blockers.append(
                    f"{len(reduced_any)} unsupported intent(s) executed on at "
                    "least one live draw: "
                    + "; ".join(t[:60] for t in reduced_any[:3]))

    if not closure_path.exists():
        blockers.append("no serving closure report; run "
                        "corpus/parser/strategy_closure.py")
    else:
        closure = json.loads(closure_path.read_text())
        reduced = closure.get("by_state", {}).get("SILENTLY_REDUCED", 0)
        evidence["silently_reduced"] = reduced
        evidence["closure_witness"] = closure.get("witness")
        if reduced:
            blockers.append(
                f"{reduced} known unsupported intent(s) still collapse into an "
                "executable plan on the serving path")

    return Gate(open=not blockers, blockers=tuple(blockers), evidence=evidence)


def main() -> int:
    gate = verdict()
    print("pre-Lean gate:", "OPEN" if gate.open else "CLOSED")
    for name, value in gate.evidence.items():
        print(f"  {name:22} {value}")
    for blocker in gate.blockers:
        print(f"  BLOCKED  {blocker}")
    return 0 if gate.open else 1


if __name__ == "__main__":
    raise SystemExit(main())
