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
            closure_path: Optional[Path] = None, now=None) -> Gate:
    """The gate, from artifacts on disk rather than from a fresh run.

    Reading files rather than measuring here is deliberate: the drift lane
    costs provider calls and must not be triggered by something asking whether
    it may start Lean. A gate that ran its own evidence would be re-earning the
    guarantee at the moment of being asked, which is when the answer is least
    likely to be scrutinised.
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
        unsafe = drift.get("execution_unsafe", [])
        evidence["execution_unsafe"] = len(unsafe)
        evidence["by_classification"] = drift.get("by_classification", {})
        if unsafe:
            blockers.append(
                f"{len(unsafe)} prompt(s) are UNSTABLE_EXECUTABLE: a draw can "
                "change what executes. " + "; ".join(t[:60] for t in unsafe[:3]))

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
