"""Bounded evaluation of model-assisted compiler stage 1.

Two things had to be true before spending money on this, and both now are:

    paraphrases converge to one canonical Mission          stability 100%
    specifications round-trip without identity or          round-trip 100%
    provenance drift

So a failure here is attributable to **model extraction**, not to an unstable
compiler or renderer underneath it. That is the whole reason this run is worth
paying for.

A semantic failure is a **result, not an error**. Retries are for transport
faults and malformed structured output only: retrying a wrong interpretation
until it comes out right measures persistence, not the model.

Everything is captured. Failures become permanent regression fixtures, and two
of the failure classes are deliberately about the harness itself — the corpus
has already shown more than once that the expectation can be the thing that is
wrong.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..mission.compiler import compile_scenario, parse
from ..mission.evolution import diff_stored_against
from ..mission.parse_model import build_system_prompt, parse_with_model

BENCHMARK_RULE = "benchmark-policy/public-default@1"


class Failure(str, Enum):
    NONE = "NONE"
    MISSED_RECOGNITION = "MISSED_RECOGNITION"
    FALSE_INFERENCE = "FALSE_INFERENCE"
    PROVENANCE_ERROR = "PROVENANCE_ERROR"
    CONTRADICTION_MISS = "CONTRADICTION_MISS"
    UNSTABLE_PARAPHRASE = "UNSTABLE_PARAPHRASE"
    SCHEMA_FAILURE = "SCHEMA_FAILURE"
    COMPILER_GAP = "COMPILER_GAP"
    """The deterministic compiler, not the model."""

    EXPECTATION_DEFECT = "EXPECTATION_DEFECT"
    """The benchmark was wrong. Kept as a category because it has happened."""


@dataclass
class Case:
    case_id: str
    family: str
    klass: str
    text: str
    expects_contradiction: bool = False
    expects_questions: bool = False


@dataclass
class Outcome:
    """One case, with everything needed to reproduce or re-judge it."""

    case_id: str
    family: str
    klass: str
    text: str

    model_available: bool = False
    model_error: str = ""
    resolved_model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency_ms: float = 0.0
    retries: int = 0
    retry_reason: str = ""

    raw_response_hash: str = ""
    accepted_from_model: Sequence[str] = ()
    rejected: Sequence[Dict[str, str]] = ()
    disagreements: Sequence[Dict[str, str]] = ()

    expected_content: str = ""
    expected_rule: str = ""
    expected_schedule: str = ""
    actual_content: str = ""
    actual_rule: str = ""
    actual_schedule: str = ""

    expected_provenance: Mapping[str, str] = field(default_factory=dict)
    actual_provenance: Mapping[str, str] = field(default_factory=dict)

    expected_questions: Sequence[str] = ()
    actual_questions: Sequence[str] = ()
    expected_contradictions: int = 0
    actual_contradictions: int = 0

    can_save: bool = False
    changes: Sequence[str] = ()
    failures: Sequence[str] = ()

    @property
    def content_exact(self) -> bool:
        return bool(self.actual_content) and self.actual_content == self.expected_content

    @property
    def rule_exact(self) -> bool:
        return bool(self.actual_rule) and self.actual_rule == self.expected_rule

    @property
    def schedule_exact(self) -> bool:
        return bool(self.actual_schedule) and self.actual_schedule == self.expected_schedule

    @property
    def provenance_exact(self) -> bool:
        return self.expected_provenance == self.actual_provenance

    def as_row(self) -> Dict[str, Any]:
        return {**self.__dict__,
                "content_exact": self.content_exact,
                "rule_exact": self.rule_exact,
                "schedule_exact": self.schedule_exact,
                "provenance_exact": self.provenance_exact}


def provenance_of(result) -> Dict[str, str]:
    """Field -> STATED | INFERRED | UNRESOLVED.

    A correct value with the wrong provenance is not fully correct; the
    round-trip work is what proved that, when restating an inference silently
    turned a system default into a user decision.
    """
    out: Dict[str, str] = {}
    for entry in result.stated:
        out.setdefault(f"span:{entry}", "STATED")
    for inference in result.inferred:
        out[inference.field] = "INFERRED"
    for question in result.unresolved:
        out[question.field] = "UNRESOLVED"
    return dict(sorted(out.items()))


def _judge(outcome: Outcome, deterministic_fields: Sequence[str],
           model_fields: Sequence[str]) -> List[str]:
    """Classify what went wrong, including whether it was the harness."""
    failures: List[str] = []
    if not outcome.model_available:
        failures.append(Failure.SCHEMA_FAILURE.value)
        return failures

    settled_more = (set(outcome.expected_questions)
                    - set(outcome.actual_questions))
    if settled_more:
        # The model answered something the description never settled. The
        # headline trust metric: a plausible answer nobody asked for.
        failures.append(Failure.FALSE_INFERENCE.value)

    if set(outcome.actual_questions) - set(outcome.expected_questions):
        failures.append(Failure.MISSED_RECOGNITION.value)

    if (outcome.expected_contradictions and not outcome.actual_contradictions):
        failures.append(Failure.CONTRADICTION_MISS.value)

    if (outcome.rule_exact and outcome.schedule_exact
            and not outcome.provenance_exact):
        failures.append(Failure.PROVENANCE_ERROR.value)

    if not outcome.rule_exact or not outcome.schedule_exact:
        if not failures:
            failures.append(Failure.MISSED_RECOGNITION.value)
    return failures or [Failure.NONE.value]


def run_case(case: Case, client, *, max_retries: int = 1) -> Outcome:
    """One case: deterministic expectation, then the model-assisted attempt."""
    outcome = Outcome(case_id=case.case_id, family=case.family,
                      klass=case.klass, text=case.text)

    expected = compile_scenario(case.text, name="eval", version=1,
                                benchmark_rule=BENCHMARK_RULE)
    if expected.scenario is not None:
        outcome.expected_content = expected.scenario.content_hash
        outcome.expected_rule = expected.scenario.rule_hash
        outcome.expected_schedule = expected.scenario.flow_schedule.schedule_hash
    outcome.expected_provenance = provenance_of(expected)
    outcome.expected_questions = tuple(u.field for u in expected.unresolved)
    outcome.expected_contradictions = len(expected.contradictions)

    attempt, verified = 0, None
    while attempt <= max_retries:
        started = time.perf_counter()
        verified = parse_with_model(case.text, client=client)
        outcome.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        meta = getattr(client, "last_response", None) or {}
        outcome.resolved_model = meta.get("resolved_model")
        outcome.input_tokens = meta.get("input_tokens")
        outcome.output_tokens = meta.get("output_tokens")

        if verified.provenance.model_available:
            break
        # Transport, timeout or unparseable output only. A wrong reading is a
        # result and is never retried.
        outcome.retries = attempt + 1
        outcome.retry_reason = verified.provenance.model_error
        attempt += 1

    outcome.model_available = verified.provenance.model_available
    outcome.model_error = verified.provenance.model_error
    outcome.raw_response_hash = hashlib.sha256(
        json.dumps(verified.parsed.to_json(), sort_keys=True).encode()
    ).hexdigest()[:32]
    outcome.accepted_from_model = tuple(verified.provenance.accepted_from_model)
    outcome.rejected = tuple(r.to_json() for r in verified.provenance.rejected)
    outcome.disagreements = tuple(d.to_json()
                                  for d in verified.provenance.disagreements)

    actual = compile_scenario(case.text, name="eval", version=1,
                              benchmark_rule=BENCHMARK_RULE,
                              parsed=verified.parsed)
    if actual.scenario is not None:
        outcome.actual_content = actual.scenario.content_hash
        outcome.actual_rule = actual.scenario.rule_hash
        outcome.actual_schedule = actual.scenario.flow_schedule.schedule_hash
    outcome.actual_provenance = provenance_of(actual)
    outcome.actual_questions = tuple(u.field for u in actual.unresolved)
    outcome.actual_contradictions = len(actual.contradictions)
    outcome.can_save = actual.can_save

    if expected.scenario is not None and actual.scenario is not None:
        diff = diff_stored_against(expected.scenario.to_json(), actual.scenario)
        outcome.changes = tuple(str(c) for c in diff.changes)

    outcome.failures = tuple(_judge(
        outcome,
        [r.field for r in parse(case.text).recognitions],
        list(outcome.accepted_from_model)))
    return outcome


def pins(client, outcomes: Sequence[Outcome]) -> Dict[str, Any]:
    """Everything needed to say what was run, and to run it again."""
    import subprocess

    try:
        from runtime_contracts.canonical import CANONICALIZATION_VERSION
    except ImportError:
        # Optional here; the contracts package pins wire semantics for the
        # control plane, not for this compiler. Recorded as absent rather than
        # guessed, because a pin nobody verified is worse than a stated gap.
        CANONICALIZATION_VERSION = "not-installed"

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:                                       # noqa: BLE001
            return "unknown"

    prompt = build_system_prompt()
    resolved = sorted({o.resolved_model for o in outcomes if o.resolved_model})
    return {
        "provider": "anthropic",
        "requested_model": getattr(client, "model", None),
        "resolved_models": resolved,
        "system_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "system_prompt_chars": len(prompt),
        "compiler_commit": git("rev-parse", "--short", "HEAD"),
        "compiler_version": __import__(
            "src.mission.evolution", fromlist=["x"]).COMPILER_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "sampling": "provider defaults; temperature is rejected by this model",
        "max_tokens": getattr(client, "max_tokens", None),
    }


def summarize(outcomes: Sequence[Outcome]) -> Dict[str, Any]:
    n = len(outcomes) or 1
    answered = [o for o in outcomes if o.model_available]

    settled_early = [o for o in outcomes
                     if set(o.expected_questions) - set(o.actual_questions)]
    expects_contradiction = [o for o in outcomes if o.expected_contradictions]

    by_family: Dict[str, set] = {}
    for o in outcomes:
        by_family.setdefault(o.family, set()).add(o.actual_content)
    converged = [f for f, hashes in by_family.items() if len(hashes) == 1]

    latencies = sorted(o.latency_ms for o in outcomes if o.latency_ms)

    def at(q: float) -> float:
        return latencies[min(len(latencies) - 1, int(q * len(latencies)))] \
            if latencies else 0.0

    failure_counts: Dict[str, int] = {}
    for o in outcomes:
        for f in o.failures:
            failure_counts[f] = failure_counts.get(f, 0) + 1

    return {
        "cases": len(outcomes),
        "model_available": len(answered),
        "schema_failure_rate": round(100 * (len(outcomes) - len(answered)) / n, 1),
        "retry_rate": round(100 * sum(1 for o in outcomes if o.retries) / n, 1),
        "content_exact": round(100 * sum(o.content_exact for o in outcomes) / n, 1),
        "rule_exact": round(100 * sum(o.rule_exact for o in outcomes) / n, 1),
        "schedule_exact": round(100 * sum(o.schedule_exact for o in outcomes) / n, 1),
        "provenance_exact": round(
            100 * sum(o.provenance_exact for o in outcomes) / n, 1),
        "false_inference_rate": round(100 * len(settled_early) / n, 1),
        # `None` when no contradictory case ran, rather than 0%. A recall of
        # zero over an empty set reads as total failure and is meaningless.
        "contradiction_cases": len(expects_contradiction),
        "contradiction_recall": (round(
            100 * sum(1 for o in expects_contradiction if o.actual_contradictions)
            / len(expects_contradiction), 1) if expects_contradiction else None),
        "families": len(by_family),
        "families_converged": len(converged),
        "family_convergence_rate": round(
            100 * len(converged) / (len(by_family) or 1), 1),
        "latency_p50_ms": at(0.50), "latency_p95_ms": at(0.95),
        "latency_p99_ms": at(0.99),
        "input_tokens": sum(o.input_tokens or 0 for o in outcomes),
        "output_tokens": sum(o.output_tokens or 0 for o in outcomes),
        "failure_classes": dict(sorted(failure_counts.items(),
                                       key=lambda kv: -kv[1])),
        "saveable_with_open_questions": sum(
            1 for o in outcomes if o.can_save and o.actual_questions),
    }
