"""Run the corpus through the compiler and record what happened.

The measurement that matters is not throughput. It is **agreement between what a
prompt owed and what it got**:

    a prompt missing a material choice must ask a question
    a prompt stating two incompatible choices must report the contradiction
    a prompt asking which is best must not answer
    a fully specified prompt must reach a saveable scenario

A run reporting "14,400 compiled" has measured a loop. A run reporting which
classes disagreed with their expectation has found something.

Latency is recorded per stage rather than end to end, because "compile took
40 ms" cannot be acted on and "stage 1 took 38 ms of it" can.
"""
from __future__ import annotations

import statistics
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from ..mission.compiler import CompilerResult, compile_scenario, parse
from .paraphrase import Expect, Klass, Prompt


@dataclass
class Outcome:
    """One prompt, and everything observed about compiling it."""

    prompt_id: str
    strategy_id: str
    family: str
    klass: str
    expect: str
    probes: Optional[str]

    ok: bool = False
    """Whether the compiler behaved as the prompt required."""

    crashed: bool = False
    error: str = ""
    disagreement: str = ""

    can_simulate: bool = False
    can_save: bool = False
    stated: int = 0
    inferred: int = 0
    unresolved: int = 0
    contradictions: int = 0
    recognized_fields: Sequence[str] = ()
    probe_recognized: Optional[bool] = None
    """Whether the field this prompt probes was actually read. `None` when the
    prompt probes nothing."""

    parse_us: int = 0
    compile_us: int = 0
    total_us: int = 0

    def as_row(self) -> Dict[str, Any]:
        return {
            "prompt_id": self.prompt_id, "strategy_id": self.strategy_id,
            "family": self.family, "klass": self.klass, "expect": self.expect,
            "probes": self.probes, "ok": self.ok, "crashed": self.crashed,
            "error": self.error, "disagreement": self.disagreement,
            "can_simulate": self.can_simulate, "can_save": self.can_save,
            "stated": self.stated, "inferred": self.inferred,
            "unresolved": self.unresolved, "contradictions": self.contradictions,
            "probe_recognized": self.probe_recognized,
            "parse_us": self.parse_us, "compile_us": self.compile_us,
            "total_us": self.total_us,
        }


def _saveable_after_confirmation(result: CompilerResult) -> bool:
    """Whether only the confirmation step stands between this and a saved plan.

    `can_save` is False while any inference is unconfirmed, which is correct and
    is what the confirmation screen exists for. What a complete description owes
    is: nothing left to ask, and a scenario that runs.
    """
    return (not result.unresolved
            and result.scenario is not None
            and result.scenario.is_runnable)


def _judge(prompt: Prompt, result: CompilerResult,
           recognized: Sequence[str]) -> str:
    """Empty string when the compiler did what the prompt required.

    Deliberately strict about the two directions that matter. Answering a
    question nobody could answer is a guess; asking about something the user
    already said is noise that trains people to click through confirmations.
    """
    if prompt.expect is Expect.ASKS_A_QUESTION:
        if result.can_save:
            return ("saved a scenario from a description that omits a material "
                    "choice — the omission was filled by a guess")
        if not result.unresolved and not result.inferred:
            return "neither asked a question nor recorded an inference"

    elif prompt.expect is Expect.REPORTS_A_CONTRADICTION:
        if not result.contradictions:
            return ("two stated choices cannot both hold and no contradiction "
                    "was reported; one was silently dropped")

    elif prompt.expect is Expect.REFUSES_TO_CHOOSE:
        if result.can_save:
            return "compiled a saveable plan from a request to be told what to do"

    elif prompt.expect is Expect.COMPILES_SAVEABLE:
        if result.contradictions:
            return ("reported a contradiction in a description that states no "
                    "incompatible choices")
        if prompt.probes and prompt.probes not in recognized:
            return (f"did not read {prompt.probes}, which this wording states "
                    "explicitly")
        if prompt.klass is Klass.COMPLETE and not _saveable_after_confirmation(result):
            # The class is named for reaching a saveable scenario. Passing it
            # without checking that is the vacuous-pass this harness exists to
            # avoid — it hid two recognizer gaps for a full run.
            #
            # "After confirmation" is the real bar. An unconfirmed inference
            # correctly blocks save; the user confirms it on the next screen.
            # Grading against `can_save` alone reports the confirmation step
            # itself as a defect.
            asked = ", ".join(sorted({u.field for u in result.unresolved}))
            return (f"states every material choice and still cannot be saved; "
                    f"still asks about: {asked or 'nothing, yet unsaveable'}")
    return ""


#: What the workspace passes. The harness must compile the way the product
#: compiles: running without it made every COMPLETE prompt unsaveable on a
#: benchmark question the real route never asks, which nearly got reported as a
#: compiler defect.
BENCHMARK_RULE = "benchmark-policy/public-default@1"


def run_prompt(prompt: Prompt, *,
               parser: Optional[Callable] = None,
               benchmark_rule: Optional[str] = BENCHMARK_RULE) -> Outcome:
    """Compile one prompt. Never raises — a crash is an observation."""
    outcome = Outcome(prompt_id=prompt.prompt_id, strategy_id=prompt.strategy_id,
                      family=prompt.family, klass=prompt.klass.value,
                      expect=prompt.expect.value, probes=prompt.probes)
    started = time.perf_counter_ns()
    try:
        parse_started = time.perf_counter_ns()
        parsed = (parser(prompt.text) if parser else parse(prompt.text))
        outcome.parse_us = (time.perf_counter_ns() - parse_started) // 1000

        compile_started = time.perf_counter_ns()
        result = compile_scenario(prompt.text, name=prompt.prompt_id,
                                  benchmark_rule=benchmark_rule, parsed=parsed)
        outcome.compile_us = (time.perf_counter_ns() - compile_started) // 1000
    except Exception as exc:                                    # noqa: BLE001
        outcome.crashed = True
        outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.disagreement = "raised instead of compiling or refusing"
        outcome.total_us = (time.perf_counter_ns() - started) // 1000
        outcome.__dict__["traceback"] = traceback.format_exc()
        return outcome

    recognized = [r.field for r in parsed.recognitions]
    outcome.recognized_fields = tuple(recognized)
    outcome.probe_recognized = (prompt.probes in recognized
                                if prompt.probes else None)
    outcome.can_simulate = result.can_simulate
    outcome.can_save = result.can_save
    outcome.stated = len(result.stated)
    outcome.inferred = len(result.inferred)
    outcome.unresolved = len(result.unresolved)
    outcome.contradictions = len(result.contradictions)
    outcome.disagreement = _judge(prompt, result, recognized)
    outcome.ok = not outcome.disagreement
    outcome.total_us = (time.perf_counter_ns() - started) // 1000
    return outcome


def run_corpus(prompts: Iterable[Prompt], *,
               parser: Optional[Callable] = None,
               benchmark_rule: Optional[str] = BENCHMARK_RULE,
               progress: Optional[Callable[[int], None]] = None) -> List[Outcome]:
    outcomes = []
    for index, prompt in enumerate(prompts, 1):
        outcomes.append(run_prompt(prompt, parser=parser,
                                   benchmark_rule=benchmark_rule))
        if progress and index % 1000 == 0:
            progress(index)
    return outcomes


# --- reporting -------------------------------------------------------------

def percentiles(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {"n": len(ordered), "p50": at(0.50), "p95": at(0.95),
            "p99": at(0.99), "max": ordered[-1],
            "mean": round(statistics.fmean(ordered), 1)}


@dataclass
class Report:
    outcomes: List[Outcome]

    @property
    def crashes(self) -> List[Outcome]:
        return [o for o in self.outcomes if o.crashed]

    @property
    def disagreements(self) -> List[Outcome]:
        return [o for o in self.outcomes if o.disagreement and not o.crashed]

    def by_class(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for outcome in self.outcomes:
            bucket = out.setdefault(outcome.klass, {
                "n": 0, "ok": 0, "crashed": 0, "disagreed": 0,
                "saveable": 0, "asked": 0, "contradicted": 0})
            bucket["n"] += 1
            bucket["ok"] += outcome.ok
            bucket["crashed"] += outcome.crashed
            bucket["disagreed"] += bool(outcome.disagreement and not outcome.crashed)
            bucket["saveable"] += outcome.can_save
            bucket["asked"] += bool(outcome.unresolved)
            bucket["contradicted"] += bool(outcome.contradictions)
        return out

    def by_family(self) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for outcome in self.outcomes:
            bucket = out.setdefault(outcome.family, {"n": 0, "problems": 0})
            bucket["n"] += 1
            bucket["problems"] += bool(outcome.disagreement or outcome.crashed)
        return out

    def latency(self) -> Dict[str, Dict[str, float]]:
        return {
            "parse_us": percentiles([o.parse_us for o in self.outcomes]),
            "compile_us": percentiles([o.compile_us for o in self.outcomes]),
            "total_us": percentiles([o.total_us for o in self.outcomes]),
        }

    def distinct_problems(self) -> Dict[str, List[str]]:
        """Grouped by message, so 3,000 instances of one defect read as one."""
        out: Dict[str, List[str]] = {}
        for outcome in self.outcomes:
            key = outcome.error if outcome.crashed else outcome.disagreement
            if key:
                out.setdefault(key, []).append(outcome.prompt_id)
        return dict(sorted(out.items(), key=lambda kv: -len(kv[1])))
