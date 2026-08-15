"""What may execute, in a form that can cross a process boundary.

    VerifiedIntent -> Mission Runtime -> StrategySpec -> Evaluation Service

`ScenarioSpecification` is the engine's own object and carries engine types.
This is the same plan expressed as data: strings, numbers, and nothing that
imports. It is what the evaluation service will receive when it is a service,
and producing it now — while the evaluator is still in-process — is what makes
that later change a deployment rather than a redesign.

**Execution semantics only.** No prose, no reader identity, no Discovery types,
no UI types. Not because those are unimportant but because they are somebody
else's: an evaluator that could see the sentence could act on it, and the whole
point of the boundary is that it cannot. `tests/test_strategy_spec.py` reads the
serialized form and fails on anything that looks like a sentence.

**Hashed over what executes, not over how it was written.** Two people who
describe the same plan differently get one `spec_hash`, and the run they share
is genuinely the same run. Two plans that differ in anything the evaluator acts
on get different hashes — including the dividend policy, which was an engine
constant until it turned out to select the price series.

**Versioned, because the hash is a promise.** `SPEC_VERSION` is part of the
hashed body: a field added later changes every hash, and a comparison across
that change must be able to say "these were computed under different rules"
rather than "these differ".
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: The shape of this contract. In the hashed body on purpose.
SPEC_VERSION = "quantify-strategy-spec@1"


@dataclass(frozen=True)
class Trigger:
    """What is watched, and what counts as the event."""

    subject: str
    window: int
    estimator: str
    kind: str


@dataclass(frozen=True)
class Funding:
    """When money arrives, and how much.

    One of two shapes, distinguished by `kind`, because an evaluator that had
    to infer which from the presence of fields would eventually infer wrong.
    """

    kind: str                       # "scheduled" | "event_triggered"
    amount: str
    cadence: str = ""
    day_rule: str = ""
    execution_timing: str = ""
    trigger: Optional[Trigger] = None


@dataclass(frozen=True)
class Allocation:
    assets: Tuple[str, ...]
    weighting: str
    weights: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Conventions:
    """QuantLib's names for the rules, so a reader outside this repository can
    check a claim against a definition that is not ours."""

    vocabulary: str
    calendar: str
    business_day: str
    annualisation: str
    sessions_per_year: str


@dataclass(frozen=True)
class StrategySpec:
    """Everything the evaluator needs, and nothing about who asked."""

    objective: str
    assets: Tuple[str, ...]
    observed_assets: Tuple[str, ...]
    funding: Funding
    allocation: Allocation
    dividend_policy: str
    """Load-bearing, and it did not look it. This selects the price series —
    total-return against price-return — so it changes the figure and the
    market-data request, and it belongs in the identity for that reason."""
    sells_allowed: bool
    rebalancing_allowed: bool
    rebalancing_cadence: str
    tax_treatment: str
    evaluation_window: str
    benchmarks: Tuple[str, ...]
    conventions: Conventions
    version: str = SPEC_VERSION

    def to_json(self) -> Dict[str, Any]:
        """The wire form. Sorted and plain, so the bytes are the contract."""
        return json.loads(json.dumps(asdict(self), sort_keys=True,
                                     default=str))

    @property
    def spec_hash(self) -> str:
        body = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return "spec1:" + hashlib.sha256(body.encode()).hexdigest()


def to_scenario(spec: StrategySpec, *, name: str = "plan", version: int = 1):
    """The specification, back as something the engine can run.

    The inverse of `from_scenario`, and it exists to be *checked*: if a
    scenario cannot be rebuilt from its own specification, the specification is
    lossy and the evaluation service would be receiving less than the engine
    needs. `test_strategy_spec` round-trips every shape and compares canonical
    forms, which is the cheapest possible version of Step 6's conformance run
    and catches the same class of defect a year earlier.

    `name` and `version` are arguments rather than fields. They label a plan and
    do not change what it executes, so putting them in the spec would put them
    in the hash, and two identical strategies with different titles would stop
    being the same run.
    """
    from decimal import Decimal

    from .funding import EventTriggered, ExecutionTiming, Scheduled
    from .funding import Trigger as EngineTrigger
    from .scenario import (AllocationRule, BenchmarkSet, FlowSchedule,
                           HoldingsPolicy, Objective, ScenarioSpecification)
    from .signals import Estimator, SignalKind

    allocation = AllocationRule(
        assets=tuple(spec.allocation.assets),
        weighting=spec.allocation.weighting,
        weights={k: float(v) for k, v in (spec.allocation.weights or {}).items()})

    if spec.funding.kind == "event_triggered":
        trigger = spec.funding.trigger
        funding = EventTriggered(
            trigger=EngineTrigger(
                subject=trigger.subject, window=int(trigger.window),
                estimator=Estimator(trigger.estimator),
                kind=SignalKind(trigger.kind)),
            amount=Decimal(spec.funding.amount),
            execution_timing=ExecutionTiming(spec.funding.execution_timing))
        schedule = FlowSchedule(cadence="event_triggered", amount=0.0,
                                day_rule=spec.funding.day_rule or
                                "first_session_of_period")
    else:
        funding = Scheduled(cadence=spec.funding.cadence,
                            amount=Decimal(spec.funding.amount),
                            day_rule=spec.funding.day_rule)
        schedule = FlowSchedule(cadence=funding.cadence,
                                amount=float(funding.amount),
                                day_rule=funding.day_rule)

    return ScenarioSpecification(
        name=name, version=version, objective=Objective(spec.objective),
        event_program=(), flow_schedule=schedule, allocation_rule=allocation,
        holdings_policy=HoldingsPolicy(
            sells_allowed=spec.sells_allowed,
            rebalancing_allowed=spec.rebalancing_allowed,
            rebalancing_cadence=spec.rebalancing_cadence,
            dividend_policy=spec.dividend_policy),
        benchmark_set=(BenchmarkSet(generated_by_rule="", members=(),
                                    ordering="unordered")
                       if spec.benchmarks or spec.benchmarks == () else None),
        tax_treatment=spec.tax_treatment,
        funding=funding)


def from_scenario(scenario, *, evaluation_window: str = "") -> StrategySpec:
    """The engine's plan, as data.

    Mechanical: every value is read from the scenario and none is decided here.
    A default applied at this step would be a third place values come from,
    after Discovery's canonicalisation and Mission's declared defaults, and the
    plan a person reviewed would not be the plan that ran.
    """
    from .funding import EventTriggered

    holdings = scenario.holdings_policy
    rule = scenario.allocation_rule
    money = scenario.funding

    if isinstance(money, EventTriggered):
        funding = Funding(
            kind="event_triggered",
            amount=str(money.amount),
            execution_timing=str(getattr(money.execution_timing, "value",
                                         money.execution_timing)),
            trigger=Trigger(
                subject=money.trigger.subject,
                window=int(money.trigger.window),
                estimator=str(getattr(money.trigger.estimator, "value",
                                      money.trigger.estimator)),
                kind=str(getattr(money.trigger.kind, "value",
                                 money.trigger.kind))))
    else:
        funding = Funding(kind="scheduled", amount=str(money.amount),
                          cadence=str(money.cadence),
                          day_rule=str(money.day_rule))

    from .conventions import declared

    named = declared()
    benchmarks = getattr(scenario.benchmark_set, "members", ()) or ()

    return StrategySpec(
        objective=str(getattr(scenario.objective, "value", scenario.objective)),
        assets=tuple(rule.assets),
        observed_assets=((funding.trigger.subject,)
                         if funding.trigger is not None else ()),
        funding=funding,
        allocation=Allocation(
            assets=tuple(rule.assets), weighting=str(rule.weighting),
            weights={k: str(v) for k, v in sorted((rule.weights or {}).items())}),
        dividend_policy=str(holdings.dividend_policy),
        sells_allowed=bool(holdings.sells_allowed),
        rebalancing_allowed=bool(holdings.rebalancing_allowed),
        rebalancing_cadence=str(holdings.rebalancing_cadence or ""),
        tax_treatment=str(scenario.tax_treatment),
        evaluation_window=evaluation_window,
        benchmarks=tuple(str(one) for one in benchmarks),
        conventions=Conventions(
            vocabulary=str(named.get("vocabulary", "")),
            calendar=str(named.get("exchange", "")),
            business_day=str(named.get("contribution_convention", "")),
            annualisation=str(named.get("annualisation", "")),
            sessions_per_year=str(named.get("sessions_per_year", ""))))
