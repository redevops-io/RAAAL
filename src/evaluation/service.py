"""`evaluate(...)` — the interface the evaluation service will present.

    evaluate(strategy_spec, market_snapshot_id, evaluation_policy,
             engine_version) -> EvaluationResult

In-process for now, deliberately. The final interface exists before the
deployment does, so moving the evaluator to its own process later is a change of
address rather than a redesign — and so the conformance run that proves the move
was faithful has something to compare against today.

**The result carries what it was computed from, not only what it computed.**
Six identities, and every one of them has been the thing that made a figure
unreproducible in this project at some point: the specification, the snapshot,
the evaluator, the engine, the conventions, and the shape of the result itself.

**Named streams, not one digest.** Equal headline numbers hide different wiring
— that is not a hypothesis here, it is how a frame digest matched while a plan
invested nothing, and how two plans with one hash turned out to be priced on
different price series. A single bundled hash tells you two runs differ and not
where. These are separate so a conformance failure names the stage.

**A stream that this run does not produce is absent, not empty.** A scheduled
plan has no signals; a plan with no trigger has no unfilled orders. Reporting
`()` for both "nothing happened" and "this run has no such stage" would make a
conformance comparison agree on two runs that did different things — which is
the exact failure the streams exist to prevent.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: The shape of an EvaluationResult. In the hashed body, so a comparison across
#: a change to this contract can say the rules differed rather than the runs.
RESULT_SCHEMA_VERSION = "quantify-evaluation-result@1"

#: Who computed it. Distinct from the engine version: the same engine behind a
#: different service build is a different thing to have called.
EVALUATOR = "quantify-in-process-evaluator@1"

#: Every stream a caller may ask about, named once so the result cannot
#: silently stop reporting one. A stage that produces nothing says so.
STREAMS = ("eligible_sessions", "signals", "contributions", "orders", "fills",
           "cash_flows", "metrics")


class EvaluationRefused(ValueError):
    """The evaluator will not answer, and says which input it could not use."""


@dataclass(frozen=True)
class Stream:
    """One named stage of a run, or a statement that this run has no such stage.

    `produced` is the distinction that matters. `rows=()` with
    `produced=True` means the stage ran and yielded nothing; with
    `produced=False` it means the stage does not exist for this plan. A
    comparison that could not tell those apart would report two different runs
    as identical.
    """

    name: str
    produced: bool
    rows: Tuple[Any, ...] = ()
    absent_because: str = ""

    @property
    def digest(self) -> str:
        body = json.dumps({"name": self.name, "produced": self.produced,
                           "rows": list(self.rows)},
                          sort_keys=True, default=str)
        return "st1:" + hashlib.sha256(body.encode()).hexdigest()[:16]

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "produced": self.produced,
                "rows": list(self.rows), "absent_because": self.absent_because,
                "digest": self.digest}


@dataclass(frozen=True)
class EvaluationResult:
    """A figure, and everything needed to ask for it again."""

    result_schema_version: str
    strategy_hash: str
    market_snapshot_hash: str
    market_snapshot_id: str
    evaluator: str
    evaluator_version: str
    engine_version: str
    conventions_version: str
    evaluation_policy: Any

    streams: Mapping[str, Stream] = field(default_factory=dict)
    figures: Mapping[str, Any] = field(default_factory=dict)
    refusals: Tuple[str, ...] = ()

    @property
    def identity(self) -> Dict[str, str]:
        """What this result was computed from, as one mapping.

        The chain the deployment proof already establishes for code, extended
        through the data: user meaning -> sealed intent -> specification ->
        snapshot -> evaluator revision -> result.
        """
        return {"result_schema_version": self.result_schema_version,
                "strategy_hash": self.strategy_hash,
                "market_snapshot_hash": self.market_snapshot_hash,
                "market_snapshot_id": self.market_snapshot_id,
                "evaluator": self.evaluator,
                "evaluator_version": self.evaluator_version,
                "engine_version": self.engine_version,
                "conventions_version": self.conventions_version,
                "evaluation_policy": self.evaluation_policy}

    def to_json(self) -> Dict[str, Any]:
        return {**self.identity,
                "streams": {name: stream.to_json()
                            for name, stream in sorted(self.streams.items())},
                "figures": dict(self.figures),
                "refusals": list(self.refusals)}


def _absent(name: str, why: str) -> Stream:
    return Stream(name=name, produced=False, absent_because=why)


def _sessions(path) -> Stream:
    index = getattr(getattr(path, "value", None), "index", None)
    if index is None:
        return _absent("eligible_sessions", "the run produced no value series")
    return Stream("eligible_sessions", True,
                  tuple(str(one.date()) if hasattr(one, "date") else str(one)
                        for one in index))


def _fills(path) -> Stream:
    fills = getattr(path, "fills", None)
    if fills is None:
        return _absent("fills", "the run produced no path")
    # Real attribute names, and no `getattr` default.
    #
    # This read `at`, `asset`, `units` and `cash`, none of which a `Fill` has —
    # and every one defaulted to `""`, so the stream reported rows of empty
    # strings while saying it had produced them. A default on a field access is
    # a silent substitution, which is the defect this whole layer was built to
    # stop, inside the layer. The conformance comparison found it in its first
    # run because the second reading uses direct access and raised.
    return Stream("fills", True, tuple(
        {"at": str(one.date), "asset": str(one.ticker),
         "units": str(one.shares), "price": str(one.price),
         "cash": str(one.notional), "cost": str(one.cost),
         "reason": str(one.reason)}
        for one in fills))


def _orders(path) -> Stream:
    """Filled and unfilled together, because an order is not a fill.

    `unfilled` is the stream that says the engine wanted to buy and could not —
    a stage a comparison must be able to see, and one that is easy to lose by
    reporting fills alone.
    """
    unfilled = getattr(path, "unfilled", None)
    fills = getattr(path, "fills", None)
    if unfilled is None and fills is None:
        return _absent("orders", "the run produced no path")
    rows = [{"at": str(one.date), "asset": str(one.ticker),
             "notional": str(one.notional), "reason": str(one.reason),
             "state": "filled"} for one in (fills or ())]
    rows += [{"at": str(one.date), "asset": str(one.ticker),
              "notional": str(one.notional), "reason": str(one.reason),
              "state": "unfilled"} for one in (unfilled or ())]
    return Stream("orders", True, tuple(rows))


def _flows(path, name: str, *, only_positive: bool = False) -> Stream:
    flows = getattr(path, "flows", None)
    if flows is None:
        return _absent(name, "the run produced no flow series")
    rows = []
    for when, value in flows.items():
        amount = float(value)
        if amount == 0.0 or (only_positive and amount <= 0.0):
            continue
        rows.append({"at": str(when.date() if hasattr(when, "date") else when),
                     "amount": f"{amount:.10f}"})
    return Stream(name, True, tuple(rows))


def _signals(spec, path) -> Stream:
    """Only a triggered plan has them.

    A scheduled plan does not compute a signal and never did; saying `()` would
    claim the stage ran and found nothing, which is a different run.
    """
    if spec.funding.kind != "event_triggered":
        return _absent("signals",
                       "this plan is scheduled, so no signal is evaluated")
    fills = getattr(path, "fills", None)
    if fills is None:
        return _absent("signals", "the run produced no path")
    return Stream("signals", True, tuple(
        {"at": str(one.date), "asset": str(one.ticker), "fired": True}
        for one in fills))


def _metrics(result) -> Stream:
    """Values *and* statuses. A rate that does not exist is not a zero.

    `money_weighted` is a result rather than a float precisely because "no
    admissible rate" and "the data does not determine one" are different
    answers, and a stream that flattened them to a number would report a run as
    matching when it did not.
    """
    if result is None:
        return _absent("metrics", "the engine returned no result")
    money = getattr(result, "money_weighted", None)
    rows = [{"metric": "time_weighted_final",
             "value": _last(getattr(result, "time_weighted", None)),
             "status": "COMPUTED" if _last(
                 getattr(result, "time_weighted", None)) is not None
             else "NOT_COMPUTED"},
            {"metric": "money_weighted",
             "value": (str(getattr(money, "rate", "")) if money is not None
                       else None),
             "status": str(getattr(money, "status", "ABSENT"))},
            {"metric": "periods_per_year",
             "value": str(getattr(result, "periods_per_year", "")),
             "status": "DECLARED"}]
    return Stream("metrics", True, tuple(rows))


def _last(series) -> Optional[str]:
    try:
        return f"{float(series.iloc[-1]):.10f}"
    except Exception:                                          # noqa: BLE001
        return None


def evaluate(strategy_spec, market_snapshot_id: str, *,
             evaluation_policy: str, engine_version: str,
             access, run_plan) -> EvaluationResult:
    """Price one specification against one snapshot, and say what it used.

    **Both the data and the engine are handed in, and neither is fetched.**
    That is the service shape rather than a convenience: an evaluator that
    resolved its own market data would be downloading prices inside the
    calculation, and one that imported the application's runner could not be
    deployed apart from it. The boundary test caught exactly that — the first
    version of this function reached into `workspace.run_boundary` for the
    engine and `market_data_for` for the frame, which is the web layer and the
    resolver, from the package that is supposed to move out first.

    It also closes a defect the injection was originally for: two resolutions
    of "the same" data are two deliveries, and a run citing one while consuming
    the other is what the delivery record exists to make impossible.

    Refuses rather than substituting. A snapshot that does not resolve, or
    resolves to a different id than the one asked for, ends the call: quietly
    evaluating against whatever came back is how a figure acquires provenance
    it does not have.
    """
    from ..mission.conventions import declared
    from ..mission.strategy_spec import to_scenario

    scenario = to_scenario(strategy_spec)
    frame = getattr(access, "frame", None)
    # `access_event`, which is what the accessor is called. Reading `.event`
    # returned None on every call and the result carried an empty snapshot
    # hash — an identity field that is blank is worse than one that is
    # missing, because it looks answered.
    event = getattr(access, "access_event", None)
    if frame is None:
        raise EvaluationRefused(
            "no market data was delivered for this evaluation, and a figure "
            "computed without it would be a figure about nothing")

    resolved_id = getattr(event, "snapshot_id", "") or ""
    if market_snapshot_id and resolved_id and market_snapshot_id != resolved_id:
        raise EvaluationRefused(
            f"asked for snapshot {market_snapshot_id!r} and the resolver "
            f"delivered {resolved_id!r}. Evaluating anyway would attribute this "
            "figure to data it was not computed from")

    run = run_plan(scenario, access)
    result = run.get("result")
    path = getattr(result, "path", None)

    named = declared()
    # The policy is an object, not a label. It used to be a bare string, which
    # named the data tier and said nothing about compounding, the annualisation
    # basis or the date the valuation was made as of — three conventions the
    # evaluator would otherwise have to infer, and one of which QuantLib keeps
    # as a mutable global that defaults to today.
    policy = evaluation_policy
    streams = {
        "eligible_sessions": _sessions(path),
        "signals": _signals(strategy_spec, path),
        "contributions": _flows(path, "contributions", only_positive=True),
        "orders": _orders(path),
        "fills": _fills(path),
        "cash_flows": _flows(path, "cash_flows"),
        "metrics": _metrics(result),
    }
    assert set(streams) == set(STREAMS), "a declared stream is not reported"

    return EvaluationResult(
        result_schema_version=RESULT_SCHEMA_VERSION,
        strategy_hash=strategy_spec.spec_hash,
        market_snapshot_hash=getattr(event, "frame_digest", "") or "",
        market_snapshot_id=resolved_id,
        evaluator=EVALUATOR,
        evaluator_version=EVALUATOR.split("@")[-1],
        engine_version=engine_version,
        conventions_version=str(named.get("vocabulary", "")),
        evaluation_policy=(policy.to_json() if hasattr(policy, "to_json")
                           else {"data_policy": str(policy)}),
        streams=streams,
        figures={"unavailable": run.get("unavailable"),
                 "strategy_not_executed": run.get("strategy_not_executed")},
        refusals=tuple(str(one) for one in (run.get("refusals") or ())))
