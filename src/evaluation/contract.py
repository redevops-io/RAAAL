"""The frozen wire contract: what a request is, and what a result is.

The in-process `evaluate()` signature and the `EvaluationResult` serialization
*are* the protocol. This module writes them and reads them back, and does
nothing else — no evaluation decision is taken here, and none may be.

**Frozen means frozen.** The extraction is a change of address. If the contract
moves during it, a conformance failure stops being interpretable: nobody can
say whether the remote evaluator computed something different or merely
described it differently. `CONTRACT_VERSION` is asserted by the conformance
suite on both sides for that reason.

**Read back, not merely written.** `EvaluationResult.to_json` existed and
nothing reversed it, which is enough for a log line and not enough for a
transport — a field that serializes and does not deserialize is a field that
silently stops crossing the wire. Every stream, every identity, and the
publish/refuse decision survive the round trip, and a test asserts it on a real
result rather than on a handmade one.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from .service import (EVALUATOR, RESULT_SCHEMA_VERSION, STREAMS,
                      EvaluationResult, Stream)

#: The shape of the request/response pair. Distinct from
#: `RESULT_SCHEMA_VERSION`, which describes the result alone: a transport can
#: change how it frames a call without the result changing, and conflating them
#: would make one version number mean two things.
CONTRACT_VERSION = "quantify-evaluation-contract@1"


def request_to_json(strategy_spec, market_snapshot_id: str, *,
                    evaluation_policy, engine_version: str) -> Dict[str, Any]:
    """Everything the evaluator is given, and nothing it could fetch itself."""
    return {
        "contract_version": CONTRACT_VERSION,
        "strategy_spec": strategy_spec.to_json(),
        "market_snapshot_id": market_snapshot_id,
        "evaluation_policy": (evaluation_policy.to_json()
                              if hasattr(evaluation_policy, "to_json")
                              else {"data_policy": str(evaluation_policy)}),
        "engine_version": engine_version,
    }


def request_from_json(payload: Mapping[str, Any]):
    """The request, rebuilt. Refuses a contract it does not implement.

    A version it does not know is not something to interpret leniently: the
    fields it recognises might mean something else under the newer rule, and
    evaluating anyway would produce a figure attributed to a request that was
    never made.
    """
    from ..mission.evaluation_policy import EvaluationPolicy
    from ..mission.strategy_spec import (Allocation, Conventions, Funding,
                                         StrategySpec, Trigger)

    version = payload.get("contract_version")
    if version != CONTRACT_VERSION:
        raise ValueError(
            f"this evaluator implements {CONTRACT_VERSION} and was sent "
            f"{version!r}. Evaluating a request under a contract it does not "
            "implement would attribute a figure to a question nobody asked")

    body = payload["strategy_spec"]
    funding = body["funding"]
    trigger = funding.get("trigger")
    spec = StrategySpec(
        objective=body["objective"],
        assets=tuple(body["assets"]),
        observed_assets=tuple(body["observed_assets"]),
        funding=Funding(
            kind=funding["kind"], amount=funding["amount"],
            cadence=funding.get("cadence", ""),
            day_rule=funding.get("day_rule", ""),
            execution_timing=funding.get("execution_timing", ""),
            trigger=(Trigger(subject=trigger["subject"],
                             window=int(trigger["window"]),
                             estimator=trigger["estimator"],
                             kind=trigger["kind"])
                     if trigger else None)),
        allocation=Allocation(
            assets=tuple(body["allocation"]["assets"]),
            weighting=body["allocation"]["weighting"],
            weights=dict(body["allocation"].get("weights") or {})),
        dividend_policy=body["dividend_policy"],
        sells_allowed=bool(body["sells_allowed"]),
        rebalancing_allowed=bool(body["rebalancing_allowed"]),
        rebalancing_cadence=body["rebalancing_cadence"],
        tax_treatment=body["tax_treatment"],
        evaluation_window=body["evaluation_window"],
        benchmarks=tuple(body["benchmarks"]),
        conventions=Conventions(**body["conventions"]),
        version=body["version"])

    policy_body = dict(payload["evaluation_policy"])
    policy = (EvaluationPolicy(**policy_body)
              if set(policy_body) >= {"compounding", "annualisation"}
              else policy_body)

    return (spec, payload["market_snapshot_id"], policy,
            payload["engine_version"])


def result_to_json(result: EvaluationResult) -> Dict[str, Any]:
    return {"contract_version": CONTRACT_VERSION, **result.to_json()}


def result_from_json(payload: Mapping[str, Any]) -> EvaluationResult:
    """The result, rebuilt, with `produced` preserved.

    The one field a careless reader would drop. `produced=False` with no rows
    and `produced=True` with no rows are different runs, and a transport that
    reconstructed both as an empty list would make the remote evaluator agree
    with the local one about a stage it never ran.
    """
    version = payload.get("contract_version")
    if version != CONTRACT_VERSION:
        raise ValueError(
            f"a result arrived under contract {version!r} and this caller "
            f"implements {CONTRACT_VERSION}")

    streams = {}
    for name, body in (payload.get("streams") or {}).items():
        streams[name] = Stream(
            name=body["name"], produced=bool(body["produced"]),
            rows=tuple(body.get("rows") or ()),
            absent_because=body.get("absent_because", ""))

    missing = set(STREAMS) - set(streams)
    if missing:
        raise ValueError(
            f"the result is missing {sorted(missing)}. A stream that does not "
            "arrive is not an empty stream, and treating it as one would hide "
            "a stage the remote evaluator stopped running")

    return EvaluationResult(
        result_schema_version=payload["result_schema_version"],
        strategy_hash=payload["strategy_hash"],
        market_snapshot_hash=payload["market_snapshot_hash"],
        market_snapshot_id=payload["market_snapshot_id"],
        evaluator=payload["evaluator"],
        evaluator_version=payload["evaluator_version"],
        engine_version=payload["engine_version"],
        conventions_version=payload["conventions_version"],
        evaluation_policy=payload["evaluation_policy"],
        streams=streams,
        figures=dict(payload.get("figures") or {}),
        refusals=tuple(payload.get("refusals") or ()))
