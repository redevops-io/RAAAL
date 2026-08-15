"""Two ways to reach one evaluator, and no evaluation in either.

    LocalEvaluator   calls `evaluate()` in this process
    HttpEvaluator    posts the same request to a service and reads the result

The application holds one of these and cannot tell which. That is the exit
condition stated as a type: `workspace` knows the evaluation *contract* and
nothing about the implementation on the other side of it.

**The HTTP half makes no decisions.** It serializes, posts, checks the status,
deserializes. It does not retry on a refusal, does not substitute a default for
a missing field, and does not interpret a partial result — every one of those
would be an evaluation decision taken by a transport, which is the failure mode
that makes a remote evaluator untrustworthy in a way nobody can see.

**An idempotency key, from the identities rather than from a clock.** The same
specification, snapshot and engine name the same computation, so a retry can be
recognised as one. A key with a timestamp in it would make every retry a new
request and quietly turn a network hiccup into two different runs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .contract import (CONTRACT_VERSION, request_to_json, result_from_json)
from .service import EvaluationRefused, EvaluationResult, evaluate


def idempotency_key(strategy_spec, market_snapshot_id: str,
                    engine_version: str) -> str:
    """What names this computation, so a retry is a retry.

    Deterministic over the identities and nothing else. `Date.now()` in a key
    is how a retried request becomes a second run, and two runs of one question
    is precisely what the whole provenance chain exists to make detectable.
    """
    body = "|".join([CONTRACT_VERSION, strategy_spec.spec_hash,
                     market_snapshot_id or "", engine_version])
    return "idem1:" + hashlib.sha256(body.encode()).hexdigest()[:32]


@dataclass(frozen=True)
class LocalEvaluator:
    """The evaluator in this process. The reference implementation."""

    run_plan: Callable
    name: str = "local"

    def evaluate(self, strategy_spec, market_snapshot_id: str, *,
                 evaluation_policy, engine_version: str,
                 access) -> EvaluationResult:
        return evaluate(strategy_spec, market_snapshot_id,
                        evaluation_policy=evaluation_policy,
                        engine_version=engine_version, access=access,
                        run_plan=self.run_plan)


@dataclass(frozen=True)
class HttpEvaluator:
    """The evaluator somewhere else. Same call, same result, no decisions here.

    `access` is accepted and *not sent*. The remote evaluator resolves its own
    delivery from the snapshot id — that is the point of the id — and shipping a
    price frame over the wire would make the request enormous and the snapshot
    identity decorative. It stays in the signature so the two evaluators are
    substitutable, which is the property the application depends on.
    """

    post: Callable
    """`(url, json, headers) -> (status, payload)`. Injected rather than a
    client built here, so this module imports no HTTP library and a test can
    drive it without a socket."""

    url: str
    name: str = "http"

    def evaluate(self, strategy_spec, market_snapshot_id: str, *,
                 evaluation_policy, engine_version: str,
                 access=None) -> EvaluationResult:
        body = request_to_json(strategy_spec, market_snapshot_id,
                               evaluation_policy=evaluation_policy,
                               engine_version=engine_version)
        status, payload = self.post(
            self.url, body,
            {"x-idempotency-key": idempotency_key(
                strategy_spec, market_snapshot_id, engine_version),
             "x-contract-version": CONTRACT_VERSION})

        if status == 422:
            # The evaluator's own refusal, carried across as a refusal rather
            # than as a transport error. A caller must be able to tell "the
            # evaluator will not answer this" from "the evaluator could not be
            # reached", because only one of them is about the plan.
            raise EvaluationRefused(str((payload or {}).get("detail", "")))
        if status != 200:
            raise EvaluationUnreachable(
                f"the evaluation service answered {status}. No figure is "
                "produced: a result assembled from a failed call would be a "
                "figure with no computation behind it")
        return result_from_json(payload)


class EvaluationUnreachable(RuntimeError):
    """The service could not be reached, which is not a statement about a plan."""
