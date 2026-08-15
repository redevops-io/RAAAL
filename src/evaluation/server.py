"""The evaluation service, as an application that answers one question.

    POST /evaluate   StrategySpec + snapshot id + policy + engine -> result
    GET  /health     serving, or not
    GET  /version    what this build is, and which contracts it implements

Everything here is transport. The route deserializes, calls `evaluate`, and
serializes — it takes no evaluation decision, and the conformance suite is
pointed through it precisely so that claim is tested rather than asserted.

**Refusals are 422, failures are 500, and the difference matters.** "This plan
cannot be evaluated" is an answer about the plan; "the evaluator broke" is not.
A caller that could not tell them apart would either retry a refusal forever or
report a broken service as a bad strategy.

**The build identity is on `/version` for the same reason the deployment proof
exists.** A figure computed here is attributable only if the evaluator can say
which evaluator it was — the exact QuantLib build included, because the
conventions are its names and a different build could mean different dates.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .contract import CONTRACT_VERSION, request_from_json, result_to_json
from .service import EVALUATOR, RESULT_SCHEMA_VERSION, EvaluationRefused


def build_identity(build: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """What this evaluator is, and which contracts it implements.

    The commit and the image digest are *given*, not read from the environment.
    One module in this system resolves the deployment, and an evaluator reading
    `os.environ` for its own identity would be a second — which is how two
    parts of one deployment end up disagreeing about which build they are. It
    is also the same rule this whole file follows: the service is handed what
    it does not decide.

    The vocabulary is read rather than supplied, because that one *is* the
    evaluator's own fact: it is the QuantLib actually imported in this process,
    and a deployment claiming a different one would be describing a library it
    is not running.
    """
    from ..mission import conventions

    supplied = dict(build or {})
    return {
        "evaluator": EVALUATOR,
        "contract_version": CONTRACT_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "conventions_version": conventions.declared().get("vocabulary", ""),
        "quantlib_available": conventions.AVAILABLE,
        "build_commit": supplied.get("build_commit", ""),
        "image_digest": supplied.get("image_digest", ""),
    }


def create_app(*, resolve_access, run_plan, build=None):
    """The service, with its data resolver and engine injected.

    Both are arguments because the evaluator does not choose them: which
    delivery a snapshot id resolves to is the market-data service's question,
    and which engine runs is the deployment's. A service that reached for
    either would be deciding something the contract says it is given.
    """
    # `FastAPI` and `Request` are imported at module level, not here. With
    # `from __future__ import annotations` every annotation is a string that
    # FastAPI resolves against the *module* namespace, so a `Request` imported
    # into this function was unresolvable — and FastAPI fell back to treating
    # the body as a missing query parameter, which surfaced as every request
    # being refused for a reason about the strategy.
    from .service import evaluate

    app = FastAPI(title="Quantify evaluation", version=CONTRACT_VERSION)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/version")
    def version():
        return build_identity(build)

    @app.post("/evaluate")
    async def evaluate_route(request: Request):
        payload = await request.json()
        correlation = (request.headers.get("x-idempotency-key")
                       or request.headers.get("x-request-id") or "")
        try:
            spec, snapshot_id, policy, engine = request_from_json(payload)
        except ValueError as wrong_contract:
            # A contract mismatch is not a plan that cannot be evaluated. 400,
            # so a caller does not read "we do not speak your version" as "your
            # strategy is unsupported" and go looking at the strategy.
            return JSONResponse(status_code=400,
                                content={"detail": str(wrong_contract),
                                         "correlation_id": correlation})

        try:
            access = resolve_access(spec, snapshot_id)
            result = evaluate(spec, snapshot_id, evaluation_policy=policy,
                              engine_version=engine, access=access,
                              run_plan=run_plan)
        except EvaluationRefused as refused:
            return JSONResponse(status_code=422,
                                content={"detail": str(refused),
                                         "correlation_id": correlation})

        return JSONResponse(
            status_code=200,
            content=result_to_json(result),
            headers={"x-correlation-id": correlation} if correlation else None)

    return app
