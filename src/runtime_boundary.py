"""RAAAL → runtime boundary adapter (dual identity).

RAAAL's internal intent model (the ``discovery_runtime`` / ``runtime_contracts``
IntentField ``VerifiedIntent`` with its native ``intent_hash``) is **not** migrated.
This adapter is the only thing that crosses the boundary to another runtime: it
turns a native intent into a canonical runtime artifact carrying TWO identities.

    RAAAL native intent
        ├─ native intent_hash                      (RAAAL owns intent semantics)
        └─ to_runtime_artifact()
              ├─ source_intent_hash = native hash  (carried verbatim, opaque)
              ├─ runtime_artifact_hash = rcv1…      (runtime-contracts owns identity)
              ├─ provenance
              └─ protocol/model payload             (the shared-form selection)

Two invariants this module upholds:

  * **Never recompute or mutate a source identity received from another runtime.**
    ``source_intent_hash`` is RAAAL's own native hash, passed through untouched.
  * **Never treat native hash and runtime-artifact hash as interchangeable.** They
    are different kinds; this module refuses to emit an artifact where they collide.

Requires runtime-contracts 0.3.x (the domain-facing ``content_hash`` that hashes a
float-carrying payload under the seal number policy). RAAAL owns extracting the
``payload`` from its native intent — this adapter never reaches into intent
semantics; it only assembles the wire artifact.
"""
from __future__ import annotations

from typing import Any, Mapping

import runtime_contracts as rc

RUNTIME_ARTIFACT_SCHEMA = "redevops/runtime-artifact"
RUNTIME_ARTIFACT_SCHEMA_VERSION = "0.1.0"
SOURCE_RUNTIME = "raaal"


class BoundaryError(ValueError):
    """A runtime artifact cannot be emitted without violating a boundary invariant."""


def to_runtime_artifact(*, source_intent_hash: str, payload: Mapping[str, Any],
                        produced_at: str = "",
                        payload_schema: str = "redevops/strategy-selection",
                        payload_schema_version: str = "0.1.0") -> dict:
    """Assemble a canonical runtime artifact from a native intent hash + a payload.

    `source_intent_hash` is RAAAL's native `intent.intent_hash`, carried verbatim.
    `payload` is the shared-form strategy selection RAAAL extracted from its native
    intent (allocation_method, target_allocation, …). The canonical identity is
    computed here, never assumed.
    """
    if not source_intent_hash:
        raise BoundaryError("a runtime artifact needs the native source_intent_hash")
    runtime_artifact_hash = rc.content_hash(dict(payload))
    if source_intent_hash == runtime_artifact_hash:
        # different kinds — a native hash is not a runtime-artifact hash
        raise BoundaryError(
            "source_intent_hash equals runtime_artifact_hash; the two identities "
            "must never be interchangeable")
    return {
        "schema": RUNTIME_ARTIFACT_SCHEMA,
        "schema_version": RUNTIME_ARTIFACT_SCHEMA_VERSION,
        "source_intent_hash": source_intent_hash,          # NATIVE, verbatim
        "runtime_artifact_hash": runtime_artifact_hash,    # canonical rcv1
        "protocol": {
            "canonicalization_version": rc.CANONICALIZATION_VERSION,
            "contract_version": rc.CONTRACT_VERSION,
            "payload_schema": payload_schema,
            "payload_schema_version": payload_schema_version,
            "digest_prefix": runtime_artifact_hash.split(":", 1)[0],
        },
        "provenance": {"source_runtime": SOURCE_RUNTIME, "produced_at": produced_at},
        "payload": dict(payload),
    }


def to_runtime_artifact_from_intent(native_intent: Any, payload: Mapping[str, Any],
                                    **kwargs) -> dict:
    """Convenience: pull `source_intent_hash` off a native intent's `intent_hash`.
    RAAAL still owns building `payload` from the intent's fields."""
    source = getattr(native_intent, "intent_hash", None)
    if not source:
        raise BoundaryError("native intent has no intent_hash to carry as source")
    return to_runtime_artifact(source_intent_hash=source, payload=payload, **kwargs)
