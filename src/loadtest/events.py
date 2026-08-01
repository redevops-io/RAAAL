"""Synthetic runtime-event streams for HarnessBench.

The columns are the event schema from the load-test plan. Generation is
deterministic in the seed, so the canonical and Polars evaluators can be run on
byte-identical input and any difference between them is theirs.

Events are *shaped* like a real fleet rather than uniform: a few tenants produce
most of the traffic, mission histories vary in length, and a small fraction of
events are decision denials. A uniform stream makes every group-by look the same
size and hides exactly the skew that decides whether a query is fast.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

EVENT_COLUMNS = (
    "event_id", "canonical_hash", "schema_version", "event_type", "tenant_id",
    "mission_id", "investigation_id", "session_id", "parent_event_id",
    "sequence_number", "actor", "source_agent", "capability_id", "artifact_id",
    "artifact_type", "visibility", "effect", "decision", "rule_id",
    "program_hash", "occurred_at", "ingested_at", "duration_us",
)

_TYPES = ("mission.transition", "artifact.dereference", "context.materialized",
          "capability.preflight", "verification.completed", "secret.read",
          "network.egress", "ledger.append")
_DECISIONS = ("GRANTED", "GRANTED", "GRANTED", "GRANTED", "DENIED_TENANT",
              "DENIED_STALE")
_EFFECTS = ("READ", "READ", "READ", "WRITE", "NONE")

#: Epoch seconds for 2026-01-01, so timestamps are stable without a clock read.
_EPOCH = 1_767_225_600


def generate(count: int, *, seed: int = 7, tenants: int = 64,
             missions: int = 512) -> List[Dict[str, Any]]:
    """`count` events across `missions` missions.

    Mission lengths follow a long tail — most short, a few very long — because
    replay cost is decided by the longest history, not the mean, and a uniform
    stream would make a grouped replay look uniformly cheap.
    """
    rng = random.Random(seed)
    # A few tenants own most of the traffic.
    weights = [rng.paretovariate(1.4) for _ in range(tenants)]
    total = sum(weights)
    tenant_cdf, running = [], 0.0
    for w in weights:
        running += w / total
        tenant_cdf.append(running)

    def pick_tenant() -> int:
        r = rng.random()
        for index, edge in enumerate(tenant_cdf):
            if r <= edge:
                return index
        return tenants - 1

    sequences: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for index in range(count):
        tenant = pick_tenant()
        mission = f"mission/{rng.randrange(missions):05d}"
        sequences[mission] = sequences.get(mission, -1) + 1
        sequence = sequences[mission]
        event_type = rng.choice(_TYPES)
        occurred = _EPOCH + index * 37 + rng.randrange(0, 29)
        rows.append({
            "event_id": f"evt-{index:09d}",
            "canonical_hash": f"rcv1:{index:064x}"[:69],
            "schema_version": "0.1",
            "event_type": event_type,
            "tenant_id": f"tenant-{tenant:03d}",
            "mission_id": mission,
            "investigation_id": (f"investigation/{rng.randrange(64):04d}"
                                 if rng.random() < 0.25 else None),
            "session_id": f"session-{rng.randrange(2048):05d}",
            "parent_event_id": (f"evt-{index - 1:09d}" if sequence else None),
            "sequence_number": sequence,
            "actor": rng.choice(("pilot", "discovery", "scheduler", "user")),
            "source_agent": rng.choice(("control-plane", "mission-runtime")),
            "capability_id": f"cap-{rng.randrange(24):02d}",
            "artifact_id": f"artifact/{rng.randrange(4096):05d}@{rng.randrange(1, 5)}",
            "artifact_type": rng.choice(("evidence", "run", "methodology")),
            "visibility": rng.choice(("PUBLIC", "INTERNAL", "PRIVATE")),
            "effect": rng.choice(_EFFECTS),
            "decision": rng.choice(_DECISIONS),
            "rule_id": f"rule-{rng.randrange(32):02d}",
            "program_hash": "rcv1:program",
            "occurred_at": occurred,
            "ingested_at": occurred + rng.randrange(0, 5),
            "duration_us": max(1, int(rng.lognormvariate(7.5, 1.1))),
        })
    return rows
