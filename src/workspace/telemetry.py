"""Website / API funnel telemetry (§10 of the public strategy-lab plan).

Measures the public journey — research viewed, evaluator opened, prompt
submitted, clarified, evaluated, saved, signed in, advisor page viewed — without
making login mandatory and without ever recording a strategy prompt.

Two rules shape it:

**Best-effort and non-blocking.** A telemetry write must never be the reason a
request fails. Every emission is wrapped; a raising emitter is swallowed and the
request proceeds. Telemetry is the expendable half — losing a count is a better
outcome than losing the page — the same rule `pilot_events.record` already states
for the study instrumentation.

**No raw strategy text (§10).** The event payload carries route, outcome, ids,
hashes, latencies and counts. It never carries the prompt. `prompt_digest` turns
a sentence into a short hash + length so the funnel can tell distinct prompts
apart without storing what anybody wrote, and `emit` drops any field that looks
like prose as a backstop against a careless caller.

**Where it lives.** A process-local ring buffer plus counters — light enough to
run in front of every request, read by a test or an operational surface through
`snapshot`. The durable study record is `pilot_events` (a database table with a
fixed vocabulary); this is the cheap website funnel beside it, deliberately not
the same store so a funnel event never needs a schema migration.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Deque, Dict, List, Mapping, Optional

LOG = logging.getLogger(__name__)

# --- the §10 funnel vocabulary --------------------------------------------

RESEARCH_VIEW = "research_view"
EVALUATOR_OPENED = "evaluator_opened"
PROMPT_SUBMITTED = "prompt_submitted"
CLARIFICATION_REQUESTED = "clarification_requested"
CLARIFICATION_COMPLETED = "clarification_completed"
EVALUATION_COMPLETED = "evaluation_completed"
EVALUATION_FAILED = "evaluation_failed"
EVALUATION_ABSTAINED = "evaluation_abstained"
EVALUATION_RERUN = "evaluation_rerun"
SAVE_CLICKED = "save_clicked"
AUTH_COMPLETED = "auth_completed"
PLAN_SAVED = "plan_saved"
ADVISOR_VIEWED = "advisor_viewed"

FUNNEL = frozenset({
    RESEARCH_VIEW, EVALUATOR_OPENED, PROMPT_SUBMITTED,
    CLARIFICATION_REQUESTED, CLARIFICATION_COMPLETED,
    EVALUATION_COMPLETED, EVALUATION_FAILED, EVALUATION_ABSTAINED,
    EVALUATION_RERUN, SAVE_CLICKED, AUTH_COMPLETED, PLAN_SAVED,
    ADVISOR_VIEWED,
})

#: Field names a payload must never carry — a backstop against a caller passing
#: the sentence in by habit. `emit` strips these before anything is recorded.
_FORBIDDEN_FIELDS = frozenset({
    "prompt", "describe", "text", "body", "sentence", "answer", "answers"})


# --- the process-local sink -----------------------------------------------

#: How many recent events to keep. A ring, not a log: the point is a live funnel
#: an operator or a test can read, not an archive.
_RING_SIZE = 2000

_LOCK = threading.Lock()
_EVENTS: Deque[Dict[str, Any]] = deque(maxlen=_RING_SIZE)
_COUNTS: Dict[str, int] = {}


def prompt_digest(text: Optional[str]) -> Dict[str, Any]:
    """A prompt reduced to what the funnel may keep: a hash and a length.

    Distinct prompts get distinct hashes, so conversion can be followed across a
    session without the words ever being stored. `None`/empty yields empty fields
    rather than a hash of nothing, so an absent prompt is legible as absent.
    """
    if not text:
        return {"prompt_sha": "", "prompt_len": 0}
    return {"prompt_sha": sha256(text.encode("utf-8")).hexdigest()[:16],
            "prompt_len": len(text)}


def _scrub(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """Drop any prose-bearing field. The last line before a value is recorded."""
    return {k: v for k, v in fields.items() if k not in _FORBIDDEN_FIELDS}


def emit(event: str, *, route: str = "", outcome: str = "",
         participant: str = "", latency_ms: Optional[float] = None,
         **fields: Any) -> None:
    """Record one funnel event. Best-effort; never raises into a request.

    The payload is route/outcome/ids/hashes/latency/counts only. Any field named
    like prose is stripped by `_scrub`, so even a careless caller cannot land a
    prompt in the funnel. Everything is inside one `try`: a telemetry failure is
    swallowed and the caller proceeds, because the request matters and the count
    does not.
    """
    try:
        if event not in FUNNEL:
            # A typo'd event name is a defect, but not one worth failing a
            # request over; note it and move on.
            LOG.warning("telemetry: unknown event %r ignored", event)
            return
        record: Dict[str, Any] = {
            "event": event,
            "route": route,
            "outcome": outcome,
            "participant": participant,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        if latency_ms is not None:
            record["latency_ms"] = round(float(latency_ms), 3)
        record.update(_scrub(fields))
        with _LOCK:
            _EVENTS.append(record)
            _COUNTS[event] = _COUNTS.get(event, 0) + 1
    except Exception:  # noqa: BLE001 - telemetry must never fail a request
        LOG.exception("telemetry emission failed; ignoring (best-effort)")


# --- reading the funnel back ----------------------------------------------


def events(event: Optional[str] = None) -> List[Dict[str, Any]]:
    """A copy of the recent events, optionally filtered to one kind."""
    with _LOCK:
        out = list(_EVENTS)
    return [e for e in out if event is None or e["event"] == event]


def counts() -> Dict[str, int]:
    with _LOCK:
        return dict(_COUNTS)


def save_without_recompute_rate() -> Optional[float]:
    """The share of saves that bound the exact artifact without recomputing it.

    Target 100% (§10), and structurally so: the exact-save path
    (`_bind_the_exact_artifact`) binds a content-addressed review and never runs
    the parser or evaluator, and every `plan_saved` it emits carries
    `recomputed=False`. `None` when no plan has been saved yet — a rate over an
    empty denominator is not 100%, it is undefined, and reporting it as 1.0 would
    invent a success that never happened.
    """
    saved = events(PLAN_SAVED)
    if not saved:
        return None
    clean = sum(1 for e in saved if e.get("recomputed") is False)
    return clean / len(saved)


def conversion() -> Dict[str, int]:
    """The anonymous-to-save funnel counters §10 asks for, as raw counts.

    Ratios are left to the reader: `saved / clicked` is the conversion, and both
    numbers travel so a zero denominator is visible rather than dividing to a NaN
    somebody has to explain.
    """
    c = counts()
    return {
        "evaluator_opened": c.get(EVALUATOR_OPENED, 0),
        "prompt_submitted": c.get(PROMPT_SUBMITTED, 0),
        "save_clicked": c.get(SAVE_CLICKED, 0),
        "auth_completed": c.get(AUTH_COMPLETED, 0),
        "plan_saved": c.get(PLAN_SAVED, 0),
    }


def snapshot() -> Dict[str, Any]:
    """Everything a metrics surface or a test needs, in one read."""
    return {
        "counts": counts(),
        "conversion": conversion(),
        "save_without_recompute_rate": save_without_recompute_rate(),
        "recent": events(),
    }


def reset() -> None:
    """Forget every event. For tests and a clean process boundary."""
    with _LOCK:
        _EVENTS.clear()
        _COUNTS.clear()
