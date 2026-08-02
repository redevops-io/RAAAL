"""Recording spans and decisions around work that must not depend on them.

    with recorder.span("intent_planning") as span:
        ...
        span.set(effect=intent.edit_effect.value)

**Telemetry never breaks the thing it observes.** Every write here is wrapped:
a full disk, a locked trace database or a missing file must not turn a
successful worksheet edit into a failed request. The financial path is
authoritative and the trace is a by-product, so the failure mode has to be a
lost trace rather than a lost edit.

That cuts the other way too. A silently swallowed error is how observability
rots — so failures are counted on the recorder and surfaced by
`Recorder.failures`, and a test asserts the count moves rather than the store
merely staying quiet.
"""
from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

from .decisions import Decision, DecisionKind, new_decision_id


def new_conversation_id() -> str:
    return f"conv-{uuid.uuid4().hex[:16]}"


def new_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:16]}"


def new_trace_id() -> str:
    return f"trace-{uuid.uuid4().hex[:16]}"


@dataclass
class Span:
    span_id: str
    trace_id: str
    name: str
    started_at: str
    parent_id: Optional[str] = None
    duration_ms: Optional[float] = None
    status: str = "OK"
    attributes: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def set(self, **attributes: Any) -> "Span":
        """Attach structured fields.

        Values must be structured or hashed. Raw instruction text, prompts and
        completions do not belong here — they may carry holdings, salary,
        employer or vesting detail, and this store outlives the request but not
        the retention window.
        """
        self.attributes.update(attributes)
        return self


class Recorder:
    """Records one request's spans and decisions.

    Constructed per request. `store` may be None, which disables recording
    entirely and is the configuration every financial test runs under — the
    behaviour of the system must not vary with whether it is being watched.
    """

    def __init__(self, store=None, *, trace_id: Optional[str] = None,
                 conversation_id: Optional[str] = None,
                 request_id: Optional[str] = None, tenant: str = "",
                 worksheet_id: Optional[str] = None,
                 clock=None) -> None:
        self.store = store
        self.trace_id = trace_id or new_trace_id()
        self.conversation_id = conversation_id or new_conversation_id()
        self.request_id = request_id or new_request_id()
        self.tenant = tenant
        self.worksheet_id = worksheet_id
        self.failures = 0
        """Telemetry writes that did not land. Counted rather than raised, and
        exposed rather than swallowed."""

        self._clock = clock or (lambda: __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(timespec="seconds"))
        self._stack: List[str] = []
        self.produced: List[str] = []

    # ---- lifecycle -------------------------------------------------------

    def start(self) -> "Recorder":
        self._guard(lambda: self.store.start_trace(
            trace_id=self.trace_id, conversation_id=self.conversation_id,
            request_id=self.request_id, tenant=self.tenant,
            worksheet_id=self.worksheet_id, started_at=self._clock()))
        return self

    def finish(self, status: str = "OK") -> None:
        self._guard(lambda: self.store.end_trace(
            self.trace_id, ended_at=self._clock(), status=status,
            produced=self.produced))

    def produced_artifact(self, reference: str) -> None:
        """Note what this request produced, so a trace can be found from it."""
        if reference and reference not in self.produced:
            self.produced.append(reference)

    # ---- spans -----------------------------------------------------------

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        current = Span(span_id=f"span-{uuid.uuid4().hex[:16]}",
                       trace_id=self.trace_id, name=name,
                       started_at=self._clock(),
                       parent_id=self._stack[-1] if self._stack else None,
                       attributes=dict(attributes))
        self._stack.append(current.span_id)
        began = time.perf_counter()
        try:
            yield current
        except Exception as failure:                              # noqa: BLE001
            current.status = "ERROR"
            # The class, not the message. A message can quote the input that
            # caused it, and that input is the thing this store must not hold.
            current.error = type(failure).__name__
            raise
        finally:
            current.duration_ms = (time.perf_counter() - began) * 1000
            self._stack.pop()
            self._guard(lambda: self.store.record_span(current))

    # ---- decisions -------------------------------------------------------

    def decide(self, kind: DecisionKind, *, outcome: str, reason: str,
               evidence_refs: Sequence[str] = (),
               confidence: Optional[float] = None,
               alternatives_considered: Sequence[str] = ()) -> Decision:
        """Record why the runtime chose what it chose.

        `confidence` stays None unless a source genuinely reports one. A
        deterministic rule match has no confidence; it has a rule.
        """
        decision = Decision(
            decision_id=new_decision_id(), trace_id=self.trace_id, kind=kind,
            outcome=outcome, reason=reason, evidence_refs=tuple(evidence_refs),
            confidence=confidence,
            alternatives_considered=tuple(alternatives_considered),
            at=self._clock())
        self._guard(lambda: self.store.record_decision(decision))
        return decision

    # ---- internals -------------------------------------------------------

    def _guard(self, write) -> None:
        """Perform a telemetry write, or count its failure and carry on."""
        if self.store is None:
            return
        try:
            write()
        except Exception:                                          # noqa: BLE001
            # Deliberately broad. Anything this store can raise — a locked
            # database, a full disk, a deleted file — must cost a trace and
            # never an edit.
            self.failures += 1
