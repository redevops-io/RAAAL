"""Operational telemetry, kept apart from financial artifacts.

    conversation -> request -> trace -> span
                                     -> decision

Spans say **when**. Decisions say **why**. A span records that intent
classification ran and took 0.1ms; a decision records that this became
`AFTER_RESULTS` because the instruction matched a result-aware selection
pattern, and names the evidence. Six months from now the questions are "why
wasn't this benchmark included?" and "why did we ask another question?", and
neither is answerable from timings.

**This store is expendable by construction.** Financial artifacts live forever;
traces expire on a retention policy. Quantify keeps only a `trace_id` — a
reference that may dangle — so that deleting every trace changes nothing about
what a worksheet means or whether it can be replayed. A test asserts exactly
that by destroying the trace database and exercising the artifact path.

Without that rule, telemetry quietly becomes a second artifact store: something
starts reading a span to answer a question about a figure, and now the figure
depends on operational data that was always meant to be deletable.
"""
from .decisions import Decision, DecisionKind
from .trace_store import TraceStore
from .tracing import Recorder, Span, new_conversation_id, new_request_id

__all__ = ["Decision", "DecisionKind", "Recorder", "Span", "TraceStore",
           "new_conversation_id", "new_request_id"]
