"""Quantify's two readers, behind one interface.

`CompilerReader` wraps today's regex compiler. It is the shadow comparator, not
a fallback: Phase 3 records what both readers saw and settles disagreements by
asking, and a comparator that only ran when the other failed would compare
nothing. `agentic-os`'s planner has exactly that shape today — a deterministic
`TemplatePlanner` reached only on exception — which is why its model layer can
never be contradicted.

`HostedReader` calls a provider. Governed by
`data/licensing/discovery-egress@1.yaml`: it sends the sentence, the schema and
the instructions, and nothing else. Not prices, not figures, not holdings, and
**not the capability manifest** — the last one for a correctness reason, since a
reader told what the engine runs renders everything else as the nearest runnable
thing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .reader import Reading, ReadingSet, RelationReading, Schema


class CompilerReader:
    """Today's ordered regex tables, read as one opinion among several."""

    id = "quantify-compiler@2"
    """Bumped from @1 when the moving-average fallback was removed. A reader
    whose behaviour changed under an unchanged id makes two runs look
    comparable when they are not — which is the whole reason `produced_by` is
    versioned."""

    #: compiler field name -> schema dimension name
    _RENAME = {"contribution_day_rule": "day_rule",
               "dividends": "dividend_policy"}

    def read(self, text: str, schema: Schema) -> ReadingSet:
        from ..mission.compiler import compile_scenario, parse

        try:
            parsed = parse(text)
            result = compile_scenario(
                text, name="discovery", version=1,
                benchmark_rule="benchmark-policy/public-default@1",
                parsed=parsed)
        except Exception as failure:                              # noqa: BLE001
            return ReadingSet(reader_id=self.id,
                              failed=f"{type(failure).__name__}: {failure}")

        readings = []
        for recognition in parsed.recognitions:
            name = self._RENAME.get(recognition.field, recognition.field)
            if name in schema.names:
                readings.append(Reading(
                    dimension=name, value=recognition.value,
                    source_span=str(recognition.span or "")))

        # Things the compiler reads but does not put in `recognitions`.
        #
        # Left out of the first version of this adapter, which made the model
        # look better than it is: four dimensions came back ONE_SIDED purely
        # because this wrapper did not go and get what the compiler already
        # knew. An unfair comparator manufactures evidence for whichever side
        # it was written more carefully for.
        if parsed.assets and "assets" in schema.names:
            readings.append(Reading(
                dimension="assets", value=", ".join(parsed.assets)))
        if parsed.observed and set(parsed.observed) != set(parsed.assets) \
                and "observed_assets" in schema.names:
            readings.append(Reading(dimension="observed_assets",
                                    value=", ".join(parsed.observed)))

        window = _window(text)
        if window and "evaluation_period" in schema.names:
            readings.append(Reading(dimension="evaluation_period", value=window))

        average = _moving_average(text)
        if average and "moving_average_window" in schema.names:
            readings.append(Reading(dimension="moving_average_window",
                                    value=average))

        # Dimensions it was asked about and did not answer. Reported rather
        # than left silent: a reader that says nothing looks like agreement.
        answered = {r.dimension for r in readings}
        unread = tuple(sorted(schema.names - answered))

        # No relations, ever, and that is a true report rather than a gap in
        # this adapter. The compiler has no representation for a role, a
        # per-member qualifier or a direction — which is precisely the finding
        # that produced schema@2. Emitting an empty tuple says "this reader
        # cannot see relationships"; inventing one from `assets` would hide the
        # thing the shadow run exists to measure.
        return ReadingSet(reader_id=self.id, readings=tuple(readings),
                          relations=(), unread=unread)


_INSTRUCTIONS = """\
You are reading one sentence written by someone describing an investment plan.
Your only job is to say what they meant. You are not deciding whether it can be
run — a separate system does that, and it will refuse things you report. Report
them anyway.

Rules that matter more than completeness:

1. Report only what the sentence says. If it does not say which account, do not
   guess: leave the dimension out and list it in `unread`.
2. Quote the words that carried each reading in `source_span`, verbatim from
   the sentence. If you cannot point at words, you are inferring, not reading.
3. Never normalise an instrument. "SPX ETF" is not "SPY". Report what was
   written.
4. `trigger_semantics` is the one that is almost never explicit. "when it
   crosses below" is a crossing_event — the day it first becomes true. "while
   it is below" or "whenever it is under" is a persistent_condition — every day
   it stays true. If the sentence genuinely does not distinguish them, leave it
   out; do not pick.
5. `evaluation_period` is canonical, not quoted: "over the past 5 years" is
   `trailing:5y`. Put the words in `source_span` as always.
6. Use a listed value where the dimension lists values. If the person clearly
   means something outside the list, report their words as the value rather
   than forcing the nearest listed one.

7. Some meaning is a *relationship*, not a value. Where the sentence gives a
   role to a participant, or a direction between participants, report it under
   `relations` instead of flattening it into a dimension. A plain list of
   holdings is not a relationship; "a core fund plus a 30% satellite" is,
   because the 30% belongs to one particular part.

Return JSON only:

{"readings": [{"dimension": str, "value": str, "confidence": "0"-"1",
               "source_span": str, "note": str}],
 "relations": [{"kind": str,
                "members": [{"role": str, "subject": str,
                             "qualifiers": {str: str}}],
                "attributes": {str: str},
                "source_span": str}],
 "unread": [str]}
"""


@dataclass
class HostedReader:
    """A provider-hosted model, behind the same interface as everything else.

    The model name is a parameter and the transport is one method, so a
    challenger provider or a local endpoint is a subclass, not a rewrite. The
    Phase 3 choice of `claude-sonnet-5` is about not making the reader the
    variable under test; it is not a structural commitment.
    """

    model: str = "claude-sonnet-5"
    version: str = "1"
    api_key_env: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 8000
    """Raised from 2000 after schema@2. The longer prompt produced longer
    answers and 16 of 144 replies were cut mid-JSON, which the parser reported
    as `unparseable output` — a reader failure indistinguishable, in the
    counts, from a reader that had nothing to say. Truncation is now reported
    as its own category by the runner as well, because a limit set by this
    file must never look like a finding about the model."""
    timeout_s: float = 60.0

    @property
    def id(self) -> str:
        return f"{self.model}@{self.version}"

    def available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    def _relations_prompt(self, schema: Schema) -> str:
        if not schema.relations:
            return ""
        lines = ["", "Relations:"]
        for r in schema.relations:
            lines.append(f"- {r.kind}: {r.describes}")
            lines.append(f"    roles: {', '.join(r.roles)}"
                         f"  (required: {', '.join(r.required_roles) or 'none'})")
            if r.qualifiers:
                for name, means in r.qualifiers.items():
                    lines.append(f"    member qualifier {name}: {means}")
            if r.attributes:
                for name, means in r.attributes.items():
                    lines.append(f"    attribute {name}: {means}")
            for example in r.examples:
                lines.append(f"    e.g. {example}")
        return "\n".join(lines)

    def _schema_prompt(self, schema: Schema) -> str:
        lines = []
        for d in schema.dimensions:
            line = f"- {d.name}: {d.describes}"
            if d.values:
                line += f"\n    one of: {', '.join(d.values)}"
            if d.examples:
                line += f"\n    e.g. {'; '.join(d.examples)}"
            lines.append(line)
        return "\n".join(lines)

    def _call(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ[self.api_key_env],
                                     timeout=self.timeout_s)
        message = client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}])
        return "".join(block.text for block in message.content
                       if getattr(block, "type", "") == "text")

    def read(self, text: str, schema: Schema) -> ReadingSet:
        if not self.available():
            return ReadingSet(reader_id=self.id,
                              failed=f"{self.api_key_env} is not set")

        prompt = (f"{_INSTRUCTIONS}\nDimensions:\n{self._schema_prompt(schema)}"
                  f"{self._relations_prompt(schema)}"
                  f"\n\nSentence:\n{text}\n")
        try:
            raw = self._call(prompt)
        except Exception as failure:                              # noqa: BLE001
            # A transport failure is not a reading. Recorded as `failed` so the
            # comparison drops it rather than scoring it as disagreement.
            return ReadingSet(reader_id=self.id,
                              failed=f"{type(failure).__name__}: {failure}")

        payload = _extract_json(raw)
        if payload is None:
            return ReadingSet(reader_id=self.id,
                              failed=f"unparseable output: {raw[:200]}")

        readings = []
        for one in payload.get("readings") or ():
            name = str(one.get("dimension", ""))
            if name not in schema.names:
                # A dimension nobody asked about. Dropped rather than carried:
                # the schema is the question, and an answer to a different
                # question is not evidence about this one.
                continue
            readings.append(Reading(
                dimension=name, value=one.get("value"),
                confidence=str(one.get("confidence", "1")),
                source_span=str(one.get("source_span", "") or ""),
                note=str(one.get("note", "") or "")))

        relations = []
        for one in payload.get("relations") or ():
            kind = str(one.get("kind", ""))
            spec = schema.relation(kind)
            if spec is None:
                # A relation kind nobody declared. Dropped for the same reason
                # an undeclared dimension is: the schema is the question, and
                # an answer to a different question is not evidence about this
                # one.
                continue
            members = tuple(
                (str(m.get("role", "")), str(m.get("subject", "")),
                 {str(k): str(v) for k, v in (m.get("qualifiers") or {}).items()})
                for m in (one.get("members") or ())
                if str(m.get("role", "")) in spec.roles)
            if not members:
                continue
            relations.append(RelationReading(
                kind=kind, members=members,
                attributes={str(k): str(v)
                            for k, v in (one.get("attributes") or {}).items()},
                source_span=str(one.get("source_span", "") or "")))

        unread = tuple(sorted(
            str(u) for u in (payload.get("unread") or ())
            if str(u) in schema.names))
        return ReadingSet(reader_id=self.id, readings=tuple(readings),
                          relations=tuple(relations), unread=unread)


@dataclass
class OpenAIReader(HostedReader):
    """The same reader, against a different provider.

    This is the subclass `HostedReader` was written to make possible: the model
    is a parameter, the transport is one method, and everything above it —
    prompt construction, JSON extraction, the reading vocabulary, the failure
    categories — is shared. Nothing about *what a reading means* changes with
    the provider, which is the property that makes swapping one a measurement
    rather than a rewrite.

    **The model is pinned to a dated snapshot on purpose.** `gpt-4.1` is a
    moving alias; `gpt-4.1-2025-04-14` is not. The reason is the one the drift
    lane exists for: an unpinned model turns "the answer changed" into a
    question nobody can answer, because the reader moved or the provider did
    and no artifact says which.

    **Why the id carries the whole model name.** Recordings are keyed
    `reader_id\ttext`, so the id is what keeps two providers' answers from
    being confused for one another. A short id like `gpt@1` would let a
    re-record silently overwrite readings taken from a different model —
    exactly the collision the Anthropic reader's `claude-sonnet-5@1` avoids.
    """

    # `@dataclass` on the subclass is load-bearing. Without it the parent's
    # generated `__init__` runs and assigns *its* defaults, so these three
    # class attributes are silently overwritten and `id` reports
    # `claude-sonnet-5@1`. That is not a cosmetic bug: recordings are keyed by
    # reader id, so an OpenAI reader announcing itself as the Anthropic one
    # would have written its answers over the other model's — two providers'
    # readings merged into one file with nothing saying so.
    model: str = "gpt-4.1-2025-04-14"
    version: str = "1"
    api_key_env: str = "OPENAI_API_KEY"

    def _call(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ[self.api_key_env],
                        timeout=self.timeout_s)
        # `max_completion_tokens`, not `max_tokens`: the older parameter is
        # rejected by current models rather than ignored, which would have
        # surfaced as a reader failure on every sentence.
        reply = client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}])
        return reply.choices[0].message.content or ""


def configured_hosted_reader():
    """The hosted reader this deployment declared, built once and named once.

    This selection existed in four places — the pilot route, the recorder, the
    pre-Lean gate — and in a fifth that never got it: `drift_lane.py` built
    `HostedReader()` directly, so after the provider moved to OpenAI the lane
    checked for `ANTHROPIC_API_KEY`, found none, and refused to run in CI with
    the OpenAI key sitting in its environment.

    That is the same defect as the workflow pointing at the wrong provider,
    one file over, and it cost a dispatch to find. A rule duplicated four
    times is a rule that will be applied three times.
    """
    from ..deploy.context import PROVIDER_DEFAULT_MODEL, ParserProvider, current

    model = current().model
    cls = (OpenAIReader if model.provider is ParserProvider.OPENAI
           else HostedReader)
    return cls(model=model.model or PROVIDER_DEFAULT_MODEL[model.provider],
               max_tokens=model.max_tokens)


def _extract_json(raw: str) -> Optional[dict]:
    """The outermost JSON object, fences or prose notwithstanding."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _window(text: str) -> Optional[str]:
    """The evaluation period, as the compiler detects it."""
    from ..mission.time_window import detect

    try:
        found = detect(text)
    except Exception:                                             # noqa: BLE001
        return None
    if found is None:
        return None
    # Canonical, not as written. The first version returned the user's words,
    # and 15 of 21 contested fields in the first shadow run were "for the past
    # 5 years" against "the past 5 years" — the same window, two span
    # boundaries. A dimension compared as prose makes every reader disagree
    # about where a phrase starts.
    return _canonical_window(found)


def _canonical_window(found) -> Optional[str]:
    kind = str(getattr(getattr(found, "kind", ""), "value", "") or "")
    years = getattr(found, "years", None)
    months = getattr(found, "months", None)

    if kind in ("trailing", "rolling"):
        if years:
            return f"{kind}:{years}y"
        if months:
            return f"{kind}:{months}m"
        # Recognised as this kind and not sized — "each month for the past"
        # truncates before the duration. Reported rather than dropped: `None`
        # here would read as "the compiler never looked", which is a different
        # and much more flattering claim than "it saw one and could not size
        # it".
        return f"{kind}:unresolved"
    if kind in ("since", "until", "explicit_range", "event_relative"):
        # Recognised and not reducible to a duration. Reported as the kind so a
        # disagreement about *which kind of window* still shows up, rather than
        # vanishing into None and reading as "the compiler did not look".
        return f"{kind}:unresolved"
    return None


def _moving_average(text: str) -> Optional[str]:
    """The moving-average length the compiler recognises, in sessions."""
    from ..mission.compiler import parse

    try:
        parsed = parse(text)
    except Exception:                                             # noqa: BLE001
        return None
    for recognition in parsed.recognitions:
        if recognition.field == "moving_average_window":
            return recognition.value

    # The compiler's own function, not an approximation of it.
    #
    # A regex fallback used to live here — `(\d+)[- ]?(day|session)` — written
    # when this adapter was made "fair" to the compiler. It read "90" out of
    # "hold annual bonus 90 days sized on event", which is a holding period and
    # not an average at all, and it produced every one of the moving-average
    # agreements in the earlier runs: the comparator was supplying a capability
    # to one side and then scoring the two as agreeing about it.
    #
    # Being generous to a reader is the same defect as being stingy with it.
    # Both let the comparator decide the result. `compiler.moving_average_window`
    # is what the compiler actually does.
    from ..mission.compiler import moving_average_window

    try:
        found = moving_average_window(text)
    except Exception:                                             # noqa: BLE001
        return None
    return str(found) if found else None
