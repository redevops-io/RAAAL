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

from .reader import Reading, ReadingSet, Schema


class CompilerReader:
    """Today's ordered regex tables, read as one opinion among several."""

    id = "quantify-compiler@1"

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
        return ReadingSet(reader_id=self.id, readings=tuple(readings),
                          unread=unread)


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

Return JSON only:

{"readings": [{"dimension": str, "value": str, "confidence": "0"-"1",
               "source_span": str, "note": str}],
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
    max_tokens: int = 2000
    timeout_s: float = 60.0

    @property
    def id(self) -> str:
        return f"{self.model}@{self.version}"

    def available(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

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

        unread = tuple(sorted(
            str(u) for u in (payload.get("unread") or ())
            if str(u) in schema.names))
        return ReadingSet(reader_id=self.id, readings=tuple(readings),
                          unread=unread)


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
    import re
    match = re.search(r"(\d{1,4})[- ]?(?:day|session)", text, re.IGNORECASE)
    return match.group(1) if match else None
