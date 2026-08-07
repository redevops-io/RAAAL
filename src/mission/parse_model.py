"""Stage 1 with a language model, and the quarantine that makes it safe.

The compiler's contract is that stages 2–10 are deterministic and would produce
the same scenario from the same parse a year from now. A model in stage 1 does
not weaken that — provided everything it returns is treated as **a claim about
the text**, checked against the text, rather than as a decision.

    model  ->  proposed recognitions  ->  verify  ->  ParsedUtterance
                                          ^^^^^^
                          deterministic, and the only thing downstream trusts

Three checks do the work, and each closes a distinct failure:

    field and value in the vocabulary   the model cannot invent a semantic it
                                        was never given. The vocabulary is
                                        *derived from the deterministic rules*,
                                        so it cannot drift out of step with them.

    span appears verbatim in the text   the model cannot support a reading by
                                        quoting words the user never wrote. This
                                        is the check that catches fabrication,
                                        and it is why the span is required.

    assets and amounts appear in the    a ticker or a figure the model supplied
    text                                on its own is the one error that silently
                                        prices the wrong security or scales
                                        every number.

What the model is for is **coverage of phrasing**, not new meaning. "whenever it
dips under its average" and "while it trades below the 200-day" are the same
plan; the regexes catch one. Anything the model proposes outside the vocabulary
becomes an unresolved question, exactly as an unmatched phrase already does.

The model is never required. No key, no network, a timeout, malformed JSON, or a
refusal all fall back to the deterministic parse, because a front door that stops
working when an API does is not a front door.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Set

from .compiler import (
    _AMOUNT,
    _CADENCE,
    _RULES,
    TRIGGER_SEMANTICS_VALUES,
    _TEMPLATE_HINTS,
    AMBIGUOUS_NAMES,
    ParsedUtterance,
    Recognition,
    parse as parse_deterministic,
)

logger = logging.getLogger(__name__)

PARSER_ID = "quantify/stage1"
PARSER_VERSION = "2"
DEFAULT_MODEL = "claude-sonnet-5"


def _vocabulary() -> Dict[str, Set[str]]:
    """The closed set of things stage 1 may recognise.

    Derived from the deterministic rules rather than written out again. A second
    hand-maintained list is a list that goes stale, and a stale one here would
    either reject a legitimate value or admit a retired one.
    """
    vocab: Dict[str, Set[str]] = {}
    for field_name, value, _pattern in _RULES:
        vocab.setdefault(field_name, set()).add(value)
    vocab["cadence"] = {name for name, _pattern in _CADENCE}
    # Resolved by precedence rather than by the flat table, so it has no entry
    # there — and deriving the vocabulary from `_RULES` alone silently stopped
    # the model proposing either value for it. Imported rather than restated:
    # a second copy of this pair is the stale list the docstring warns about.
    vocab["trigger_semantics"] = set(TRIGGER_SEMANTICS_VALUES)
    return vocab


VOCABULARY: Mapping[str, Set[str]] = _vocabulary()

#: Read from the text by the deterministic extractor, so a model proposal about
#: them is redundant at best. Never taken from the model: an amount is the one
#: value that rescales every figure downstream.
_TEXT_DERIVED = frozenset({"amount"})

_WHITESPACE = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z0-9.$,]+")


def _normalized(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().lower()


class ModelClient(Protocol):
    """The whole surface stage 1 needs. Narrow so it is trivial to fake."""

    def complete(self, *, system: str, user: str) -> str:
        ...


@dataclass(frozen=True)
class Rejection:
    """A proposal that did not survive verification.

    Kept rather than dropped. "I did not understand this part" is a useful thing
    to show a user, and a silently discarded proposal is indistinguishable from
    one that was never made.
    """

    field: str
    value: str
    span: str
    why: str

    def to_json(self) -> Dict[str, Any]:
        return {"field": self.field, "value": self.value, "span": self.span,
                "why": self.why}


@dataclass(frozen=True)
class Disagreement:
    """Both parsers read one field, and read it differently."""

    field: str
    deterministic: str
    model: str

    def to_json(self) -> Dict[str, Any]:
        return {"field": self.field, "deterministic": self.deterministic,
                "model": self.model}


@dataclass(frozen=True)
class ParseProvenance:
    """How stage 1 was produced, so the compile can be accounted for.

    A saved plan records this. Recompiling it does not call a model again — the
    parse is the input to the deterministic stages, and re-deriving it from a
    model that has since changed would silently alter a saved plan.
    """

    parser_id: str = PARSER_ID
    parser_version: str = PARSER_VERSION
    model: Optional[str] = None
    model_available: bool = False
    model_error: str = ""
    mode: str = "DETERMINISTIC"
    """Which parser the deployment declared when this parse was produced.

    Recorded on the parse rather than read from the deployment later. A plan
    reopened after the configuration moves must show how it was actually
    interpreted, not how the service would interpret it today — the same rule
    that stops a stored figure being re-read against a snapshot that has since
    changed. `model_available` says whether a model answered; this says what
    was asked for, and the two differ when a fallback happened.
    """

    rejected: Sequence[Rejection] = ()
    disagreements: Sequence[Disagreement] = ()
    accepted_from_model: Sequence[str] = ()

    def to_json(self) -> Dict[str, Any]:
        return {
            "parser_id": self.parser_id,
            "parser_version": self.parser_version,
            "model": self.model,
            "model_available": self.model_available,
            "model_error": self.model_error,
            "mode": self.mode,
            "rejected": [r.to_json() for r in self.rejected],
            "disagreements": [d.to_json() for d in self.disagreements],
            "accepted_from_model": list(self.accepted_from_model),
        }


@dataclass(frozen=True)
class VerifiedParse:
    parsed: ParsedUtterance
    provenance: ParseProvenance


# --- the prompt ------------------------------------------------------------

def build_system_prompt() -> str:
    """Built from the vocabulary, so it cannot describe a retired value."""
    lines = [
        "You extract structured readings from a description of an investing "
        "plan. You do not give advice, judge the plan, or add anything the "
        "description does not say.",
        "",
        "Return JSON only, of the form:",
        '{"recognitions": [{"field": ..., "value": ..., "span": ...}],',
        ' "assets": [...], "unclear": [...]}',
        "",
        "`span` MUST be copied verbatim from the description — the exact "
        "characters, not a paraphrase. A recognition whose span is not present "
        "in the text is discarded.",
        "",
        "`field` and `value` MUST come from this closed list. If the "
        "description implies something outside it, put a short phrase in "
        "`unclear` instead of inventing a field or a value:",
    ]
    for field_name in sorted(VOCABULARY):
        for value in sorted(VOCABULARY[field_name]):
            lines.append(f"  {field_name} = {value}")
    lines += [
        "",
        "`assets` are ticker symbols that appear literally in the description. "
        "Do not resolve a company name to a ticker; if the description names a "
        "company, put the name in `unclear`.",
        "",
        "Omit a field entirely when the description does not settle it. An "
        "omission becomes a question the user is asked. A guess becomes a "
        "number they are not.",
    ]
    return "\n".join(lines)


# --- verification ----------------------------------------------------------

def verify_proposals(
    payload: Mapping[str, Any],
    text: str,
) -> tuple:
    """Check every model proposal against the text and the vocabulary.

    Deterministic and pure: same payload and same text, same result. This is the
    boundary the rest of the compiler trusts, so it is the one function here
    that must be readable end to end.

    Returns `(recognitions, assets, unclear, rejections)`.
    """
    haystack = _normalized(text)
    tokens = {t.upper() for t in _TOKEN.findall(text)}

    recognitions: List[Recognition] = []
    rejections: List[Rejection] = []
    claimed: Set[str] = set()

    for item in payload.get("recognitions") or []:
        if not isinstance(item, Mapping):
            continue
        field_name = str(item.get("field", ""))
        value = str(item.get("value", ""))
        span = str(item.get("span", ""))

        if field_name in _TEXT_DERIVED:
            continue          # checked below, against the text rather than here
        if field_name not in VOCABULARY:
            rejections.append(Rejection(field_name, value, span,
                                        "not a field stage 1 recognises"))
            continue
        # Case-insensitive on the value. The vocabulary is lowercase and the
        # model returns a JSON boolean, which serializes as "False" — so the
        # quarantine was rejecting a *correct* reading of "I don't sell
        # anything" on capitalization alone, 14 times in a 205-case run. The
        # field name stays exact; only the value is normalized.
        canonical = {v.lower(): v for v in VOCABULARY[field_name]}
        if value.lower() not in canonical:
            rejections.append(Rejection(
                field_name, value, span,
                f"{value!r} is not one of "
                f"{sorted(VOCABULARY[field_name])} for {field_name}"))
            continue
        value = canonical[value.lower()]
        if not span or _normalized(span) not in haystack:
            rejections.append(Rejection(
                field_name, value, span,
                "the quoted span does not appear in the description; a reading "
                "supported by words the user did not write is a fabrication"))
            continue
        if field_name in claimed:
            rejections.append(Rejection(field_name, value, span,
                                        "the field was already recognised"))
            continue

        recognitions.append(Recognition(field=field_name, value=value,
                                        span=span.strip()))
        claimed.add(field_name)

    # Amounts come from the text, never from the model — the deterministic
    # extractor has already read them. A proposal that agrees is therefore not
    # reported at all: telling a user "we could not use amount 500" on a plan
    # whose amount was read correctly would undermine the one screen whose job
    # is trust. A proposal that disagrees is the single most consequential
    # fabrication available, and is reported plainly.
    stated_amounts = {m.group(1).replace(",", "") for m in _AMOUNT.finditer(text)}
    for item in payload.get("recognitions") or []:
        if isinstance(item, Mapping) and item.get("field") == "amount":
            value = str(item.get("value", "")).replace(",", "").lstrip("$")
            if value not in stated_amounts:
                rejections.append(Rejection(
                    "amount", value, str(item.get("span", "")),
                    "no such figure appears in the description"))

    assets: List[str] = []
    for asset in payload.get("assets") or []:
        symbol = str(asset).strip().upper()
        if symbol and symbol in tokens:
            assets.append(symbol)
        elif symbol:
            rejections.append(Rejection(
                "assets", symbol, "",
                "the symbol does not appear in the description; resolving a "
                "company name to a ticker is how a scenario prices the wrong "
                "security"))

    unclear = [str(u).strip() for u in (payload.get("unclear") or [])
               if str(u).strip()]
    return tuple(recognitions), tuple(dict.fromkeys(assets)), tuple(unclear), \
        tuple(rejections)


def merge(deterministic: ParsedUtterance,
          model_recognitions: Sequence[Recognition],
          model_assets: Sequence[str],
          model_unclear: Sequence[str]) -> tuple:
    """Combine both readings. The deterministic one wins every contested field.

    The regexes are narrow and high-precision by construction: each matches a
    specific phrase that distinguishes two economically different readings. When
    one fires, it saw that phrase. The model's job is the phrasings it does not
    cover, not second-guessing the ones it does.

    A disagreement is surfaced rather than resolved silently, because a plan
    where two readers of the same sentence differ is exactly a plan whose
    confirmation screen should say so.
    """
    by_field = {r.field: r for r in deterministic.recognitions}
    disagreements: List[Disagreement] = []
    accepted: List[str] = []
    combined = list(deterministic.recognitions)

    for proposal in model_recognitions:
        existing = by_field.get(proposal.field)
        if existing is None:
            combined.append(proposal)
            accepted.append(proposal.field)
        elif existing.value != proposal.value:
            disagreements.append(Disagreement(
                field=proposal.field, deterministic=existing.value,
                model=proposal.value))

    assets = tuple(dict.fromkeys([*deterministic.assets, *model_assets]))
    # Kept apart. `unrecognized` names instruments with known alternatives to
    # offer; `unclear` is prose the vocabulary had no home for.
    unclear = tuple(dict.fromkeys([*deterministic.unclear, *model_unclear]))
    return tuple(combined), assets, tuple(deterministic.unrecognized), \
        unclear, tuple(disagreements), tuple(accepted)


# --- stage 1 ---------------------------------------------------------------

def parse_with_model(text: str, *,
                     client: Optional[ModelClient] = None,
                     model: str = DEFAULT_MODEL,
                     mode: str = "") -> VerifiedParse:
    """Stage 1. Deterministic rules, widened by a verified model reading.

    Never raises on the model's account. A missing key, a timeout, malformed
    JSON or a refusal all produce the deterministic parse with the reason
    recorded — the compiler degrades to narrower recognition and more questions,
    which is the correct direction to fail in.
    """
    deterministic = parse_deterministic(text)

    if client is None:
        return VerifiedParse(deterministic,
                             ParseProvenance(model=None, model_available=False,
                                             mode=mode or "DETERMINISTIC",
                                             model_error="no client configured"))

    # One line per provider call, so the count can be measured against a live
    # server rather than a fixture. A journey that pins its parse should
    # produce exactly one of these; a helper-level counter proved the helper
    # and missed a route that bypassed it.
    logger.info("stage1 provider call: model=%s mode=%s", model, mode)
    try:
        raw = client.complete(system=build_system_prompt(), user=text)
        payload = _load_json(raw)
    except Exception as exc:                                    # noqa: BLE001
        # A fallback is legitimate and is *not* a pinned replay. Logged
        # distinctly so a journey that fell back cannot pass a strict one-call
        # conformance check by looking the same as one that did not.
        logger.warning("stage1 fallback to deterministic: %s", exc)
        return VerifiedParse(
            deterministic,
            ParseProvenance(model=model, model_available=False,
                            mode=mode or "DETERMINISTIC",
                            model_error=f"{type(exc).__name__}: {exc}"))

    recognitions, assets, unclear, rejections = verify_proposals(payload, text)
    combined, all_assets, unrecognized, unclear_phrases, disagreements, accepted = \
        merge(deterministic, recognitions, assets, unclear)

    # The template hint stays deterministic. A life-event template states cited,
    # checkable rules; letting a model choose one would put a paraphrase of them
    # in front of a user who has no way to tell.
    parsed = ParsedUtterance(
        text=text, recognitions=combined, assets=all_assets,
        unrecognized=unrecognized, unclear=unclear_phrases,
        template_hint=deterministic.template_hint)
    return VerifiedParse(
        parsed,
        ParseProvenance(model=model, model_available=True,
                        mode=mode or "MODEL_ASSISTED",
                        rejected=rejections, disagreements=disagreements,
                        accepted_from_model=accepted))


def parse_from_stored(payload: Mapping[str, Any], text: str) -> ParsedUtterance:
    """Rebuild a pinned parse, re-verifying every claim against the text.

    Used for two things that look different and are the same: reloading a saved
    plan, and accepting a parse posted back from a confirmation screen. Neither
    is trusted. The stored form travels through a database row or a browser, and
    a recognition arriving without a span still present in the description is
    refused here exactly as it would be coming from the model.

    Built on top of a fresh deterministic parse rather than from the payload
    alone. Everything the rules can derive from the text — amounts, tickers,
    ambiguous company names, the template hint — is re-derived, so the round trip
    cannot lose it and a stored value cannot override it. The payload contributes
    only what the model added: vocabulary recognitions whose spans still hold.

    Re-verifying costs nothing, and it means there is one verification path
    rather than a trusted one and a checked one.
    """
    if payload.get("text") not in (None, text):
        raise ValueError(
            "the stored parse is of different text than the description it is "
            "being applied to")

    deterministic = parse_deterministic(text)
    recognitions, assets, unclear, _rejected = verify_proposals(payload, text)
    stored_unclear = [str(u) for u in (payload.get("unclear") or [])]

    combined, all_assets, unrecognized, unclear_phrases, _disagree, _accepted = \
        merge(deterministic, recognitions, assets, [*stored_unclear, *unclear])

    return ParsedUtterance(
        text=text, recognitions=combined, assets=all_assets,
        unrecognized=unrecognized, unclear=unclear_phrases,
        # Re-derived from the text, like every other rule-derivable field.
        #
        # Left off entirely, a rebuilt parse carried no watched instrument, so
        # the funding policy fell back to taking the held asset as the signal
        # subject — and a plan buying VTI on an SPY signal came back watching
        # VTI. The content hash drifted between the first compile and every
        # later one, which is how the round-trip suite caught it.
        #
        # Derived rather than read from the payload: the roles follow from the
        # sentence, and a stored value could disagree with the words the plan
        # still holds.
        observed=deterministic.observed,
        # Re-derived, not read back: a stored hint could name a template that
        # has since been retired, and the hint selects cited rules.
        template_hint=deterministic.template_hint)


def _load_json(raw: str) -> Mapping[str, Any]:
    """Tolerate a fenced block; refuse anything that is not a JSON object."""
    body = raw.strip()
    if body.startswith("```"):
        body = body.split("```")[1]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
    loaded = json.loads(body)
    if not isinstance(loaded, Mapping):
        raise ValueError("stage 1 expects a JSON object")
    return loaded


# --- the Anthropic client --------------------------------------------------

class AnthropicClient:
    """A thin adapter. Imported lazily so the package stays optional.

    Nothing here makes the model deterministic, and nothing needs to. Sampling
    settings are not exposed — newer models reject `temperature` outright — so
    two calls on one description may differ. Reproducibility does not come from
    the call: it comes from verifying every proposal against the text, and from
    pinning the resulting parse to the saved plan so it is never re-derived.

    That is the stronger arrangement anyway. A temperature of zero would have
    been a claim about the vendor's behaviour; the pinned parse is a fact about
    ours.

    `max_tokens` is generous because a truncated response is unparseable JSON,
    which falls back to the deterministic rules and silently loses the model's
    contribution. Failing safe is right; failing safe because the budget was too
    tight is waste.
    """

    def __init__(self, *, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 max_tokens: int = 4096, timeout: float = 20.0) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._timeout = timeout
        self._client = None

    def _ensure(self):
        if self._client is None:
            import anthropic                                    # noqa: PLC0415

            self._client = anthropic.Anthropic(api_key=self._api_key,
                                               timeout=self._timeout)
        return self._client

    #: Metadata from the last call. An alias like "claude-sonnet-5" is not an
    #: identifier — the provider resolves it, and a run that records only the
    #: alias cannot be reproduced once it moves. The concrete id the API
    #: returned is captured here alongside token counts.
    last_response: Dict[str, Any] = None

    def complete(self, *, system: str, user: str) -> str:
        response = self._ensure().messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=system, messages=[{"role": "user", "content": user}])
        usage = getattr(response, "usage", None)
        self.last_response = {
            "requested_model": self.model,
            "resolved_model": getattr(response, "model", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }
        return "".join(block.text for block in response.content
                       if getattr(block, "type", None) == "text")
