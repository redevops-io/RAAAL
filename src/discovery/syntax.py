"""Deterministic syntax evidence — what modifies what, and nothing about money.

Discovery has one semantic reader today and it is hosted. This layer is the
cheap local complement: a dependency parse says *which token modifies which*,
and that is enough to separate the readings a semantic reader most often
confuses, without knowing anything about investing.

    "contribute $500 monthly, rebalanced annually"

A reader that collects cadences finds two and picks one. A parse says `monthly`
modifies `contribute` and `annually` modifies `rebalanced`, and the ambiguity
disappears without anybody deciding what a contribution is.

**This layer is never the authority.** It emits evidence with a score;
`DecisionEvidence` carries it beside the model's, and fusion decides. There is
no rule anywhere that syntax wins. The reason is not politeness — a parser that
could overrule the semantic reader would be the legacy regex compiler with a
linguistics dependency, and the whole migration exists because that architecture
required a code change for every sentence nobody anticipated.

    raw prompt ─► SyntaxReader ─► SyntaxEvidence[] ─► fusion ─► VerifiedIntent
                                        ▲
                       normalisation ───┘   (money, ratios, durations, windows)

**What is deliberately not here.** No semantic decisions: this module never
decides that a cadence *is* a contribution cadence, only that a token attaches
to a verb. No hand-written grammar either — the parser is injected behind
`SyntaxReader`, and tests run against *recorded* parses rather than a fallback
implementation, because a hand-rolled fallback is the regex compiler wearing a
new name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

#: Bumped when the *meaning* of a score or an edge changes, so stored evidence
#: can be told apart from evidence produced by a different scoring rule.
SCORING_VERSION = "quantify-syntax-scoring@1"


@dataclass(frozen=True)
class Token:
    """One word, and what it hangs off.

    Universal Dependencies shape, because that is what both candidate parsers
    emit and because it is the one representation that survives changing them.
    `head` is an index into the sentence's own token list; 0 is the root.
    """

    index: int
    text: str
    lemma: str
    upos: str
    head: int
    relation: str
    start_char: int
    end_char: int

    @property
    def span(self) -> tuple:
        return (self.start_char, self.end_char)


@dataclass(frozen=True)
class Sentence:
    sentence_id: int
    tokens: Sequence[Token]

    def head_of(self, token: Token) -> Optional[Token]:
        if token.head == 0:
            return None
        for other in self.tokens:
            if other.index == token.head:
                return other
        return None

    def governor_chain(self, token: Token, limit: int = 8) -> Sequence[Token]:
        """From a token up to the root. Bounded: a malformed parse can cycle,
        and an unbounded walk over one would hang the reader rather than
        produce a bad reading, which is harder to notice."""
        chain, current, seen = [], token, {token.index}
        for _ in range(limit):
            current = self.head_of(current)
            if current is None or current.index in seen:
                break
            chain.append(current)
            seen.add(current.index)
        return tuple(chain)

    def children_of(self, token: Token) -> Sequence[Token]:
        return tuple(t for t in self.tokens if t.head == token.index)


@dataclass(frozen=True)
class Parse:
    """One text, parsed. Everything needed to replay the evidence it produced."""

    language: str
    sentences: Sequence[Sentence]
    parser: str
    """Implementation and version — `stanza@1.8.2`, `trankit@1.1.1`."""

    model: str
    """Which model produced it. A parser upgrade that changes an attachment
    changes a reading, and without this nobody can tell that is what happened."""

    def tokens(self):
        for sentence in self.sentences:
            for token in sentence.tokens:
                yield sentence, token


@runtime_checkable
class SyntaxReader(Protocol):
    """The one thing every parser is.

    Deliberately the same shape as `DiscoveryReader`, and deliberately not the
    same type: a `SyntaxReader` returns structure, a `DiscoveryReader` returns
    meaning, and a single interface over both would invite exactly the
    substitution this layer must not make.
    """

    id: str

    def parse(self, text: str, language: str = "en") -> Parse:
        ...


# ── Phase 2: deterministic normalisation ─────────────────────────────────────
#
# Values, not meanings. `$1k` is 1000 USD in every language and every domain;
# whether 1000 USD is a contribution is not this module's business.


@dataclass(frozen=True)
class Value:
    """A normalised literal and the characters it came from.

    `source_span` is carried on every one of these because a normalisation that
    cannot point at its own input cannot be checked, and the plan's own
    falsification list is entirely about *which* number was read.
    """

    kind: str
    canonical: Any
    source_span: str
    start_char: int
    end_char: int
    unit: str = ""

    def to_json(self) -> dict:
        return {"kind": self.kind,
                "canonical": (str(self.canonical)
                              if isinstance(self.canonical, Decimal)
                              else self.canonical),
                "unit": self.unit or None,
                "source_span": self.source_span,
                "start_char": self.start_char, "end_char": self.end_char}


_MULTIPLIER = {"k": 1000, "m": 1_000_000, "bn": 1_000_000_000, "b": 1_000_000_000}

#: `$1,500`, `1.5k`, `€2m`, `500 dollars`, and `500 €` — the postfix order most
#: of Europe writes, which the first version of this missed entirely and the
#: corpus caught on its first run.
_MONEY = re.compile(
    r"(?P<symbol>[$£€])\s?(?P<amount>\d[\d.,]*)\s?(?P<mult>k|m|bn|b)?\b"
    r"|(?P<amount2>\d[\d.,]*)\s?(?P<mult2>k|m|bn|b)?\s?"
    r"(?P<word>dollars?|usd|pounds?|gbp|euros?|eur)\b"
    r"|(?P<amount3>\d[\d.,]*)\s?(?P<mult3>k|m|bn|b)?\s?(?P<symbol2>[$£€])",
    re.I)

#: Which separator groups and which one divides.
#:
#: This is the one genuinely locale-dependent thing in the module, and the
#: first version had no opinion about it — which is not neutrality, it is
#: silently assuming en. `1.000 €` is a thousand euros in Madrid and one euro
#: in New York; `500,50` is five hundred and a half in Berlin and fifty
#: thousand and fifty in Chicago. Both readings are common and neither is
#: recoverable from the digits.
#:
#: So `normalize` takes a language, `en` is a *declared* default rather than an
#: assumed one, and a string that is not well-formed under the convention in
#: force is refused instead of read under the other.
_GROUPING = {
    "en": re.compile(r"^\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^\d+(?:\.\d+)?$"),
}
_EUROPEAN = re.compile(r"^\d{1,3}(?:\.\d{3})*(?:,\d+)?$|^\d+(?:,\d+)?$")
for _language in ("es", "de", "fr", "it", "pt", "nl", "ru", "tr", "pl"):
    _GROUPING[_language] = _EUROPEAN

_SYMBOL_CURRENCY = {"$": "USD", "£": "GBP", "€": "EUR"}
_WORD_CURRENCY = {"dollar": "USD", "dollars": "USD", "usd": "USD",
                  "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
                  "euro": "EUR", "euros": "EUR", "eur": "EUR"}

#: `60%`, `4.5 %`
_PERCENT = re.compile(r"(?P<amount>\d+(?:\.\d+)?)\s?%")

#: `60/40`, `70 / 20 / 10`, and `60-40` — a split, which is not two percentages
#: that happen to be adjacent. Kept as ordered weights because which sleeve gets
#: which is the whole content.
#:
#: The hyphen form was missing entirely until the Stack Exchange harvest, where
#: four of the first twenty-nine attested sentences wrote `60-40` or `80-20`.
#: Every invented case had used a slash. The sums-to-100 test does the same work
#: for both: `80-20` is a split, `2012-2015` and `10-20` are not.
_RATIO = re.compile(
    r"\b(?P<parts>\d{1,3}(?:\s?/\s?\d{1,3}){1,4})\b"
    r"|\b(?P<hyphen>\d{1,3}(?:\s?-\s?\d{1,3}){1,4})\b(?!\s?%)")

#: `$200-$220k`, `10-20%` — a range, whose ends this module refuses to read.
#:
#: Both were attested and both were read wrongly. `10-20%` came back as 20%,
#: silently collapsing to the upper bound. `~$200-$220k` came back as two
#: amounts, `200` and `220000`, so the low end was out by a factor of a
#: thousand — the multiplier at the far end governs both, and a reader taking
#: the first match cannot know that.
#:
#: Refused rather than resolved. A range is a thing to ask the user about, and
#: a plausible wrong amount is worse than a question.
#: `and` is a range marker only after `between`. On its own it joins two
#: separate amounts — "contribute $500 and $200 to the second sleeve" is two
#: contributions, not a range, and treating every `and` as one would refuse
#: perfectly readable sentences.
_AMOUNT = r"[$£€]\s?\d[\d.,]*\s?(?:k|m|bn|b)?"
_MONEY_RANGE = re.compile(
    rf"{_AMOUNT}\s?(?:-|–|\bto\b)\s?[$£€]?\s?\d[\d.,]*\s?(?:k|m|bn|b)?"
    rf"|\bbetween\s+{_AMOUNT}\s+and\s+[$£€]?\s?\d[\d.,]*\s?(?:k|m|bn|b)?",
    re.I)
_PERCENT_RANGE = re.compile(r"\d+(?:\.\d+)?\s?(?:-|–|\bto\b)\s?\d+(?:\.\d+)?\s?%")

#: `5 years`, `60 months`, `90 days`, `18-month`
#:
#: `(?! old)` because "Me - 32 years old" was attested and came back as an
#: 11,680-day duration. An age is not a horizon, and a reader that cannot tell
#: them apart turns a biography into a backtest length.
_DURATION = re.compile(
    r"\b(?P<amount>\d+(?:\.\d+)?)[\s-]?(?P<unit>day|days|week|weeks|month|months|"
    r"year|years|yr|yrs)\b(?!\s+old\b)", re.I)

_DURATION_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365,
                  "yr": 365}

#: `200-day moving average`, `50 day MA`, `12-month moving average`
_WINDOW = re.compile(
    r"\b(?P<amount>\d+)[\s-]?(?P<unit>day|days|week|weeks|month|months)?\s*"
    r"(?:moving[\s-]average|ma|sma|ema)\b", re.I)


def _decimal(raw: str) -> Optional[Decimal]:
    try:
        return Decimal(raw.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _amount(raw: str, language: str) -> Optional[Decimal]:
    """A written number under one convention, or nothing.

    Returns `None` for a string that is not well-formed under the language in
    force rather than falling back to the other convention. A wrong amount that
    looks plausible is worse than a dimension the user is asked about, which is
    the same reason this project has no synonym table.
    """
    raw = raw.strip().rstrip(".,")
    grouping = _GROUPING.get(language.lower()[:2])
    if grouping is None or not grouping.match(raw):
        return None
    if grouping is _EUROPEAN:
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(",", "")
    try:
        return Decimal(raw)
    except (InvalidOperation, ValueError):
        return None


def normalize(text: str, language: str = "en") -> Sequence[Value]:
    """Every literal this module can canonicalise, with its span.

    Order matters and is deliberate: a moving-average window is recognised
    *before* durations, because "90-day moving average" contains "90-day" and
    reading it as a holding period is one of the plan's named falsification
    cases. Overlapping spans are dropped in favour of the earlier, more
    specific match rather than both being emitted, so a consumer never sees one
    stretch of text carrying two incompatible readings.
    """
    found: list = []

    # Ranges are claimed first and emit nothing, so a later pass cannot read one
    # end of a range as a standalone value. Claiming the span is the mechanism:
    # the same rule that stops "90-day moving average" becoming a holding
    # period stops "$200-$220k" becoming two amounts.
    refused = [match.span() for pattern in (_MONEY_RANGE, _PERCENT_RANGE)
               for match in pattern.finditer(text)]

    def claim(value: Value) -> None:
        for start, end in refused:
            if not (value.end_char <= start or value.start_char >= end):
                return
        for existing in found:
            if not (value.end_char <= existing.start_char
                    or value.start_char >= existing.end_char):
                return
        found.append(value)

    for match in _WINDOW.finditer(text):
        amount = _decimal(match.group("amount"))
        if amount is not None:
            unit = (match.group("unit") or "day").lower().rstrip("s")
            claim(Value("moving_average_window", int(amount), match.group(0),
                        *match.span(), unit=unit))

    for match in _MONEY.finditer(text):
        raw = (match.group("amount") or match.group("amount2")
               or match.group("amount3"))
        amount = _amount(raw, language)
        if amount is None:
            continue
        mult = (match.group("mult") or match.group("mult2")
                or match.group("mult3") or "").lower()
        amount *= _MULTIPLIER.get(mult, 1)
        currency = (_SYMBOL_CURRENCY.get(match.group("symbol")
                                         or match.group("symbol2") or "")
                    or _WORD_CURRENCY.get((match.group("word") or "").lower(), ""))
        claim(Value("money", amount, match.group(0), *match.span(), unit=currency))

    for match in _RATIO.finditer(text):
        raw = match.group("parts") or match.group("hyphen")
        parts = [int(p) for p in re.split(r"\s?[/-]\s?", raw)]
        # A ratio whose parts sum to 100 is a split. One that does not is
        # probably a date or an identifier, and guessing which would be the
        # substitution this project refuses.
        if sum(parts) == 100:
            claim(Value("ratio", tuple(parts), match.group(0), *match.span()))

    for match in _PERCENT.finditer(text):
        amount = _decimal(match.group("amount"))
        if amount is not None:
            claim(Value("percentage", amount / Decimal(100), match.group(0),
                        *match.span()))

    for match in _DURATION.finditer(text):
        amount = _decimal(match.group("amount"))
        if amount is None:
            continue
        unit = match.group("unit").lower().rstrip("s")
        claim(Value("duration", int(amount * _DURATION_DAYS[unit]),
                    match.group(0), *match.span(), unit="days"))

    return tuple(sorted(found, key=lambda v: v.start_char))


# ── Phase 3/4: attachment scoring ────────────────────────────────────────────


@dataclass(frozen=True)
class SyntaxEvidence:
    """One deterministic observation, scored, never decisive.

    The score is interpretable rather than probabilistic: it is a sum of named
    features, and `features` carries them so a reader can see the arithmetic.
    A probability here would invite thresholding, and a threshold is a decision
    this layer is not allowed to make.
    """

    dimension: str
    proposed_value: Any
    score: int
    features: Sequence[str] = ()
    source_span: str = ""
    sentence_id: int = 0
    parser: str = ""
    model: str = ""
    scoring_version: str = SCORING_VERSION

    def to_json(self) -> dict:
        return {"source_type": "syntax", "dimension": self.dimension,
                "proposed_value": self.proposed_value, "score": self.score,
                "features": list(self.features), "source_span": self.source_span,
                "sentence_id": self.sentence_id,
                "parser": self.parser, "model": self.model,
                "scoring_version": self.scoring_version}


@dataclass(frozen=True)
class Attachment:
    """A named scoring rule: which governors support a dimension, and which
    argue against it.

    `against` is not the complement of `supports`. A verb absent from both is
    neutral and scores zero, which is the honest answer for a sentence this
    rule has never seen — the alternative, treating unknown as negative, makes
    the layer confidently wrong on exactly the unanticipated phrasings it was
    added to handle.
    """

    dimension: str
    supports: Mapping[str, int] = field(default_factory=dict)
    against: Mapping[str, int] = field(default_factory=dict)
    same_clause_bonus: int = 2


#: The failure classes already observed, one rule each.
#:
#: `rebalance`/`harvest`/`adjust` score *against* contribution cadence rather
#: than merely not supporting it, because the sentence that produced this rule
#: — "invest $500 monthly, rebalanced annually" — has two cadences and a reader
#: that scores both at zero has learned nothing.
ATTACHMENTS: Mapping[str, Attachment] = {
    "cadence": Attachment(
        dimension="cadence",
        supports={"contribute": 3, "invest": 3, "deposit": 3, "add": 2,
                  "buy": 2, "save": 2, "put": 2},
        against={"rebalance": -4, "harvest": -4, "adjust": -4,
                 "withdraw": -3, "review": -3}),
}


@dataclass(frozen=True)
class Aligned:
    """A normalised value, and the tokens whose characters it covers.

    **Normalisation happens before scoring, and this is what makes that true.**
    Stanza tokenises `401k (50/50), Roth IRA (85/15), taxable brokerage (70/30)`
    three different ways in one sentence — `50/50` whole, `85/15` into three
    tokens, `70/30` into `70/` and `30`. A scorer reasoning over tokens is
    reasoning over whatever the tokenizer happened to do that time.

    So the scorer is handed a `Value` with a character span, and the tokens are
    looked up from the span rather than the other way round. The unit it
    reasons about is then the same unit in every sentence, whatever the parser
    did to the characters.
    """

    value: Value
    sentence: Sentence
    tokens: Sequence[Token]

    @property
    def anchor(self) -> Optional[Token]:
        """The token to walk up from: the last one the value covers.

        The last rather than the first, because the head of a numeric phrase
        sits at its right edge in English — in `200-day moving average` it is
        `average`, not `200`, that attaches to the verb.
        """
        return self.tokens[-1] if self.tokens else None


def align(parse: Parse, values: Sequence[Value]) -> Sequence[Aligned]:
    """Match normalised values to the tokens covering their characters.

    A value whose span no token overlaps is dropped rather than guessed at —
    that means the parse and the normaliser disagree about the text, and
    picking a nearby token would invent an attachment neither of them made.
    """
    aligned = []
    for value in values:
        for sentence in parse.sentences:
            covered = tuple(
                token for token in sentence.tokens
                if not (token.end_char <= value.start_char
                        or token.start_char >= value.end_char))
            if covered:
                aligned.append(Aligned(value=value, sentence=sentence,
                                       tokens=covered))
                break
    return tuple(aligned)


def score_value(aligned: Aligned, rule: Attachment) -> tuple:
    """`(score, features)` for a normalised value rather than a token."""
    if aligned.anchor is None:
        return 0, ()
    return score_attachment(aligned.sentence, aligned.anchor, rule)


def score_attachment(sentence: Sentence, token: Token,
                     rule: Attachment) -> tuple:
    """`(score, features)` for one token under one rule.

    Walks the governor chain rather than only the immediate head, because
    "invest five hundred a month" puts a noun between the adverbial and the
    verb in most parses, and a rule that only looked one step up would score it
    zero for a purely structural reason.
    """
    score, features = 0, []
    for distance, governor in enumerate(sentence.governor_chain(token), start=1):
        lemma = governor.lemma.lower()
        if lemma in rule.supports:
            weight = rule.supports[lemma]
            score += weight if distance == 1 else max(1, weight - 1)
            features.append(f"+{weight}:modifies({lemma})@{distance}")
        elif lemma in rule.against:
            weight = rule.against[lemma]
            score += weight if distance == 1 else min(-1, weight + 1)
            features.append(f"{weight}:modifies({lemma})@{distance}")
    return score, tuple(features)
