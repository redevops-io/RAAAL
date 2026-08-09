"""`SyntaxReader` over Stanza, and the recorded-parse reader that replaces it.

Two implementations of one protocol, and which one runs is the whole point.

    StanzaReader      real parser, real model, ~500MB, seconds to load
    RecordedReader    a JSON file of parses Stanza produced earlier

Tests use `RecordedReader`. That is not a convenience: a suite that loads a
neural model is a suite nobody runs on a laptop, and a suite that ships a
hand-written stand-in parser has reinvented the regex compiler with a
linguistics vocabulary. Recording the real parser's output keeps the evidence
real and the run fast.

**What keeps a recording honest.** A fixture that has drifted from the parser
makes every test above it meaningless while staying green — the same shape as a
test whose subject was deleted. So each recording carries the parser and model
versions that produced it, and `tests/test_parse_fixtures_are_current.py` re-runs
the live parser and diffs. That check needs the model, so it runs on its own CI
job rather than on every push; what runs everywhere is the fixture-backed suite,
and what stops the fixtures rotting is a job that is allowed to be slow.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .syntax import Parse, Sentence, Token

#: Everything needed to attach a modifier to its governor, and nothing else.
#: `ner` and `constituency` are deliberately absent — they would slow every
#: parse for evidence this layer does not score.
PROCESSORS = "tokenize,pos,lemma,depparse"

FIXTURES = (Path(__file__).resolve().parent.parent.parent
            / "corpus" / "parser" / "parses.json")


def _token(raw, offset: int) -> Token:
    return Token(index=int(raw.id if not isinstance(raw.id, tuple) else raw.id[0]),
                 text=raw.text,
                 lemma=(raw.lemma or raw.text).lower(),
                 upos=raw.upos or "",
                 head=int(raw.head or 0),
                 relation=raw.deprel or "",
                 start_char=int(raw.start_char if raw.start_char is not None
                                else offset),
                 end_char=int(raw.end_char if raw.end_char is not None
                              else offset + len(raw.text)))


class StanzaReader:
    """The real parser. Loaded lazily, because importing it costs seconds."""

    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._pipeline = None
        self._version = ""

    @property
    def id(self) -> str:
        return f"stanza@{self._version or 'unloaded'}:{self.language}"

    def _load(self):
        if self._pipeline is None:
            import stanza

            self._version = stanza.__version__
            self._pipeline = stanza.Pipeline(
                lang=self.language, processors=PROCESSORS,
                download_method=None, verbose=False)
        return self._pipeline

    def parse(self, text: str, language: str = "en") -> Parse:
        if language != self.language:
            raise ValueError(
                f"this reader was built for {self.language!r} and was asked "
                f"for {language!r}. A parser answering in the wrong language "
                "produces confident nonsense rather than an error")
        document = self._load()(text)
        sentences = tuple(
            Sentence(sentence_id=index,
                     tokens=tuple(_token(word, 0)
                                  for word in sentence.words))
            for index, sentence in enumerate(document.sentences))
        return Parse(language=language, sentences=sentences,
                     parser=f"stanza@{self._version}",
                     model=f"{language}/{PROCESSORS}")


class RecordedReader:
    """Parses Stanza produced earlier, keyed by text and language.

    Raises on a miss rather than parsing live. A reader that quietly fell back
    to the real parser would make the fast suite occasionally slow and, worse,
    would hide the fact that a fixture is missing — the run would pass and
    nobody would know the recording had never been made.
    """

    id = "recorded-parse@1"

    def __init__(self, path: Path = FIXTURES) -> None:
        self.path = path
        self._by_key: Dict[str, Any] = {}
        if path.exists():
            document = json.loads(path.read_text())
            self.recorded_with = document.get("recorded_with", {})
            for entry in document["parses"]:
                self._by_key[key(entry["text"], entry["language"])] = entry
        else:
            self.recorded_with = {}

    def __len__(self) -> int:
        return len(self._by_key)

    def has(self, text: str, language: str = "en") -> bool:
        return key(text, language) in self._by_key

    def parse(self, text: str, language: str = "en") -> Parse:
        entry = self._by_key.get(key(text, language))
        if entry is None:
            raise KeyError(
                f"no recorded parse for {language}:{text!r}. Run "
                "`python corpus/parser/record_parses.py` — falling back to the "
                "live parser here would hide the gap behind a green run")
        return from_json(entry)


def key(text: str, language: str) -> str:
    """The lookup key. One function, so a recorder and a reader that disagreed
    about it could not produce a file that loads and never matches."""
    return f"{language}\t{text}"


def to_json(parse: Parse, text: str) -> dict:
    return {
        "text": text, "language": parse.language,
        "parser": parse.parser, "model": parse.model,
        "sentences": [
            {"sentence_id": sentence.sentence_id,
             "tokens": [{"index": t.index, "text": t.text, "lemma": t.lemma,
                         "upos": t.upos, "head": t.head, "relation": t.relation,
                         "start_char": t.start_char, "end_char": t.end_char}
                        for t in sentence.tokens]}
            for sentence in parse.sentences]}


def from_json(entry: Mapping[str, Any]) -> Parse:
    return Parse(
        language=entry["language"], parser=entry["parser"], model=entry["model"],
        sentences=tuple(
            Sentence(sentence_id=s["sentence_id"],
                     tokens=tuple(Token(**t) for t in s["tokens"]))
            for s in entry["sentences"]))


def default_reader(prefer_live: bool = False) -> Optional[Any]:
    """A `RecordedReader`, or the live parser when explicitly asked for it."""
    return StanzaReader() if prefer_live else RecordedReader()
