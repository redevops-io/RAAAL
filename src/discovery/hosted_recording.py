"""Recorded model readings — the same discipline the parses already have.

The hosted reader is the *other* independent witness, not a fallback for when
syntax comes up empty. That distinction is the architecture, and it has a cost:
the model runs on every utterance, so a corpus run would make one call per case
and parser CI would depend on network and provider availability.

So the same pattern as `syntax_stanza.RecordedReader`:

    utterance → live HostedReader once
              → recorded reply + model id + prompt and schema version
              → corpus replays the recording

**A recorded reading is never authoritative because it was recorded.** It
replays what the model proposed and nothing more; fusion still decides whether
that proposal proceeds, exactly as it would for a live reply. The recording is a
transport, and this file must never grow a shortcut that treats a stored
`Reading` as a settled field — `test_hosted_recording.py` holds that line.

**What identity a recording carries.** Model id, prompt version and schema
fingerprint, because a reply produced under a different prompt or a different
schema is a reply to a different question. A recording missing any of those
cannot be compared with a live reply later, which is what the drift lane exists
to do.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from .reader import Reading, ReadingSet, RelationReading, Schema

FIXTURES = (Path(__file__).resolve().parent.parent.parent
            / "corpus" / "parser" / "hosted.json")

RECORDING_SCHEMA = "quantify-hosted-recording@1"

#: Bumped when the prompt changes in a way that could change a reply. A
#: recording carries it, so a fixture produced under an older prompt is
#: identifiable rather than silently mixed in with newer ones.
PROMPT_VERSION = "quantify-hosted-prompt@1"


def key(text: str, reader_id: str) -> str:
    """One function, so a recorder and a reader cannot disagree about it."""
    return f"{reader_id}\t{text}"


def to_json(reading_set: ReadingSet, text: str, *, schema_version: str,
            prompt_version: str = PROMPT_VERSION,
            question: str = "") -> dict:
    """One recording.

    `question` is the digest of what was actually asked, and it is what the
    replay guard compares. `schema_version` is kept beside it as history —
    which schema this was recorded under — and is no longer the thing that
    decides whether the reply still applies.
    """
    return {
        "text": text,
        "reader_id": reading_set.reader_id,
        "schema_version": schema_version,
        "question_digest": question,
        "prompt_version": prompt_version,
        "failed": reading_set.failed,
        "readings": [{"dimension": r.dimension, "value": r.value,
                      "confidence": r.confidence, "source_span": r.source_span,
                      "note": r.note} for r in reading_set.readings],
        "relations": [r.to_json() for r in reading_set.relations],
        "unread": list(reading_set.unread)}


def from_json(entry: Mapping[str, Any]) -> ReadingSet:
    return ReadingSet(
        reader_id=entry["reader_id"],
        readings=tuple(Reading(**r) for r in entry["readings"]),
        relations=tuple(
            RelationReading(
                kind=r["kind"],
                members=tuple((m["role"], m["subject"], m["qualifiers"])
                              for m in r["members"]),
                attributes=dict(r["attributes"]),
                source_span=r.get("source_span", ""))
            for r in entry["relations"]),
        unread=tuple(entry["unread"]),
        failed=entry["failed"])


class RecordedHostedReader:
    """Replays model readings recorded earlier. Never calls a provider.

    Raises on a miss rather than falling back to a live call: a quiet fallback
    would make the fast suite occasionally slow and, worse, would hide that a
    recording was never made — the run would pass and nobody would know the
    model had not been consulted for that sentence.
    """

    def __init__(self, path: Path = FIXTURES) -> None:
        self.path = path
        self._by_key: Dict[str, Any] = {}
        self.recorded_with: Dict[str, Any] = {}
        if path.exists():
            document = json.loads(path.read_text())
            self.recorded_with = document.get("recorded_with", {})
            for entry in document["readings"]:
                self._by_key[key(entry["text"], entry["reader_id"])] = entry

    @property
    def id(self) -> str:
        return self.recorded_with.get("reader_id", "recorded-hosted@1")

    def __len__(self) -> int:
        return len(self._by_key)

    def has(self, text: str, reader_id: Optional[str] = None) -> bool:
        return key(text, reader_id or self.id) in self._by_key

    def read(self, text: str, schema: Schema) -> ReadingSet:
        entry = self._by_key.get(key(text, self.id))
        if entry is None:
            raise KeyError(
                f"no recorded model reading for {text!r}. Run "
                "`python corpus/parser/record_hosted.py` — calling the provider "
                "here would hide the gap behind a green run")
        if not answers_the_same_question(entry, schema):
            raise ValueError(
                f"recorded under schema {entry['schema_version']}, asked under "
                f"{schema.version}, and the two ask different questions. A "
                "reply to a different question is not an answer to this one")
        return from_json(entry)

    def entry_for(self, text: str) -> Optional[Mapping[str, Any]]:
        return self._by_key.get(key(text, self.id))


def question_digest(schema: Schema) -> str:
    """What this schema actually asks a reader, as a digest.

    The schema is the input; the *question* is what reaches the model, and the
    two are not the same thing. A dimension carried with `asked=False` changes
    the schema and does not change a word of the prompt.

    Computed from the reader's own prompt builders rather than restated, so a
    change to how a dimension is rendered moves this without anybody
    remembering to.
    """
    from hashlib import sha256

    from .readers_quantify import OpenAIReader

    builder = OpenAIReader.__new__(OpenAIReader)
    asked = (builder._schema_prompt(schema) + "\n"
             + builder._relations_prompt(schema))
    return sha256(asked.encode()).hexdigest()[:16]


def answers_the_same_question(entry: Mapping[str, Any], schema: Schema) -> bool:
    """Whether a recorded reply answers the question this schema asks.

    It used to compare `schema.version`, which is a proxy and was wrong in a
    way that only showed up once a schema moved without the prompt moving:
    `@7` added `factor_tilt` and `age_based_allocation`, both `asked=False`, so
    every recorded reply became "an answer to a different question" while the
    question was byte-identical. 1375 tests failed on a distinction that did
    not exist.

    The proxy was also *weaker* than it looked in the other direction. Two
    schemas can share a version and ask different things — that is what the
    fingerprint exists for — and this check would have accepted the recording.

    So it compares the question. `question_digest` is recorded going forward;
    an entry without one falls back to the version, which is the strict old
    behaviour and never accepts more than it used to.
    """
    recorded = entry.get("question_digest")
    if recorded:
        return recorded == question_digest(schema)
    return entry.get("schema_version") == schema.version


def proposals(reading_set: ReadingSet, fields: Optional[Sequence[str]] = None):
    """A reader's readings, in the shape fusion takes.

    A translation of *results* — the contract specifies the utterance both
    readers see, not the shape of what they return. Stated here rather than
    hidden, and deliberately lossy in one direction only: it drops nothing and
    decides nothing, it renames.

    `failed` produces no proposals at all. A transport failure is not a reading,
    and scoring one as silence would let a timeout look like a model that had
    nothing to say — the defect the shadow runner already carries a category
    for.
    """
    from .claims import Proposal

    if not reading_set.ok:
        return ()
    return tuple(
        Proposal(dimension=r.dimension, value=r.value,
                 reader_id=reading_set.reader_id, source_span=r.source_span)
        for r in reading_set.readings
        if fields is None or r.dimension in fields)
