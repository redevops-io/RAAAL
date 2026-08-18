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
            prompt_version: str = PROMPT_VERSION) -> dict:
    return {
        "text": text,
        "reader_id": reading_set.reader_id,
        "schema_version": schema_version,
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
        if entry["schema_version"] != schema.version:
            raise ValueError(
                f"recorded under schema {entry['schema_version']}, asked under "
                f"{schema.version}. A reply to a different question is not an "
                "answer to this one")
        return from_json(entry)

    def entry_for(self, text: str) -> Optional[Mapping[str, Any]]:
        return self._by_key.get(key(text, self.id))


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
