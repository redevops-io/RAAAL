"""`MarketSnapshot` — immutable, content-addressed, and self-describing.

    ResolutionRequest -> MarketSnapshot

**The hard invariant.** Resolving snapshot hash `H` always returns the canonical
observations whose digest is `H`. That is checkable, and `verify` is what checks
it — against the bytes, not against a stored copy of the same claim.

**The hash is over the observations, and the request is part of what produces
them.** This is the finding that started the whole data-lake thread and it is
structural here rather than remembered: the same `Snapshot` resolved with
dividends reinvested returns a *different frame* — the total-return twin instead
of the price series — so a snapshot identity naming only the dataset does not
determine the bytes. `ResolutionRequest` is therefore a field, and two snapshots
differing only in it are two snapshots.

**Immutable and independently serialized are two properties, not one.** A frozen
dataclass whose `to_json` hands back the objects it holds is immutable in the
sense that its fields cannot be reassigned, and mutable in every sense that
matters — an evaluation result did exactly that, and a mutation applied to "the
wire body" changed the result it was supposed to be compared against. So this
copies on the way out, and `from_json` is written as a reader rather than as the
inverse of a dict literal.

**Nothing here is inferred.** Corporate-action treatment, the alignment
calendar, the source adapter and its version are declared or explicitly
`NOT_DECLARED`. A snapshot that quietly said nothing about corporate actions
would be one somebody assumed had handled them — the `dividend_policy` failure,
one layer down.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

#: The shape of this contract, in the hashed body of the *descriptor* rather
#: than of the observations. Changing how a snapshot is described must not
#: change the address of the data it describes.
SNAPSHOT_CONTRACT_VERSION = "quantify-market-snapshot@1"

#: What this build says when a fact about the data is not established.
#:
#: Spelled rather than left empty. An empty string reads as "nothing to say";
#: this reads as "nobody has established it", and only one of those is true.
NOT_DECLARED = "NOT_DECLARED"

#: How distributions are treated in the observations.
#:
#: Not a policy choice at this layer — a description of which series was
#: delivered. `TOTAL_RETURN` means the observations already credit
#: distributions; `PRICE_ONLY` means they do not and nothing here restores them.
TOTAL_RETURN = "TOTAL_RETURN"
PRICE_ONLY = "PRICE_ONLY"


@dataclass(frozen=True)
class SessionRange:
    """The span actually delivered, which is not the span requested."""

    start: str
    end: str
    sessions: int

    def to_json(self) -> Dict[str, Any]:
        return {"start": self.start, "end": self.end,
                "sessions": self.sessions}


@dataclass(frozen=True)
class SourceAdapter:
    """Which code produced the observations, and which version of it.

    Two adapters reading one vendor can disagree about adjustment, session
    alignment and what a missing day means. A snapshot that named the vendor
    and not the adapter would attribute a difference to the market.
    """

    name: str
    version: str

    def to_json(self) -> Dict[str, Any]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class MarketSnapshot:
    """One immutable set of observations, and everything needed to ask again."""

    snapshot_hash: str
    """The content address of the observations. `verify` recomputes it."""

    snapshot_id: str
    dataset_id: str
    symbols: Tuple[str, ...]
    session_range: SessionRange
    resolution: Mapping[str, Any]
    """What was asked for, beyond which dataset. Part of the identity because
    it is part of what produced the bytes."""

    corporate_actions: str
    calendar: str
    source_adapter: SourceAdapter
    source_uri: str
    data_as_of: str
    license_class: str
    license_review_status: str
    content_digest_version: str
    version: str = SNAPSHOT_CONTRACT_VERSION

    def to_json(self) -> Dict[str, Any]:
        """A copy, all the way down.

        `dict(self.resolution)` would hand back the mapping this holds, and a
        caller editing the serialized body would edit the snapshot through it.
        That is not hypothetical — an `EvaluationResult` did exactly this, and
        the mutation test that should have caught a changed field watched both
        sides change together.
        """
        return {
            "version": self.version,
            "snapshot_hash": self.snapshot_hash,
            "snapshot_id": self.snapshot_id,
            "dataset_id": self.dataset_id,
            "symbols": list(self.symbols),
            "session_range": self.session_range.to_json(),
            "resolution": copy.deepcopy(dict(self.resolution)),
            "corporate_actions": self.corporate_actions,
            "calendar": self.calendar,
            "source_adapter": self.source_adapter.to_json(),
            "source_uri": self.source_uri,
            "data_as_of": self.data_as_of,
            "license_class": self.license_class,
            "license_review_status": self.license_review_status,
            "content_digest_version": self.content_digest_version,
        }

    @property
    def descriptor_hash(self) -> str:
        """An address for the *description*, distinct from the data's own.

        Two snapshots of identical observations described differently — a new
        adapter version, a corrected licence review — are the same data and
        different records. Collapsing the two would make a licence correction
        look like a change to the market.
        """
        body = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return "mds1:" + hashlib.sha256(body.encode()).hexdigest()

    def verify(self, observations) -> Sequence[str]:
        """Whether these observations are the ones this snapshot names.

        The hard invariant, asked of the bytes. Returns what is wrong rather
        than a boolean: "the digest differs" and "the symbols differ" send
        somebody to different places, and a snapshot that failed for both
        reasons would report one.
        """
        from .access_event import frame_digest

        problems = []
        if observations is None:
            return ("no observations were resolved, so there is nothing to "
                    "check this snapshot against",)

        recomputed = frame_digest(observations)
        if recomputed != self.snapshot_hash:
            problems.append(
                f"the observations digest to {recomputed} and this snapshot is "
                f"{self.snapshot_hash}. Resolving a hash must return the bytes "
                "whose digest it is, or the figure it produced cannot be "
                "checked against the data")

        delivered = tuple(sorted(str(one) for one in observations.columns))
        if delivered != tuple(sorted(self.symbols)):
            problems.append(
                f"this snapshot names {len(self.symbols)} symbols and "
                f"{len(delivered)} were delivered")

        if len(observations.index) != self.session_range.sessions:
            problems.append(
                f"this snapshot names {self.session_range.sessions} sessions "
                f"and {len(observations.index)} were delivered")
        return tuple(problems)


def from_json(payload: Mapping[str, Any]) -> MarketSnapshot:
    """Read a snapshot record. Written as a reader, not as an inverse.

    Every field is named here, which is the point: a reader built by splatting
    a dict would accept a record missing anything and produce a snapshot that
    described less than it claimed. A field that does not survive the wire is a
    field that silently stops being part of the identity.
    """
    span = payload["session_range"]
    adapter = payload["source_adapter"]
    return MarketSnapshot(
        snapshot_hash=payload["snapshot_hash"],
        snapshot_id=payload["snapshot_id"],
        dataset_id=payload["dataset_id"],
        symbols=tuple(payload["symbols"]),
        session_range=SessionRange(start=span["start"], end=span["end"],
                                   sessions=int(span["sessions"])),
        resolution=copy.deepcopy(dict(payload["resolution"])),
        corporate_actions=payload["corporate_actions"],
        calendar=payload["calendar"],
        source_adapter=SourceAdapter(name=adapter["name"],
                                     version=adapter["version"]),
        source_uri=payload["source_uri"],
        data_as_of=payload["data_as_of"],
        license_class=payload["license_class"],
        license_review_status=payload["license_review_status"],
        content_digest_version=payload["content_digest_version"],
        version=payload["version"])


def from_access(access, *, source=None,
                adapter: Optional[SourceAdapter] = None) -> MarketSnapshot:
    """Build a snapshot from a delivery, taking the request from the record.

    The one entry point production should use. `describe` accepts a resolution
    as an argument, which is right for a test constructing a case and wrong for
    a caller that already has one: two statements of the same request can
    disagree, and the one that would be believed is whichever the snapshot
    happened to be handed.

    So this reads `access.access_event.resolution` — the request recorded
    beside the digest, by the resolver, at the moment the frame was produced.
    That record exists precisely because nothing else could say which request
    the bytes came from, and restating it here would reintroduce the gap it was
    added to close.
    """
    event = getattr(access, "access_event", None)
    if event is None:
        raise ValueError(
            "this delivery carries no access event, so nothing says which "
            "request produced it — and a snapshot built without that cannot "
            "be resolved again")
    recorded = getattr(event, "resolution", None)
    if recorded is None:
        raise ValueError(
            f"delivery {event.access_event_id} predates recorded resolution "
            "requests. It is coherent and not reproducible, and a snapshot "
            "claiming otherwise would say the data can be fetched again when "
            "nobody knows what to ask for")

    if source is None:
        from .loader import synthetic_snapshot

        source = synthetic_snapshot()
    return describe(source, access.frame,
                    resolution=recorded.to_json(), adapter=adapter)


def describe(snapshot, observations, *, resolution: Mapping[str, Any],
             adapter: Optional[SourceAdapter] = None) -> MarketSnapshot:
    """Build the contract from a delivery, digesting what was delivered.

    The digest is taken here, over the observations being described, for the
    same reason `resolve` takes it over the frame it is about to return: a
    digest computed later describes whatever the computer was holding, and
    whether that is still the delivered data is the whole question.

    `corporate_actions` is read from the request rather than assumed. A
    reinvested resolution is served by the total-return twin, in which
    distributions are already credited — so the treatment is a fact about which
    series arrived, not a policy this layer applies.
    """
    from .access_event import frame_digest

    index = sorted(observations.index)
    return MarketSnapshot(
        snapshot_hash=frame_digest(observations),
        snapshot_id=str(getattr(snapshot, "snapshot_id", "") or NOT_DECLARED),
        dataset_id=str(getattr(snapshot, "dataset_id", "") or NOT_DECLARED),
        symbols=tuple(sorted(str(one) for one in observations.columns)),
        session_range=SessionRange(
            start=str(index[0].date() if hasattr(index[0], "date") else index[0]),
            end=str(index[-1].date() if hasattr(index[-1], "date") else index[-1]),
            sessions=len(index)),
        resolution=copy.deepcopy(dict(resolution)),
        corporate_actions=(TOTAL_RETURN if resolution.get("reinvested")
                           else PRICE_ONLY),
        calendar=str(getattr(snapshot, "calendar", "") or NOT_DECLARED),
        # Declared by the caller or declared absent. A default naming some
        # adapter would attribute this data to code that may not have produced
        # it, which is worse than saying nobody wrote it down.
        source_adapter=adapter or SourceAdapter(NOT_DECLARED, NOT_DECLARED),
        source_uri=str(getattr(snapshot, "uri", "") or NOT_DECLARED),
        data_as_of=str(getattr(snapshot, "data_as_of", "") or NOT_DECLARED),
        license_class=str(getattr(snapshot, "license_class", "")
                          or NOT_DECLARED),
        license_review_status=str(getattr(snapshot, "license_review_status", "")
                                  or NOT_DECLARED),
        content_digest_version=str(
            getattr(snapshot, "content_digest_version", "") or NOT_DECLARED))
