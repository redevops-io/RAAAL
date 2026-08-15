"""The factual record that one request received one realized frame.

    MarketDataProvenance   what the authorized source *is*
    MarketDataAccessEvent  what a specific run was *given*

Kept apart deliberately. Provenance is a reusable description of a snapshot and
the decision that permitted it; two runs a month apart under the same policy
carry identical provenance and are not evidence of the same delivery. The event
is the delivery: this request, these columns, this many rows, this digest, at
this instant.

Without it, a stored run cites what its producer *declared* it used. The
producer is the one component whose claim is not independent evidence — it is
exactly the thing a defect would corrupt — and the run path has already been
found dropping the resolver's answer while looking correct.

    run_id allocated -> resolve() -> digest the returned frame
        -> event -> compute -> commit run citing the event

**The digest is computed inside `resolve`, over the frame it is about to
return.** Not by the caller afterwards. A caller-computed digest reintroduces
the seam this closes: the value would describe whatever the caller happened to
be holding, which is the thing in question. `MarketDataAccess` therefore carries
frame, provenance and event as one inseparable value.

**What a digest does and does not prove.** It proves the resolver returned
exactly these canonical rows. It does not prove that downstream code did not
drop rows, reorder them, substitute another frame or mutate it before
simulating. That gap is closed separately, by recording an
`execution_input_digest` over the frame actually handed to the engine: equal
digests mean the transformation was the identity, and an unequal pair must name
a declared, versioned transformation or the run is not verifiable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .provenance import AccessDecision, MarketDataProvenance

ACCESS_EVENT_VERSION = "market-data-access-event@1"

#: How a frame is reduced to bytes. Versioned because changing it changes every
#: digest, and a comparison across the change must be able to say "these were
#: computed under different rules" rather than "these differ".
FRAME_DIGEST_VERSION = "mdf1"

#: How a resolution request is spelled. Versioned for the same reason, and for
#: one more: adding a parameter changes what "the same request" means.
RESOLUTION_VERSION = "mdr1"


@dataclass(frozen=True)
class ResolutionRequest:
    """What was asked for, beyond which snapshot.

    The snapshot says which observations exist. This says which of them were
    delivered and in what form, and **the two together determine the frame** —
    `resolve(reinvested=True)` and `resolve(reinvested=False)` return different
    frames, with different digests, from the same snapshot. A record naming
    only the snapshot therefore cannot be recomputed, and a verifier that tried
    would compare a digest against a frame nobody delivered.

    That is not hypothetical. The journey test recomputed a digest without this
    and compared a price-return frame against a total-return one; it read as
    "resolution is not deterministic", which would have condemned the whole
    snapshot-by-hash design. Resolution was deterministic. The record was
    incomplete.

    Versioned because adding a parameter changes the meaning of an existing
    record: an `mdr1` event carries no value for a parameter invented later,
    and must be reported as unreproducible under the newer rule rather than
    quietly reproduced using today's default for it.
    """

    reinvested: bool
    version: str = RESOLUTION_VERSION

    def to_json(self) -> Dict[str, Any]:
        return {"reinvested": bool(self.reinvested), "version": self.version}

    def as_arguments(self) -> Dict[str, Any]:
        """The keyword arguments that reproduce this delivery.

        Named so a verifier can `resolve(context=..., **request.as_arguments())`
        without knowing which parameters exist. A verifier that listed them
        itself would go on passing the day one was added, silently reproducing
        with a default — which is the failure this whole record exists to stop.
        """
        return {"reinvested": bool(self.reinvested)}


def resolution_from_json(payload: Optional[Mapping[str, Any]]
                         ) -> Optional["ResolutionRequest"]:
    """Read a stored request. Absent stays absent rather than becoming a default.

    An event written before requests were recorded genuinely does not say
    whether dividends were reinvested. Defaulting it here would manufacture an
    answer and let such an event be "verified" against a frame it may never
    have been delivered.
    """
    if not payload:
        return None
    return ResolutionRequest(
        reinvested=bool(payload.get("reinvested", False)),
        version=str(payload.get("version", RESOLUTION_VERSION)))


class UndigestibleFrame(TypeError):
    """The resolver was handed something it cannot canonically describe."""


def frame_digest(frame: Any) -> str:
    """A digest of the exact canonical content of a price frame.

    Canonicalised on the *application's* values, never on the engine's or the
    library's representation: columns sorted, index sorted and rendered as ISO
    dates, and each value spelled with `repr`, which round-trips a float
    exactly in Python 3. A digest taken over `to_string()` or a pickle would
    change when pandas changed its formatting, which would report every stored
    run as tampered on a library upgrade.

    Missing values are spelled `null` rather than skipped, because a dropped
    cell and an absent one must not produce the same bytes — that is precisely
    the substitution this is meant to detect.
    """
    if frame is None:
        raise UndigestibleFrame(
            "no frame to digest; a resolution that produced no data has no "
            "delivery to record")
    try:
        columns = sorted(str(one) for one in frame.columns)
        ordered = frame[columns].sort_index()
    except (AttributeError, KeyError, TypeError) as exc:
        raise UndigestibleFrame(
            f"{type(frame).__name__} is not a frame this can canonicalise"
        ) from exc

    # `.tolist()` rather than cell-by-cell `.at[]`: this runs on every request
    # that touches market data, and the indexed form took 300ms on the pilot
    # frame — a cost that would have been paid per page view and blamed on the
    # database. It also converts numpy scalars to Python floats, which matters
    # for correctness and not only speed: `repr(np.float64(1.0))` is
    # `'np.float64(1.0)'` on numpy 2 and `'1.0'` on numpy 1, so digesting numpy
    # scalars would silently change every stored digest on an upgrade.
    stamps = [_stamp(one) for one in ordered.index]
    rows = [f"{stamp}|" + "|".join(
                "null" if cell != cell else repr(cell)      # NaN != NaN
                for cell in cells)
            for stamp, cells in zip(stamps, ordered.to_numpy().tolist())]

    body = "\n".join([FRAME_DIGEST_VERSION, ",".join(columns), *rows])
    return f"{FRAME_DIGEST_VERSION}:{hashlib.sha256(body.encode()).hexdigest()}"


def _stamp(value: Any) -> str:
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def provenance_digest(provenance: MarketDataProvenance) -> str:
    """A digest of the provenance record, so the event pins which one it meant.

    Two provenances differing only in `accessed_at` are different records, and
    an event that cited only a snapshot id could not tell them apart.
    """
    body = json.dumps(provenance.to_json(), sort_keys=True, separators=(",", ":"))
    return "mdp1:" + hashlib.sha256(body.encode()).hexdigest()


@dataclass(frozen=True)
class TimeRange:
    """The span actually delivered, which is not the span requested."""

    start: str
    end: str

    def to_json(self) -> Dict[str, str]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class MarketDataAccessEvent:
    """One delivery of market data to one execution. Immutable, append-only."""

    access_event_id: str
    request_id: str
    run_id: Optional[str]
    """Allocated before computation where possible, so the chain needs no
    later binding step. `None` only for a read that will never become a run —
    a preview, a dashboard — and a run may not cite such an event."""

    snapshot_id: Optional[str]
    provenance_digest: str
    frame_digest: str

    selected_columns: Tuple[str, ...]
    row_count: int
    time_range: Optional[TimeRange]

    policy_version: str
    access_decision: AccessDecision
    accessed_at: str
    version: str = ACCESS_EVENT_VERSION

    resolution: Optional[ResolutionRequest] = None
    """What was asked for, so the delivery can be recomputed. `None` only for
    events written before requests were recorded, which are verifiable as
    self-consistent and not reproducible — a distinction `reproducible` keeps
    rather than blurring."""

    @property
    def identifies_delivery(self) -> bool:
        """Whether this names a specific realized frame, not merely a source."""
        return bool(self.frame_digest) and bool(self.provenance_digest) \
            and self.row_count > 0

    @property
    def reproducible(self) -> bool:
        """Whether the delivery can be resolved again and checked.

        Separate from `identifies_delivery`: an event can name a specific frame
        exactly and still not say which request produced it, and those are
        different deficiencies. The first makes the record useless as evidence;
        the second makes it uncheckable against the data.
        """
        return self.identifies_delivery and self.resolution is not None \
            and self.resolution.version == RESOLUTION_VERSION

    def content_hash(self) -> str:
        """Over the whole event, so a tampered field is detectable.

        Excludes nothing. A hash over a subset would let the excluded fields be
        edited without trace, and the fields most worth editing are the ones a
        careless exclusion would pick.
        """
        body = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return "mde1:" + hashlib.sha256(body.encode()).hexdigest()

    def to_json(self) -> Dict[str, Any]:
        body = {"access_event_id": self.access_event_id,
                "request_id": self.request_id, "run_id": self.run_id,
                "snapshot_id": self.snapshot_id,
                "provenance_digest": self.provenance_digest,
                "frame_digest": self.frame_digest,
                "selected_columns": list(self.selected_columns),
                "row_count": self.row_count,
                "time_range": self.time_range.to_json() if self.time_range
                else None,
                "policy_version": self.policy_version,
                "access_decision": self.access_decision.value,
                "accessed_at": self.accessed_at, "version": self.version}
        # Omitted rather than spelled `null` when absent, so events written
        # before requests were recorded still hash to the value stored beside
        # them. A key that always appeared would change every historical
        # content hash and report every existing delivery as edited — the exact
        # false alarm `frame_digest` avoids by refusing to digest pandas'
        # formatting.
        if self.resolution is not None:
            body["resolution"] = self.resolution.to_json()
        return body


def from_json(payload: Optional[Mapping[str, Any]]
              ) -> Optional[MarketDataAccessEvent]:
    """Read a stored event. Absence stays absence rather than becoming a blank."""
    if not payload:
        return None
    span = payload.get("time_range")
    return MarketDataAccessEvent(
        access_event_id=payload["access_event_id"],
        request_id=payload.get("request_id", ""),
        run_id=payload.get("run_id"),
        snapshot_id=payload.get("snapshot_id"),
        provenance_digest=payload.get("provenance_digest", ""),
        frame_digest=payload.get("frame_digest", ""),
        selected_columns=tuple(payload.get("selected_columns") or ()),
        row_count=int(payload.get("row_count") or 0),
        time_range=TimeRange(span["start"], span["end"]) if span else None,
        policy_version=payload.get("policy_version", ""),
        access_decision=AccessDecision(payload["access_decision"]),
        accessed_at=payload.get("accessed_at", ""),
        version=payload.get("version", ACCESS_EVENT_VERSION),
        resolution=resolution_from_json(payload.get("resolution")))


def build(*, access_event_id: str, request_id: str, run_id: Optional[str],
          frame: Any, provenance: MarketDataProvenance,
          policy_version: str, decision: AccessDecision,
          accessed_at: str,
          resolution: Optional[ResolutionRequest] = None
          ) -> MarketDataAccessEvent:
    """Describe a delivery from the frame that is actually being delivered.

    Called only by `access.resolve`, which is the one place holding the frame,
    the provenance and the request at the same instant. `resolution` is passed
    rather than inferred from the frame for the same reason the digest is taken
    here: a value derived after the fact describes what the deriver believed,
    and that belief is the thing in question.
    """
    columns = tuple(sorted(str(one) for one in frame.columns))
    index = sorted(frame.index)
    span = TimeRange(_stamp(index[0]), _stamp(index[-1])) if len(index) else None
    return MarketDataAccessEvent(
        access_event_id=access_event_id, request_id=request_id, run_id=run_id,
        snapshot_id=provenance.snapshot_id,
        provenance_digest=provenance_digest(provenance),
        frame_digest=frame_digest(frame),
        selected_columns=columns, row_count=len(index), time_range=span,
        policy_version=policy_version, access_decision=decision,
        accessed_at=accessed_at, resolution=resolution)


def verify(stored: Mapping[str, Any]) -> Sequence[str]:
    """Problems with a stored event, read from the record itself.

    Consults nothing current. The question is whether the record is internally
    coherent and self-consistent, not whether it matches today's configuration
    — a run made under snapshot A must not be reported as invalid because the
    default moved to B.
    """
    event = from_json(stored)
    if event is None:
        return ("no access event was stored",)

    problems = []
    if not event.frame_digest:
        problems.append(
            "no frame digest, so the event names a source rather than a "
            "delivery and proves nothing about what was consumed")
    elif not event.frame_digest.startswith(FRAME_DIGEST_VERSION + ":"):
        problems.append(
            f"frame digest {event.frame_digest.split(':')[0]!r} was computed "
            f"under a different rule than {FRAME_DIGEST_VERSION}")
    if not event.provenance_digest:
        problems.append("no provenance digest; the event cannot say which "
                        "provenance record it was delivered under")
    if event.access_decision is AccessDecision.DENIED:
        problems.append(
            "a delivery is recorded for a DENIED decision, which should not "
            "have yielded data at all")
    if event.row_count <= 0:
        problems.append("a delivery of no rows is not a delivery")
    if event.row_count and not event.selected_columns:
        problems.append("rows were delivered with no columns named")
    if not event.accessed_at:
        problems.append("no access time")
    if not event.request_id:
        problems.append("no request identity, so the delivery names no consumer")

    # The stored hash must be the hash of the stored body. Checked here rather
    # than at read time by each consumer, which is how one consumer ends up not
    # checking.
    claimed = stored.get("content_hash")
    if claimed and claimed != event.content_hash():
        problems.append(
            "the stored content hash does not match the stored body; this "
            "event has been edited since it was written")
    return tuple(problems)
