"""One interface over three sources of the same table.

    committed fixture     synthetic, redistributable, no credentials, no network
    local cache           a previously verified copy of a pinned snapshot
    immutable object      the licensed snapshot, pinned by version id and hash

Callers say which *dataset* they need, not where it lives. That matters because
the answer changes by environment — a unit test must never reach the network, a
benchmark must reach exactly one immutable object, and a developer wants
whatever they already downloaded — while the code computing returns must not
know which of those happened.

The one thing this module will not do is silently substitute. If a pinned
snapshot cannot be produced, it raises. Falling back to the newest object, or to
the fixture, would mean a run reports figures computed against data it never
names — and the fixture in particular is invented, so quietly standing in for
licensed data would turn a test double into a published number.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import yaml

from .integrity import IntegrityError, content_digest, file_sha256, verify

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
FIXTURE_MANIFEST = REPO_ROOT / "tests" / "fixtures" / "prices_synthetic_manifest.yaml"

DEFAULT_CACHE = Path(
    os.environ.get("QUANTIFY_MARKET_DATA_CACHE",
                   str(Path.home() / ".cache" / "quantify" / "market-data")))

_ENV_REF = re.compile(r"^\$\{([A-Z0-9_]+)\}$")



class SnapshotUnavailable(RuntimeError):
    """A pinned snapshot could not be produced. Never downgraded to a fallback."""


class EgressDenied(PermissionError):
    """Data was about to leave the system by a route its licence does not permit.

    Raised, not logged. The failure mode this closes is a developer who knows
    the dataset is restricted, and a code path six months later written by
    someone who does not.
    """


class Egress(str, Enum):
    """What downstream code is about to do with the data."""

    PUBLIC_EXPORT = "public_export"
    CASE_BUNDLE = "case_bundle"
    MODEL_PROVIDER_UPLOAD = "model_provider_upload"
    DERIVED_AGGREGATE = "derived_aggregate"
    INTERNAL_BENCHMARK = "internal_benchmark"
    CUSTOMER_RESULT = "customer_result"


class Decision(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REVIEW = "REVIEW"
    """Not yet decided. Treated as a refusal at the call site — an undecided
    question answered by whoever happens to be running the code is a decision
    nobody made."""

    @property
    def permits(self) -> bool:
        return self is Decision.ALLOW


#: Routes that stay inside the system. While a licence review is open these
#: still follow the declared policy; everything else is forced to REVIEW,
#: because the difference that matters is whether a person outside sees it.
_INTERNAL_ONLY = frozenset({Egress.INTERNAL_BENCHMARK})


def _resolve(value: Any) -> Optional[str]:
    """Expand `${VAR}` from the environment, so a bucket name need not be
    committed while the snapshot identity still is."""
    if not isinstance(value, str):
        return value
    match = _ENV_REF.match(value.strip())
    if not match:
        return value
    return os.environ.get(match.group(1))


@dataclass(frozen=True)
class Snapshot:
    """What a manifest says, after environment resolution."""

    dataset_id: str
    snapshot_id: str
    kind: str
    uri: Optional[str]
    schema_version: str
    calendar: Optional[str] = None
    data_as_of: Optional[str] = None
    object_version_id: Optional[str] = None
    sha256: Optional[str] = None
    content_digest: Optional[str] = None
    license_class: str = "restricted"
    redistributable: bool = False
    license_review_status: str = "UNCONFIRMED"
    content_digest_version: str = "mdv1"
    egress_policy: Mapping[str, str] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_local(self) -> bool:
        return bool(self.uri) and not str(self.uri).startswith("s3://")

    @property
    def may_be_exported(self) -> bool:
        """Whether this data may leave the system in a public artifact.

        Read by the export path rather than assumed. A licensed snapshot that
        reached a public case bundle is a licence breach no later review undoes.
        """
        return self.redistributable and self.license_class != "restricted"

    @property
    def review_complete(self) -> bool:
        return self.license_review_status == "CONFIRMED"

    def decision_for(self, egress: "Egress") -> Decision:
        """What the policy says about one route out of the system.

        An unlisted route is DENY rather than ALLOW. A policy that permits what
        it forgot to mention grows permissions by omission.

        While the licence review is open, everything a pilot user could see is
        refused regardless of the policy — the policy describes what the
        agreement would allow, and nobody has read the agreement yet.
        """
        declared = Decision(self.egress_policy.get(egress.value, "DENY"))
        if self.review_complete or declared is Decision.DENY:
            return declared
        if egress in _INTERNAL_ONLY:
            return declared
        return Decision.REVIEW

    def check_egress(self, egress: "Egress", *, context: str = "") -> None:
        """Ask before data leaves. Raises unless the answer is ALLOW."""
        decision = self.decision_for(egress)
        if decision.permits:
            return
        where = f" ({context})" if context else ""
        raise EgressDenied(
            f"{self.snapshot_id} may not be used for {egress.value}{where}: "
            f"{decision.value}. license_class={self.license_class}, "
            f"review={self.license_review_status}. "
            + ("The licence review is still open, so nothing derived from this "
               "snapshot may reach anyone outside the system."
               if not self.review_complete else
               "The dataset policy forbids this route."))

    def describe(self) -> Dict[str, Any]:
        """What a Run records about its input. Identity, never bytes."""
        return {
            "dataset_id": self.dataset_id,
            "snapshot_id": self.snapshot_id,
            "kind": self.kind,
            "schema_version": self.schema_version,
            "calendar": self.calendar,
            "data_as_of": self.data_as_of,
            "object_version_id": self.object_version_id,
            "content_digest": self.content_digest,
            # The normalization rules that produced the digest, not only the
            # digest. A future reader who has the hash but not the rules cannot
            # reproduce it.
            "content_digest_version": self.content_digest_version,
            "license_class": self.license_class,
            "license_review_status": self.license_review_status,
        }


def load_manifest(path: Path | str) -> Snapshot:
    body = yaml.safe_load(Path(path).read_text()) or {}
    coverage = body.get("coverage") or {}
    return Snapshot(
        dataset_id=body.get("dataset_id", "unknown"),
        snapshot_id=body.get("snapshot_id", "unknown"),
        kind=body.get("kind", "licensed"),
        uri=_resolve(body.get("uri")),
        schema_version=str(body.get("schema_version", "0")),
        calendar=body.get("calendar"),
        data_as_of=str(body.get("data_as_of")) if body.get("data_as_of") else None,
        object_version_id=_resolve(body.get("object_version_id")),
        sha256=_resolve(body.get("sha256")),
        content_digest=body.get("content_digest"),
        license_class=body.get("license_class", "restricted"),
        redistributable=bool(body.get("redistributable", False)),
        license_review_status=str(body.get("license_review_status", "UNCONFIRMED")),
        content_digest_version=str(body.get("content_digest_version", "mdv1")),
        egress_policy=dict(body.get("egress_policy") or {}),
        raw={**body, "coverage": coverage},
    )


def synthetic_snapshot() -> Snapshot:
    """The committed fixture. Always available, never licensed."""
    return load_manifest(FIXTURE_MANIFEST)


def production_snapshot(name: str = "prices-production") -> Snapshot:
    return load_manifest(MANIFEST_DIR / f"{name}.yaml")


# --- loading ---------------------------------------------------------------

def load_prices(snapshot: Optional[Snapshot] = None, *,
                cache_dir: Path = DEFAULT_CACHE,
                allow_network: bool = False) -> pd.DataFrame:
    """Produce the pinned table, or raise.

    `allow_network` defaults to False so the ordinary suite cannot reach S3 by
    accident. A test that silently acquires a network dependency passes locally,
    fails in CI, and is diagnosed as flaky.
    """
    snapshot = snapshot or synthetic_snapshot()

    if snapshot.is_local:
        path = Path(snapshot.uri)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise SnapshotUnavailable(
                f"{snapshot.snapshot_id}: no file at {path}")
        frame = pd.read_parquet(path)
        verify(frame, expected_content_digest=snapshot.content_digest,
               source=str(path))
        return frame

    cached = cache_dir / snapshot.dataset_id.replace("/", "_") / \
        f"{snapshot.snapshot_id}.parquet"
    if cached.exists():
        return _verified(cached, snapshot)

    if not allow_network:
        raise SnapshotUnavailable(
            f"{snapshot.snapshot_id} is not cached at {cached} and network "
            "access was not requested. Run the market-data integration suite, "
            "or pre-populate the cache, rather than letting an ordinary test "
            "reach object storage.")

    _fetch(snapshot, cached)
    return _verified(cached, snapshot)


def _verified(path: Path, snapshot: Snapshot) -> pd.DataFrame:
    """Bytes first, then content. A truncated download can still parse."""
    if snapshot.sha256:
        actual = file_sha256(path)
        if actual != snapshot.sha256:
            raise IntegrityError(
                f"{snapshot.snapshot_id} at {path} has sha256 {actual}, pinned "
                f"as {snapshot.sha256}. The object was replaced, or the copy is "
                "incomplete. Refusing rather than using the newest version: the "
                "recorded results were computed against the pinned one.")
    frame = pd.read_parquet(path)
    verify(frame, expected_content_digest=snapshot.content_digest,
           source=snapshot.snapshot_id)
    return frame


def _fetch(snapshot: Snapshot, destination: Path) -> None:
    """Retrieve one immutable object version into the cache.

    The version id is required, not optional. Without it the request resolves to
    whatever the key holds now, which is the overwrite problem the manifest
    exists to close — same commit, same URI, different result.
    """
    if not snapshot.uri:
        raise SnapshotUnavailable(
            f"{snapshot.snapshot_id} has no URI. Set QUANTIFY_MARKET_DATA_URI "
            "or populate the manifest.")
    if not snapshot.object_version_id:
        raise SnapshotUnavailable(
            f"{snapshot.snapshot_id} names no object version id. An unversioned "
            "object can be overwritten, and then the same commit against the "
            "same URI produces a different result.")

    try:
        import boto3                                            # noqa: PLC0415
    except ImportError as exc:                                  # pragma: no cover
        raise SnapshotUnavailable(
            "boto3 is not installed; it is an extra, because the ordinary "
            "suite must not depend on object storage") from exc

    bucket, _, key = snapshot.uri[len("s3://"):].partition("/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".partial")

    logger.info("fetching %s version %s", snapshot.uri, snapshot.object_version_id)
    boto3.client("s3").download_file(
        bucket, key, str(partial),
        ExtraArgs={"VersionId": snapshot.object_version_id})
    # Renamed only once complete, so an interrupted download cannot be picked up
    # later as a cache hit.
    partial.replace(destination)


def describe_for_run(snapshot: Snapshot, frame: pd.DataFrame) -> Dict[str, Any]:
    """What a Run should record about the data it used.

    The realized content digest is computed here rather than copied from the
    manifest, so the record states what was actually loaded — which is the only
    version of this fact worth storing.
    """
    return {**snapshot.describe(),
            "realized_content_digest": content_digest(frame),
            "sessions": int(len(frame)),
            "assets": int(frame.shape[1]),
            "first_session": str(frame.index.min().date()),
            "last_session": str(frame.index.max().date())}
