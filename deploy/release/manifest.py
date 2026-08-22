"""The immutable release manifest + promotion gate (freeze plan §5).

A release is identified by ONE immutable manifest binding the code, the image, and
the exact dependency versions it was built against. Canary validation and
production promotion must reference the *same* manifest — never a mutable tag or a
stale digest file (the deploy incident where a promotion picked an image
independently of the reconciled code).

    build → write_manifest(...)      # emitted beside the pushed digest
    promote → verify(manifest, actual)   # the checks below must all pass

`build_manifest`/`verify` are pure and unit-tested; the CLI wires them into
`scripts/build_image.sh` and `.github/workflows/deploy-aws.yml`.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import Any, Mapping

MANIFEST_SCHEMA = "quantify/release-manifest"
MANIFEST_SCHEMA_VERSION = "1"

#: The runtime-contracts this application line requires (freeze target). Kept in
#: step with src/deploy/preflight.REQUIRED_RUNTIME_CONTRACTS_MIN.
REQUIRED_RUNTIME_CONTRACTS_MIN = (0, 3, 0)
REQUIRED_CANONICALIZATION = "rcv1"


@dataclasses.dataclass(frozen=True)
class ReleaseManifest:
    app_commit: str
    image_digest: str
    runtime_contracts: str
    discovery_runtime: str
    canonicalization: str
    payload_schema: str
    migration_head: str = ""
    build_timestamp: str = ""     # passed in; never generated here (reproducibility)

    def to_dict(self) -> dict:
        return {"schema": MANIFEST_SCHEMA, "schema_version": MANIFEST_SCHEMA_VERSION,
                **dataclasses.asdict(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReleaseManifest":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


def _version_tuple(raw: str) -> tuple:
    out = []
    for piece in str(raw).split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def build_manifest(**kwargs) -> ReleaseManifest:
    return ReleaseManifest(**kwargs)


def verify(manifest: ReleaseManifest, actual: Mapping[str, Any]) -> list[str]:
    """The promotion gate: what `actual` (the image about to be promoted) must
    match in the manifest. Returns a list of problems; empty ⇒ safe to promote.

    Checks (freeze plan §5): image digest, app commit, runtime-contracts package,
    protocol/canonicalization, migration compatibility, and that the required
    runtime-contracts floor is satisfied. Readiness and the runtime-artifact smoke
    test run against the live pod separately (they need a running service)."""
    problems = []
    checks = [
        ("image_digest", manifest.image_digest, actual.get("image_digest")),
        ("app_commit", manifest.app_commit, actual.get("app_commit")),
        ("runtime_contracts", manifest.runtime_contracts, actual.get("runtime_contracts")),
        ("discovery_runtime", manifest.discovery_runtime, actual.get("discovery_runtime")),
        ("canonicalization", manifest.canonicalization, actual.get("canonicalization")),
    ]
    for name, want, got in checks:
        if want and got is not None and want != got:
            problems.append(f"{name}: manifest {want!r} != actual {got!r}")
    if manifest.migration_head and actual.get("migration_head") not in (None, manifest.migration_head):
        problems.append(
            f"migration_head: manifest {manifest.migration_head!r} != DB "
            f"{actual.get('migration_head')!r}")
    # the manifest itself must satisfy the required runtime-contracts floor + canon
    if _version_tuple(manifest.runtime_contracts) < REQUIRED_RUNTIME_CONTRACTS_MIN:
        problems.append(
            f"runtime_contracts {manifest.runtime_contracts} < required "
            f"{'.'.join(map(str, REQUIRED_RUNTIME_CONTRACTS_MIN))}")
    if manifest.canonicalization != REQUIRED_CANONICALIZATION:
        problems.append(
            f"canonicalization {manifest.canonicalization!r} != required "
            f"{REQUIRED_CANONICALIZATION!r}")
    return problems


def _cli(argv=None) -> int:
    p = argparse.ArgumentParser(description="release manifest build/verify")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    for f in ("app-commit", "image-digest", "runtime-contracts", "discovery-runtime",
              "canonicalization", "payload-schema", "migration-head", "build-timestamp"):
        b.add_argument(f"--{f}", default="")
    b.add_argument("--out", default="")
    v = sub.add_parser("verify")
    v.add_argument("--manifest", required=True)
    v.add_argument("--actual", required=True, help="JSON of the image about to promote")
    a = p.parse_args(argv)

    if a.cmd == "build":
        m = build_manifest(
            app_commit=a.app_commit, image_digest=a.image_digest,
            runtime_contracts=a.runtime_contracts, discovery_runtime=a.discovery_runtime,
            canonicalization=a.canonicalization, payload_schema=a.payload_schema,
            migration_head=a.migration_head, build_timestamp=a.build_timestamp)
        text = json.dumps(m.to_dict(), indent=2, sort_keys=True)
        (open(a.out, "w").write(text + "\n") if a.out else print(text))
        return 0

    manifest = ReleaseManifest.from_dict(json.load(open(a.manifest)))
    actual = json.loads(a.actual) if a.actual.strip().startswith("{") else json.load(open(a.actual))
    problems = verify(manifest, actual)
    if problems:
        print("PROMOTION REFUSED — release identity mismatch:", file=sys.stderr)
        for pr in problems:
            print(f"  - {pr}", file=sys.stderr)
        return 1
    print("promotion OK — image matches the immutable release manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
