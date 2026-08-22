"""The immutable release manifest + promotion gate (freeze plan §5): canary and
production must reference the same release identity, and a mismatched image (the
stale-digest incident) is refused rather than promoted."""
from __future__ import annotations

import pytest

from deploy.release.manifest import ReleaseManifest, build_manifest, verify

GOOD = dict(
    app_commit="c8b5a8b", image_digest="sha256:a1c461dd",
    runtime_contracts="0.3.0", discovery_runtime="0.1.10",
    canonicalization="rcv1", payload_schema="redevops/strategy-selection",
    migration_head="a7d2f5e91bc4")


def _actual(**over):
    a = {k: GOOD[k] for k in ("app_commit", "image_digest", "runtime_contracts",
                              "discovery_runtime", "canonicalization", "migration_head")}
    a.update(over)
    return a


def test_manifest_round_trips():
    m = build_manifest(**GOOD)
    assert ReleaseManifest.from_dict(m.to_dict()) == m
    assert m.to_dict()["schema"] == "quantify/release-manifest"


def test_a_matching_image_promotes():
    assert verify(build_manifest(**GOOD), _actual()) == []


def test_a_stale_digest_is_refused():
    # the incident: promotion picked a different image than the manifest
    problems = verify(build_manifest(**GOOD), _actual(image_digest="sha256:1fd2e6ca"))
    assert any("image_digest" in p for p in problems)


def test_a_downgraded_runtime_contracts_is_refused():
    problems = verify(build_manifest(**GOOD), _actual(runtime_contracts="0.2.4"))
    assert any("runtime_contracts" in p for p in problems)


def test_a_manifest_below_the_required_floor_is_refused():
    stale = {**GOOD, "runtime_contracts": "0.2.4"}
    problems = verify(build_manifest(**stale), _actual(runtime_contracts="0.2.4"))
    assert any("required" in p for p in problems)


def test_a_migration_head_mismatch_is_refused():
    problems = verify(build_manifest(**GOOD), _actual(migration_head="f4b81e7c9a26"))
    assert any("migration_head" in p for p in problems)


def test_commit_and_canonicalization_mismatches_are_refused():
    assert any("app_commit" in p for p in verify(build_manifest(**GOOD), _actual(app_commit="deadbee")))
    assert any("canon" in p.lower() for p in verify(build_manifest(**GOOD), _actual(canonicalization="rcv2")))
