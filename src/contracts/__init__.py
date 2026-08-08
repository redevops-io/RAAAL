"""Vendored copy of the externally-owned runtime contracts.

**This is not ours.** `VerifiedIntent` is the Discovery → Mission boundary, and
a boundary between two runtimes cannot be owned by either — `runtime-contracts`
is where it belongs, and where it was written.

The originating commit is `redevops-io/runtime-contracts@feat/verified-intent`,
now pushed.

It is copied here for a narrower reason than the first version of this note
gave. That repository was archived and read-only when the copy was taken, so it
could not be published to at all; it has since been unarchived and the branch is
upstream. It remains **private**, and `RAAAL` is public — so a clean clone of
this repository still cannot resolve it as a dependency, and a test that
depended on it would pass here and fail for anyone else. That is the same
"passes only on my machine" failure the vendored market-data manifest already
caused once.

The copy therefore stays until either the contract branch lands on `main` and
the package is published, or the repository is made public. Neither is this
repository's decision.

The rule while this copy exists: **treat it as read-only.** A change made here
is a change to a contract two runtimes are supposed to agree on, made by one of
them, in a copy the other cannot see. If something here is wrong, fix it in
`runtime-contracts` and re-copy — even though that is slower, because the
alternative is a private fork of a shared contract, which is the failure the
contract package exists to prevent.

`scripts/check_vendored_contracts.py` compares this directory against the
source and fails if they have diverged, so the copy cannot rot silently.
"""
from .intent import (  # noqa: F401
    Amendment,
    Author,
    CapabilityRefusal,
    DecisionEvidence,
    Derivation,
    IntentField,
    OpenReason,
    ReaderKind,
    RefusalKind,
    Unresolved,
    VerifiedIntent,
)

#: The upstream this was copied from, so the drift check knows where to look.
VENDORED_FROM = "redevops-io/runtime-contracts@feat/verified-intent"
VENDORED_PATH = "/projects/runtime-contracts"
