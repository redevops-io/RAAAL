"""Vendored copy of the externally-owned runtime contracts.

**This is not ours.** `VerifiedIntent` is the Discovery → Mission boundary, and
a boundary between two runtimes cannot be owned by either — `runtime-contracts`
is where it belongs, and where it was written.

It is copied here because that repository is **archived and read-only on
GitHub**, so it cannot currently be depended on or published to. The originating
commit is on the local branch `feat/verified-intent` in
`redevops-io/runtime-contracts`, ready to push if the repository is unarchived.

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
