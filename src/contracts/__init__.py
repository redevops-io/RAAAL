"""Vendored copy of the externally-owned runtime contracts.

**This is not ours.** `VerifiedIntent` is the Discovery → Mission boundary, and
a boundary between two runtimes cannot be owned by either — `runtime-contracts`
is where it belongs, and where it was written.

The originating commit is `redevops-io/runtime-contracts@feat/verified-intent`,
now pushed.

The reason for the copy has now changed twice, and both earlier reasons are
gone. The repository was archived and read-only when the copy was taken; it was
unarchived. It was then private while this one was public; it is now public
under AGPL-3.0-or-later WITH Commons-Clause, the same terms as this repository.

What remains is narrower and is the only thing still holding: **the contract
lives on `feat/verified-intent` and not on `main`.** Depending on an unmerged
branch pins this repository to a ref that can be force-pushed or rebased under
it — the same fragility as `mission-sdk`'s bare-commit pin on `agentic-os`,
which this project has already criticised in writing.

So the swap is unblocked but not yet correct. It becomes correct when the
contract lands on `main` and carries a tag, at which point:

    src/contracts/            deleted
    scripts/check_vendored…   deleted
    requirements              runtime-contracts @ <tag>

Until then the drift check is doing more work than before, not less: it is the
only thing keeping this copy honest against a branch that is still moving.

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
    IntentState,
    MissionOutcome,
    MissionProposal,
    NotSealable,
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
