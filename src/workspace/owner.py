"""Whose workspace this request is in.

Every plan on this deployment has belonged to the literal string `"pilot"`. That
was tolerable while one shared basic-auth password guarded the door: everybody
inside had been let in deliberately, and "the pilot user" was an honest name for
them collectively.

It stopped being tolerable the moment registration opened. Anyone with an email
could create an account, sign in, and read somebody else's plans — because the
session gate asked *whether* you were signed in and never *who you are*. The two
halves of that change shipped hours apart and the gap between them was the
exposure.

**The subject, not the email.** A verified token's `sub` is the provider's
stable identifier for a person. An email address is a display name that people
change, and a plan keyed by one would move to whoever inherited the address.

**And a deployment with no provider keeps `"pilot"`.** That configuration is
still legitimate — one shared credential, one workspace, nobody pretending
otherwise — and it is what the pilot ran under for months. Returning a subject
where there is no identity would be inventing a tenant.
"""
from __future__ import annotations

#: What a deployment with no identity provider calls its single workspace.
#:
#: Kept as a name rather than removed. It is the honest owner of a deployment
#: where everybody shares one credential, and the rows already written under it
#: are not orphaned by this change — they simply belong to that workspace.
SHARED = "pilot"


def current() -> str:
    """The owner every read and write in this request is scoped to.

    Read from the verified session, never from a form field or a query
    parameter. An owner a request can *state* is an owner a request can
    choose, and choosing somebody else's is the whole attack.

    Falls back to `SHARED` rather than raising. The private surfaces are
    already behind `require_a_signed_in_viewer`, so a request reaching a store
    without a session on a deployment that has accounts is a bug in the middle
    rather than a user action — and answering it with one workspace's data is a
    smaller failure than a 500 in a store.
    """
    from .routes import _LOOKING

    viewer = _LOOKING.get()
    subject = getattr(viewer, "subject", "") if viewer is not None else ""
    return subject or SHARED


def describe() -> str:
    """What to show a person about whose workspace they are in."""
    owner = current()
    return "the shared pilot workspace" if owner == SHARED else owner
