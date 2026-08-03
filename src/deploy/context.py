"""Every deployment-stable fact, interpreted once.

    environment -> resolve() -> DeploymentContext -> every consumer

The preflight validated PostgreSQL and the store opened a local SQLite file.
Both read the environment; both were correct about what they read; nothing
asked whether they had read the same thing. A shared resolver made them agree,
which is *resolution parity* — they agree because both call the same function,
not because the question is answered once.

This is the answer being answered once. Consumers receive the resolved object.
Nothing below this module reads an operational identity, and
`tests/test_single_resolution.py` derives its inventory from the syntax tree
rather than a list, because the list version asserted that no route reads an
identity while a route read two.

**Resolved values, not variable names.** A context carrying `"QUANTIFY_DATABASE_URL"`
would have moved the lookup rather than removed it, and two consumers could
still interpret the same string differently.

**Deployment-stable only.** Owner, request id, trace id and budget are
request-scoped and belong to a different object. A context that accepted them
would become the next accidental control plane, and its immutability would stop
meaning anything.

**Secrets stay encapsulated.** Diagnostic paths take a redacted identity, never
the raw value: `DatabaseTarget.display` for the connection string,
`ModelTarget.available` for whether a key exists. `to_json` — which is what the
startup proof logs and what `/health/ready` reports — is built from those, so
nothing has to remember to redact at each call site. The raw values are reachable
(`url`, `api_key()`), because a store must open a database and a client must
authenticate; what they are not is the default thing a caller picks up.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .preflight import Profile, configured_profile


def _redact(url: str) -> str:
    """A connection identity safe to log, print or put in a proof record."""
    return re.sub(r"://[^@/]*@", "://***@", url or "")


@dataclass(frozen=True)
class DatabaseTarget:
    """Where records live, without spreading the credential to say so."""

    url: str
    """Private. Held here so exactly one object knows it."""

    dialect: str

    configured: bool = False
    """Whether a deployment stated this target or it was defaulted. The
    preflight refuses an unstated production target, and once a URL has been
    resolved a default is indistinguishable from a choice."""

    problem: str = ""
    """Why this target cannot be opened, if it cannot. Resolution records the
    objection and continues; deciding whether it is fatal is the preflight's
    job, and a resolver that raised would leave the preflight with nothing to
    report and no context to log."""

    @property
    def display(self) -> str:
        """What may be logged: engine and host, never credentials.

        Every diagnostic path goes through this rather than through `url`, so
        the credential is not one careless format string away from a container
        log — `to_json`, the preflight's `facts` dict and `/health/ready` all
        report `display`.
        """
        return _redact(self.url)

    def to_json(self) -> dict:
        return {"engine": self.dialect, "identity": self.display}


@dataclass(frozen=True)
class MarketDataTarget:
    """Which data a request may obtain, and under which policy."""

    policy: Optional[Any]
    policy_error: str = ""
    """Why no policy resolved. Kept because "unset" and "unusable" are
    different, and a caller that only sees `None` cannot tell them apart."""

    cache_directory: Optional[str] = None

    @property
    def configured(self) -> bool:
        return self.policy is not None

    def to_json(self) -> dict:
        return {"policy": getattr(self.policy, "value", None),
                "configured": self.configured,
                "problem": self.policy_error}


@dataclass(frozen=True)
class ModelTarget:
    """Stage 1's model configuration, resolved rather than looked up.

    `workspace/routes.py` read `ANTHROPIC_API_KEY` and `QUANTIFY_PARSER_MODEL`
    directly — a request handler deciding for itself whether a model was
    available. The key never leaves this object; consumers ask whether one is
    configured.
    """

    _api_key: Optional[str] = field(default=None, repr=False)
    model: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    def api_key(self) -> Optional[str]:
        """The one accessor. Named so a grep for it finds every use."""
        return self._api_key

    def to_json(self) -> dict:
        return {"model": self.model, "available": self.available}


@dataclass(frozen=True)
class DeploymentContext:
    """What this deployment is. Immutable, and resolved exactly once."""

    profile: Profile
    database: DatabaseTarget
    market_data: MarketDataTarget
    model: ModelTarget
    build: Any
    """The `BuildManifest`, which already resolves itself from the environment
    and is carried here so nothing re-reads it."""

    @property
    def is_production(self) -> bool:
        return self.profile is Profile.PRODUCTION

    def to_json(self) -> dict:
        """Safe to log. Carries no credential and no connection string."""
        return {"profile": self.profile.value,
                "database": self.database.to_json(),
                "market_data": self.market_data.to_json(),
                "model": self.model.to_json(),
                "build": {"observable": getattr(self.build, "observable", None)}}


#: The context this process is serving under, once a deployment has established
#: one. A holder rather than a module global so `bind` is the only way it
#: changes and the mutation is greppable.
_BOUND: dict = {"context": None}


def bind(context: DeploymentContext) -> DeploymentContext:
    """Establish the deployment. `create_app` calls this, and only it should.

    Called *after* the preflight has judged this same object, so what the
    application serves under is what was validated — not an equal object
    resolved a second time, which is what "both read the environment" already
    proved insufficient.
    """
    _BOUND["context"] = context
    return context


def bound() -> Optional[DeploymentContext]:
    """The established deployment, or `None` if nothing has established one.

    Distinct from `current()`, which resolves rather than answering `None`.
    A caller deciding *whether* to establish one needs to tell the two apart.
    """
    return _BOUND["context"]


def unbind() -> None:
    """Forget the deployment. For tests, which build several in one process."""
    _BOUND["context"] = None


def current() -> DeploymentContext:
    """The deployment in force.

    Unbound means no deployment established one — a test, a CLI invocation, a
    notebook — and the environment is then the only authority there is, so one
    is resolved. That is a fresh read, deliberately: caching it would make a
    test's `monkeypatch.setenv` silently ineffective, and a control that
    silently ignores configuration is the failure this module exists to end.
    """
    return _BOUND["context"] or resolve()


def resolve(environ: Optional[Mapping[str, str]] = None) -> DeploymentContext:
    """Read the environment. The only place that does, for these identities."""
    from ..db.engine import (
        DATABASE_URL_VAR,
        DEFAULT_SQLITE_PATH,
        dialect_of,
        resolve_target,
    )
    from ..market_data.loader import CACHE_VARIABLE
    from ..market_data.pilot_policy import PilotPolicyMissing, configured_policy
    from .manifest import read_manifest

    source = os.environ if environ is None else environ

    # The default is passed explicitly. `resolve_target(None)` now asks
    # `current()`, which lands back here — so calling it with `None` from the
    # resolver would recurse forever. Naming the fallback at this one site is
    # also what keeps it a single answer: everything below can safely say
    # "whatever the deployment decided" without any of them owning a default.
    stated = source.get(DATABASE_URL_VAR)
    url = resolve_target(stated if stated else DEFAULT_SQLITE_PATH)
    try:
        dialect, objection = dialect_of(url).value, ""
    except Exception as exc:
        dialect, objection = url.split("://")[0], str(exc)
    target = DatabaseTarget(url=url, dialect=dialect, configured=bool(stated),
                            problem=objection)

    try:
        policy = configured_policy(source)
        problem = ""
    except PilotPolicyMissing as missing:
        policy, problem = None, str(missing)[:200]

    return DeploymentContext(
        profile=configured_profile(source),
        database=target,
        market_data=MarketDataTarget(
            policy=policy, policy_error=problem,
            cache_directory=source.get(CACHE_VARIABLE)),
        model=ModelTarget(_api_key=source.get("ANTHROPIC_API_KEY"),
                          model=source.get("QUANTIFY_PARSER_MODEL")),
        build=read_manifest(source))
