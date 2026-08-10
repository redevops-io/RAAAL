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
from enum import Enum
from typing import Any, Mapping, Optional

from .preflight import Profile, configured_profile


#: Where operational traces are written. Set empty to disable recording — an
#: explicit choice a deployment can make, distinct from the variable being
#: absent, which keeps the default.
TRACE_PATH_VAR = "QUANTIFY_TRACE_PATH"

TRACE_RETENTION_VAR = "QUANTIFY_TRACE_RETENTION_DAYS"


#: Whether the deterministic syntax witness runs beside the model on the
#: serving path. Declared, never inferred from whether Stanza imports: an
#: image that happens to have the package is not a deployment that decided to
#: use it, and `WitnessProfile` is carried onto every artifact.
SYNTAX_WITNESS_VAR = "QUANTIFY_SYNTAX_WITNESS"

#: Whether pilot participants' own sentences are retained, and for how long.
TRANSCRIPTS_VAR = "QUANTIFY_PILOT_TRANSCRIPTS"
TRANSCRIPT_RETENTION_VAR = "QUANTIFY_PILOT_TRANSCRIPT_DAYS"


def _retention(raw: Optional[str], *, default: Optional[int] = None) -> int:
    """Days, or the default. An unreadable value keeps the default rather than
    raising: a malformed retention setting must not stop a deployment serving,
    because telemetry is the expendable half."""
    if default is None:
        from ..telemetry.trace_store import DEFAULT_RETENTION_DAYS

        default = DEFAULT_RETENTION_DAYS

    try:
        days = int(str(raw))
    except (TypeError, ValueError):
        return default
    return days if days > 0 else default


def _affirmative(raw: Optional[str]) -> bool:
    """Only an explicit yes turns transcript retention on.

    Anything unrecognised reads as off. The asymmetry is deliberate: a typo in
    this variable must fail towards *not* keeping what people typed, and a
    deployment that meant to retain will notice an empty transcript store far
    sooner than a cohort would notice prose being kept it was not told about.
    """
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


#: Declared, never inferred. A deployment states which parser it is running.
PARSER_MODE_VAR = "QUANTIFY_PARSER_MODE"
PARSER_FALLBACK_VAR = "QUANTIFY_PARSER_FALLBACK"
PARSER_PROMPT_VERSION_VAR = "QUANTIFY_PARSER_PROMPT_VERSION"

#: Which reader the pilot interpreter consults. `recorded` replays fixtures
#: instead of calling the provider — for acceptance tests and for a demo
#: without a key.
#:
#: Resolved here rather than read in the route, which is the whole rule of this
#: module: a request handler deciding for itself where its answers come from is
#: how one instance serves fixtures while reporting them as the model's.
PILOT_READER_VAR = "QUANTIFY_PILOT_READER"


def _model_target(source: Mapping[str, str]) -> "ModelTarget":
    from ..mission.parse_model import PARSER_VERSION

    raw_mode = (source.get(PARSER_MODE_VAR) or "").strip().upper()
    try:
        mode = ParserMode(raw_mode) if raw_mode else ParserMode.DETERMINISTIC
    except ValueError:
        mode = ParserMode.DETERMINISTIC
    raw_fallback = (source.get(PARSER_FALLBACK_VAR) or "").strip().upper()
    try:
        fallback = (ParserFallback(raw_fallback) if raw_fallback
                    else ParserFallback.REFUSE)
    except ValueError:
        fallback = ParserFallback.REFUSE
    raw_reader = (source.get(PILOT_READER_VAR) or "").strip().upper()
    try:
        reader = PilotReader(raw_reader) if raw_reader else PilotReader.HOSTED
    except ValueError:
        reader = PilotReader.HOSTED
    return ModelTarget(
        _api_key=source.get("ANTHROPIC_API_KEY"),
        model=source.get("QUANTIFY_PARSER_MODEL"),
        mode=mode, fallback=fallback, declared=bool(raw_mode),
        pilot_reader=reader,
        syntax_witness=_affirmative(source.get(SYNTAX_WITNESS_VAR)),
        prompt_version=source.get(PARSER_PROMPT_VERSION_VAR) or PARSER_VERSION)


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


class ParserMode(str, Enum):
    MODEL_ASSISTED = "MODEL_ASSISTED"
    DETERMINISTIC = "DETERMINISTIC"

    RUNTIME = "RUNTIME"
    """The pilot interpreter: hosted reader → fusion → `VerifiedIntent` →
    `compile_intent`.

    A third *declared* mode rather than a flag on `MODEL_ASSISTED`, for the
    reason that enum's docstring already gives: a deployment states what it is,
    and an entire pilot was once measured model-assisted because a variable was
    set in a shell nobody had decided about. `RUNTIME` and `MODEL_ASSISTED`
    both call a model and produce different artifacts — one a
    `ScenarioSpecification` from prose, the other a plan compiled from a pinned
    intent — and a plan cannot say which it was unless the deployment did.

    Model-only is the *witness profile* under this mode, not a fourth mode.
    Whether the deterministic reader is installed is a property of the image;
    `discovery.witnesses` carries it onto the artifact."""


class PilotReader(str, Enum):
    HOSTED = "HOSTED"
    """Call the provider. What a real deployment does."""

    RECORDED = "RECORDED"
    """Replay recorded readings. A *declared* choice, never a fallback.

    A route that replayed fixtures because a key was missing would serve
    answers from a file and report them as the model's — the same class of
    failure as parsing deterministically under a model-assisted declaration,
    and harder to notice because the answers look right."""


class ParserFallback(str, Enum):
    REFUSE = "REFUSE"
    """A model-assisted deployment that cannot reach its model refuses the
    request. Silently parsing deterministically instead would hand two users
    different products under one deployment — one gets model-widened
    recognition, the other a narrower grammar, and neither is told."""

    EXPLICIT_DETERMINISTIC = "EXPLICIT_DETERMINISTIC"
    """Fall back, and record on the plan that it happened."""


@dataclass(frozen=True)
class ModelTarget:
    """Stage 1's parser configuration, resolved rather than looked up.

    `workspace/routes.py` read `ANTHROPIC_API_KEY` and `QUANTIFY_PARSER_MODEL`
    directly — a request handler deciding for itself whether a model was
    available. The key never leaves this object; consumers ask whether one is
    configured.

    **Mode is declared, never inferred from whether a key happens to exist.**
    An entire pilot was measured model-assisted because the variable was set in
    the shell, which nobody had decided. A deployment states what it is and the
    preflight refuses an incoherent statement.
    """

    _api_key: Optional[str] = field(default=None, repr=False)
    model: Optional[str] = None
    mode: "ParserMode" = ParserMode.DETERMINISTIC
    fallback: "ParserFallback" = ParserFallback.REFUSE
    pilot_reader: "PilotReader" = PilotReader.HOSTED
    syntax_witness: bool = False
    """Whether a second, deterministic reader constrains the model's semantics.

    Off by default because the parser is a ~500MB model that not every image
    carries. Turned on for a serving deployment because a single stochastic
    witness cannot support a safety gate: two recordings of one model, same
    prompt version, differed on 24 of 36 corpus sentences — two losing a
    `sell_action` they previously had, and one inverting `persistent_condition`
    to `crossing_event`. Syntax does not decide meaning; it stops an unstable
    reader from changing meaning between runs unnoticed.
    """

    prompt_version: str = ""

    declared: bool = False
    """Whether a deployment stated the mode or it was defaulted.

    The default is deterministic, so a developer checkout cannot become
    model-assisted because a stray key exists in a shell — which is exactly how
    an entire pilot came to be measured model-assisted without anyone deciding.

    But a *production* deployment that omits the variable would quietly serve
    the narrower product while the startup proof reported a valid
    configuration: users would get a different parser than the pilot was
    reviewed against, with fewer recognitions and different blockers, and
    nothing would say so. That is configuration drift wearing a valid default,
    so production requires the statement. `configured` is what lets the
    preflight tell "chose deterministic" from "said nothing"."""

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @property
    def model_assisted(self) -> bool:
        return self.mode is ParserMode.MODEL_ASSISTED

    @property
    def witnesses(self):
        """The declared witness profile, for the artifact to carry."""
        from ..discovery.witnesses import BOTH, MODEL_ONLY

        return BOTH if self.syntax_witness else MODEL_ONLY

    @property
    def uses_the_runtime(self) -> bool:
        """The pilot interpreter: hosted reader, fusion, pinned intent."""
        return self.mode is ParserMode.RUNTIME

    @property
    def needs_a_model(self) -> bool:
        """Both model-calling modes, so the coherence check covers each.

        Written as a property rather than repeated at the call site: adding
        `RUNTIME` to the enum without adding it here would have let a
        deployment declare the pilot interpreter with no API key and pass the
        preflight, then refuse every description at request time with the
        startup proof still reporting a valid configuration. That is the exact
        failure the `MODEL_ASSISTED` check exists to prevent, one mode later.
        """
        return self.model_assisted or self.uses_the_runtime

    def api_key(self) -> Optional[str]:
        """The one accessor. Named so a grep for it finds every use."""
        return self._api_key

    def problems(self, *, require_declaration: bool = False) -> tuple:
        """Why this parser configuration cannot be served, if it cannot.

        Reported rather than raised: the preflight decides whether a problem
        stops a deployment, and a resolver that raised would leave it with
        nothing to log.
        """
        if require_declaration and not self.declared:
            return (
                f"{PARSER_MODE_VAR} is not set. A production deployment must "
                "state which parser it runs: defaulting to deterministic would "
                "serve a narrower product than the one reviewed, with fewer "
                "recognitions and different blockers, while the startup proof "
                "reported a valid configuration",)
        if not self.needs_a_model:
            return ()
        found = []
        if not self.available:
            found.append(
                f"parser mode is {self.mode.value} and no API key is "
                "configured. "
                "Serving would mean either refusing every description or "
                "silently parsing with a narrower grammar than this "
                "deployment declares")
        if not self.model:
            found.append(
                f"parser mode is {self.mode.value} and no model is pinned. An "
                "unpinned model changes what a description means without a "
                "version anyone can cite")
        return tuple(found)

    def to_json(self) -> dict:
        return {"model": self.model, "available": self.available,
                "mode": self.mode.value, "fallback": self.fallback.value,
                "pilot_reader": self.pilot_reader.value,
                "syntax_witness": self.syntax_witness,
                "prompt_version": self.prompt_version,
                "declared": self.declared}

    def identity(self) -> dict:
        """What a *plan* records about how it was interpreted.

        Distinct from `to_json`, which reports what the service intends to use.
        A plan must say what it actually used, because deployment
        configuration moves and a stored interpretation must not be re-read
        against a parser that has since changed — the same rule market-data
        provenance already follows.
        """
        return {"mode": self.mode.value,
                "provider": "anthropic" if self.model_assisted else "",
                "model": self.model if self.model_assisted else "",
                "prompt_version": self.prompt_version}


@dataclass(frozen=True)
class TelemetryTarget:
    """Where operational traces go, and whether they go anywhere.

    Separate from `database` on purpose, and not merely as a different file.
    Telemetry is expendable: it expires on a retention policy while financial
    artifacts do not, and it must be able to fail without taking a request with
    it. One target would make retention a per-table convention some future
    query forgets, and would put deletable rows in the transaction that writes
    permanent ones.

    Omitted from the first version of this context because nothing consumed it.
    That was right at the time — a declared field no producer fills is the
    defect `discriminating strictness` names — and it is here now because
    `plan_and_record` reaches it.
    """

    path: Optional[str]
    """`None` disables recording entirely, which is what every financial test
    runs under: the system must behave identically whether or not it is being
    watched."""

    retention_days: int = 90

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    def store(self):
        """A `TraceStore`, or `None`. Constructing it can fail — a read-only
        volume, a full disk — and that failure must not reach a request, so it
        is caught here rather than at the call site that merely wanted to
        record a span."""
        if not self.path:
            return None
        try:
            from ..telemetry.trace_store import TraceStore

            return TraceStore(self.path)
        except Exception:                                        # noqa: BLE001
            return None

    def to_json(self) -> dict:
        return {"enabled": self.enabled,
                "retention_days": self.retention_days}


@dataclass(frozen=True)
class StudyTarget:
    """Whether this deployment keeps what pilot participants typed.

    Separate from `telemetry` because it is a different kind of thing held for a
    different reason. Telemetry is counts, it needs no permission, and it is
    expendable. A transcript is a person's own words, kept so an interview can
    quote them accurately instead of from the interviewer's memory — and the
    only honest default for that is off.

    **Declared, never inferred.** Not derived from `PARSER_MODE=RUNTIME`, and
    not switched on because a pilot happens to be running: a deployment that
    started retaining prose because some other variable changed would be
    keeping people's words on the strength of a side effect. Whoever set this
    is the person who told the cohort it was set.
    """

    retain_transcripts: bool = False
    retention_days: int = 30
    """Shorter than telemetry's 90 by default. Counts stay useful long after a
    pilot; the sentence someone typed is evidence for one conversation, and
    keeping it past that conversation is keeping it for no stated reason."""

    def to_json(self) -> dict:
        return {"retain_transcripts": self.retain_transcripts,
                "retention_days": self.retention_days}


@dataclass(frozen=True)
class DeploymentContext:
    """What this deployment is. Immutable, and resolved exactly once."""

    profile: Profile
    database: DatabaseTarget
    market_data: MarketDataTarget
    model: ModelTarget
    telemetry: "TelemetryTarget"
    study: "StudyTarget"
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
                "telemetry": self.telemetry.to_json(),
                "study": self.study.to_json(),
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
    from ..telemetry.trace_store import (
        DEFAULT_PATH as DEFAULT_TRACE_PATH,
        DEFAULT_RETENTION_DAYS,
    )
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
        model=_model_target(source),
        telemetry=TelemetryTarget(
            path=source.get(TRACE_PATH_VAR, str(DEFAULT_TRACE_PATH)) or None,
            retention_days=_retention(source.get(TRACE_RETENTION_VAR))),
        study=StudyTarget(
            retain_transcripts=_affirmative(source.get(TRANSCRIPTS_VAR)),
            retention_days=_retention(source.get(TRANSCRIPT_RETENTION_VAR),
                                      default=30)),
        build=read_manifest(source))
