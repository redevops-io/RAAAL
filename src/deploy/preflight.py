"""What must be true before a production instance serves anything.

    validate build identity
        -> validate the database URL
        -> connect
        -> validate the PostgreSQL version
        -> validate the migration head
        -> validate schema parity
        -> ready

Every step has its own outcome. Collapsing them into one "startup failed" would
throw away the only thing an operator needs at three in the morning — *which*
of these is wrong — and the public surface can still report a single
`unavailable` without the internals being flattened to match it.

**The URL is judged before anything opens it.** `Database` creates the parent
directory of a SQLite path on construction, so checking the dialect after
building one would already have written to disk in production. The check reads
the resolved URL string and nothing else.

**There is no fallback.** Absent configuration, `resolve_target` falls back to
`data/workspace.db`, which is correct for a developer checkout and is exactly
the shape of the `_prices()` bypass in production: a live path quietly reading
something nobody authorised. Production refuses instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence

#: Which profile a process is running under. Production is the only one that
#: refuses SQLite, and it must be asked for explicitly — defaulting to it would
#: break every developer checkout, and defaulting *away* from it would let a
#: deployment run unchecked because someone forgot a variable.
PROFILE_VAR = "QUANTIFY_DEPLOYMENT_PROFILE"


class Profile(str, Enum):
    PRODUCTION = "production"
    LOCAL = "local"
    TEST = "test"


class Result(str, Enum):
    READY = "READY"
    REFUSED_CONFIGURATION = "REFUSED_CONFIGURATION"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    UNSUPPORTED_DATABASE = "UNSUPPORTED_DATABASE"
    MIGRATION_MISMATCH = "MIGRATION_MISMATCH"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    BUILD_UNOBSERVABLE = "BUILD_UNOBSERVABLE"


#: The major version the production lane has actually been proven against.
#:
#: Stated as *proven*, not as a ceiling. A later major is not unsupported
#: forever; it is unsupported until the same lane passes against it, which is a
#: day's work and not a re-architecture.
PROVEN_POSTGRES_MAJOR = 16

#: How long to wait for a database that may not be there. Short and bounded:
#: an unreachable database should make a deployment unready quickly, not hang
#: a rollout while a connection attempt blocks.
CONNECT_TIMEOUT_SECONDS = 5


@dataclass
class Preflight:
    """The outcome, with an operator's view and a client's view kept apart."""

    result: Result
    profile: Profile
    detail: str = ""
    """Private. May name hosts, versions and revisions."""

    facts: Dict[str, Any] = field(default_factory=dict)
    checked_at: str = ""

    @property
    def ready(self) -> bool:
        return self.result is Result.READY

    def proof(self) -> Dict[str, Any]:
        """The startup proof record. Carries no credentials or network detail."""
        return {"deployment_profile": self.profile.value,
                "database": self.facts.get("database", {}),
                "build": self.facts.get("build", {}),
                "checked_at": self.checked_at,
                "result": self.result.value}

    def public(self) -> Dict[str, Any]:
        """What a client may see: ready or not, and nothing about why."""
        return {"ready": self.ready}


def configured_profile(environ: Mapping[str, str]) -> Profile:
    """Interpret a profile. The mapping is required — see `configured_policy`."""
    raw = (environ.get(PROFILE_VAR) or Profile.LOCAL.value).strip().lower()
    try:
        return Profile(raw)
    except ValueError:
        raise ValueError(
            f"{PROFILE_VAR}={raw!r} is not a deployment profile. "
            f"Expected one of {', '.join(one.value for one in Profile)}")


def _postgres_major(version: str) -> Optional[int]:
    match = re.search(r"PostgreSQL\s+(\d+)", version)
    return int(match.group(1)) if match else None


def run(environ: Optional[Mapping[str, str]] = None,
        checked_at: str = "", context: Optional[Any] = None) -> Preflight:
    """The whole preflight, in order, stopping at the first refusal.

    Inspects a resolved `DeploymentContext` and reads nothing itself. The
    preflight validated PostgreSQL while the store opened SQLite because both
    read the environment separately and neither could be wrong on its own
    terms; judging the object the application will actually use is what makes
    that class of divergence impossible rather than merely unlikely. Passing
    `environ` resolves a context through the one resolver — a convenience for
    callers describing a hypothetical deployment, not a second reader.
    """
    from ..db.engine import DATABASE_URL_VAR, Dialect
    from .context import resolve

    if context is None:
        context = resolve(environ)
    profile = context.profile
    facts: Dict[str, Any] = {}

    def refuse(result: Result, detail: str) -> Preflight:
        return Preflight(result=result, profile=profile, detail=detail,
                         facts=facts, checked_at=checked_at)

    # 1. Build identity. Checked first because a deployment that cannot say
    #    what it is cannot be diagnosed when a later step fails.
    manifest = context.build
    facts["build"] = {"observable": manifest.observable,
                      **{k: v for k, v in manifest.deployment.items()
                         if k in ("commit", "release_ref", "image_digest",
                                  "snapshot_id")},
                      "migration_head": manifest.versions.get("migration_head"),
                      "scope_schema_version":
                          manifest.versions.get("scope_schema_version"),
                      "canonicalization_version":
                          manifest.versions.get("canonicalization_version")}
    if profile is Profile.PRODUCTION and not manifest.observable:
        return refuse(
            Result.BUILD_UNOBSERVABLE,
            "the build cannot state its own provenance; missing "
            f"{', '.join(manifest.missing)}. A deployment that cannot say "
            "which code it is running cannot be diagnosed when something else "
            "fails, and package self-report answers a different question")

    # 1b. The parser this deployment claims to be running.
    #
    #     An entire pilot was measured model-assisted because ANTHROPIC_API_KEY
    #     happened to be set in a shell. Mode is declared now, and a declaration
    #     that cannot be served is refused here rather than discovered by the
    #     first user whose description needs the model.
    facts["parser"] = context.model.to_json()
    parser_problems = context.model.problems(
        require_declaration=profile is Profile.PRODUCTION)
    if parser_problems and profile is Profile.PRODUCTION:
        return refuse(Result.REFUSED_CONFIGURATION, "; ".join(parser_problems))

    # 2. The target, judged before anything opens it.
    if profile is Profile.PRODUCTION and not context.database.configured:
        return refuse(
            Result.REFUSED_CONFIGURATION,
            f"{DATABASE_URL_VAR} is not set. There is no production fallback: "
            "defaulting to a local SQLite file would be a live path quietly "
            "reading a database nobody authorised")

    url = context.database.url
    if context.database.problem:
        return refuse(Result.REFUSED_CONFIGURATION, context.database.problem)
    dialect = Dialect(context.database.dialect)

    # The engine, and nothing about where. `facts` becomes the startup proof, a
    # durable artifact, and a redacted URL still names the host and port — that
    # is network topology, not provenance. The host belongs in the operator log
    # alongside `detail`, which already names it.
    facts["database"] = {"engine": dialect.value}
    if profile is Profile.PRODUCTION and dialect is not Dialect.POSTGRESQL:
        return refuse(
            Result.REFUSED_CONFIGURATION,
            f"the production profile requires PostgreSQL and is configured for "
            f"{dialect.value}. SQLite proves domain behaviour and cannot "
            "evidence locking, constraint enforcement, upsert semantics or "
            "migration parity — see docs/Runbook.md")

    if dialect is not Dialect.POSTGRESQL:
        # SQLite is created by `create_all` and has no migration history to be
        # at odds with, so there is nothing further to establish.
        return Preflight(result=Result.READY, profile=profile, facts=facts,
                         checked_at=checked_at,
                         detail=f"{profile.value} profile on {dialect.value}; "
                                "no migration history to check")

    # The database checks run for *any* profile pointed at PostgreSQL. What the
    # profile decides is whether a refusal stops the service, not whether the
    # question is asked — a developer pointed at an unmigrated database wants
    # to know, and only production refuses to serve.
    # Last, because it is the expensive one.
    #
    # Loading the parser takes seconds and every check above is a string
    # comparison or a socket. An operator at three in the morning wants to
    # learn the database URL is wrong immediately, not after a model has
    # loaded — and this was first placed between two configuration checks for
    # no reason but where it was easy to insert.
    problem = _syntax_witness_problem(context, profile)
    if problem:
        return refuse(Result.REFUSED_CONFIGURATION, problem)

    return _check_database(url, profile, facts, checked_at)


def _syntax_witness_problem(context, profile: Profile) -> str:
    """Why the declared syntax witness cannot serve, if it cannot.

    Empty when the deployment declared none — that is a profile, not a fault —
    and empty outside production, where a developer without a 500MB model
    should still get a running service and the MODEL_ONLY behaviour it implies.

    Loading is the check. A pipeline that constructs has its model on disk; an
    import does not, and the reader refuses to download one at request time.
    """
    if not getattr(getattr(context, "model", None), "syntax_witness", False):
        return ""
    if profile is not Profile.PRODUCTION:
        return ""

    try:
        from ..discovery.syntax_stanza import StanzaReader

        StanzaReader("en").parse("a smoke sentence")
    except Exception as failure:                               # noqa: BLE001
        return (f"{PROFILE_VAR}=production declares a syntax witness and it "
                f"could not parse: {failure.__class__.__name__}. The parser's "
                "model is not fetched at request time by design, so it has to "
                "be in the image. Serving anyway would report WitnessProfile="
                "BOTH on every plan while one witness silently never spoke — "
                "and the guards that refuse unsupported families run only on "
                "that witness.")
    return ""


def _check_database(url: str, profile: Profile, facts: Dict[str, Any],
                    checked_at: str) -> Preflight:
    """Connect, then everything that requires a connection."""
    from ..db.engine import Database
    from ..db.migrate import applied_revision, code_head

    def refuse(result: Result, detail: str) -> Preflight:
        return Preflight(result=result, profile=profile, detail=detail,
                         facts=facts, checked_at=checked_at)

    # 3. Connect. The original exception is kept for logs and never returned.
    database = Database(url)
    try:
        import psycopg

        with psycopg.connect(url,
                             connect_timeout=CONNECT_TIMEOUT_SECONDS) as conn:
            version = conn.execute("SELECT version()").fetchone()[0]
    except Exception as exc:
        return refuse(
            Result.DATABASE_UNAVAILABLE,
            f"could not reach the configured database: {type(exc).__name__}")

    # 4. Version.
    major = _postgres_major(version)
    facts["database"]["version"] = version.split(" on ")[0].replace(
        "PostgreSQL ", "")
    if major is None or major < PROVEN_POSTGRES_MAJOR:
        return refuse(
            Result.UNSUPPORTED_DATABASE,
            f"PostgreSQL {major} is running and the production lane has been "
            f"proven against {PROVEN_POSTGRES_MAJOR}. A later major is not "
            "unsupported forever — it is unsupported until the same lane "
            "passes against it")

    # 5. Migration head, in both directions.
    expected = code_head()
    actual = applied_revision(database)
    facts["database"]["migration_head"] = actual
    if actual is None:
        return refuse(
            Result.MIGRATION_MISMATCH,
            f"the database has never been migrated; this build expects "
            f"{expected}")
    if actual != expected:
        return refuse(
            Result.MIGRATION_MISMATCH,
            f"the database is at {actual} and this build expects {expected}. "
            "A database *ahead* of the application is not safe either: it may "
            "encode semantics this code does not know about")

    # 6. Schema parity, against the connected database rather than a scratch
    #    one. A freshly migrated scratch database proves the migrations agree
    #    with the model; only this catches a hand-edited column, a partially
    #    applied migration or a dropped index in the instance being started.
    drift = schema_drift(database)
    facts["database"]["schema_parity"] = "PASS" if not drift else "FAIL"
    if drift:
        return refuse(
            Result.SCHEMA_MISMATCH,
            "the deployed schema does not match the model: "
            + "; ".join(str(one) for one in drift[:5]))

    return Preflight(result=Result.READY, profile=profile, facts=facts,
                     checked_at=checked_at,
                     detail="preflight passed")


def schema_drift(database) -> List[Any]:
    """Differences between the connected database and the model."""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from ..db.schema import metadata

    engine = database.sqlalchemy_engine()
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection, opts={"compare_type": True})
            return [one for one in compare_metadata(context, metadata)
                    if not (isinstance(one, tuple) and one
                            and one[0] == "modify_default")]
    finally:
        engine.dispose()
