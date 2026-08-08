"""Questions several components depend on are answered once.

`WorkspaceStore()` substituted a default before the resolver could read the
configured database URL. The preflight validated PostgreSQL and the application
wrote to a local SQLite file — both correct about the question they asked, and
nothing asked whether they had asked the same question.

**This test was previously an instance of the defect it checks.** It scanned
`src/` for string constants matching a hard-coded list of three variable names,
and contained an assertion literally named "no route reads an operational
identity". `src/workspace/routes.py` read `ANTHROPIC_API_KEY` and
`QUANTIFY_PARSER_MODEL`, and the assertion passed — because the scan was
parametrised from the same list the assertion was about, so a reader of a
fourth variable was invisible to both. Evidence produced by the thing being
checked is not evidence.

So the inventory is now derived from the syntax tree: *every* environment
access in `src/`, whatever it reads. A new variable nobody has thought of is in
scope automatically, which is the only version of this check that can catch the
next one.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: The one module permitted to read the environment for an operational
#: identity. Everything else takes the resolved `DeploymentContext`.
THE_RESOLVER = "src/deploy/context.py"

#: Reads that are not operational identities: process-level or vendor
#: credentials that name no deployment fact any other component must agree
#: about. Each is a decision, stated with its reason.
#:
#: The distinction that matters is *agreement*. Two components can hold
#: different opinions about `ALPACA_API_KEY` without a silent divergence,
#: because only one of them uses it. Two components holding different opinions
#: about which database this is produced fourteen days of work.
DECLARED_NON_IDENTITIES = {
    "src/reproducibility.py":
        "reads PYTHONHASHSEED to report whether the interpreter was started "
        "hash-randomised. A process fact, not a deployment choice — it "
        "describes the interpreter and nothing else consults it.",
    "src/execution.py":
        "broker credentials for the paper-trading adapter. Not on the pilot "
        "path; no other component forms a view about them.",
    "src/sentiment.py":
        "a vendor API key for an optional signal. Same reasoning as above.",
    "src/discovery/readers_quantify.py":
        "a provider API key for the Discovery reader, named by the reader "
        "rather than fixed in code so a challenger provider is a parameter "
        "and not a rewrite. A credential, not a deployment identity: nothing "
        "else forms a view about it, and which reader ran is recorded on the "
        "evidence as `produced_by` rather than inferred from the environment. "
        "The deployment choice that *is* an identity — whether a hosted "
        "reader may run at all — belongs in the context and is not read here.",
    "src/market_data/loader.py":
        "expands `${VAR}` references written in a snapshot manifest, so a "
        "bucket name need not be committed. The name comes from data rather "
        "than from code and the manifest is its only consumer, so no second "
        "component can hold a different view of it. `RESERVED_NAMES` refuses "
        "the identities the context owns, so it cannot become a back door.",
}


class EnvironmentRead:
    """One place in `src/` that reaches for the environment."""

    def __init__(self, module: str, line: int, names: frozenset) -> None:
        self.module, self.line, self.names = module, line, names

    def __repr__(self) -> str:                               # pragma: no cover
        named = ", ".join(sorted(self.names)) or "<indirect>"
        return f"{self.module}:{self.line} ({named})"


class _Reads(ast.NodeVisitor):
    """Every environment access, by shape rather than by variable name.

    Detects `os.environ`, `os.getenv(...)` and a bare `getenv(...)`. Names read
    are collected where they are string literals, for the failure message — but
    the *finding* does not depend on recognising the name, which is precisely
    what the previous version got wrong.
    """

    def __init__(self, module: str) -> None:
        self.module = module
        self.found: list = []

    def _record(self, node, names=frozenset()) -> None:
        self.found.append(EnvironmentRead(self.module, node.lineno, names))

    @staticmethod
    def _literals(node) -> frozenset:
        return frozenset(
            child.value for child in ast.walk(node)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
            and child.value.isupper() and "_" in child.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "environ":
            self._record(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = node.func
        name = (target.attr if isinstance(target, ast.Attribute)
                else getattr(target, "id", ""))
        if name == "getenv":
            self._record(node, self._literals(node))
        self.generic_visit(node)


def environment_reads():
    """Every environment access under `src/`, from the syntax tree."""
    found = []
    for path in sorted(Path("src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:                                  # pragma: no cover
            continue
        visitor = _Reads(str(path))
        visitor.visit(tree)
        # `os.environ.get("X")` produces both an Attribute and no Call match;
        # attach the literals from the enclosing statement so the message names
        # the variable even though the finding did not depend on it.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for read in visitor.found:
                if read.line == node.lineno and not read.names:
                    read.names = _Reads._literals(node)
        found.extend(visitor.found)
    return found


def readers():
    return {read.module for read in environment_reads()}


class TestOnlyOneModuleReadsTheEnvironment:
    def test_no_undeclared_reader(self):
        undeclared = readers() - {THE_RESOLVER} - set(DECLARED_NON_IDENTITIES)
        assert undeclared == set(), (
            f"these read the environment directly: {sorted(undeclared)}. Take "
            "the resolved DeploymentContext instead, or declare the read as a "
            "non-identity with its reason")

    def test_the_resolver_is_among_them(self):
        """A scan finding nothing would pass every check above."""
        assert THE_RESOLVER in readers(), (
            "the resolver does not appear to read the environment; the scan "
            "is broken and every assertion here is vacuous")

    def test_each_declaration_records_why(self):
        for module, reason in DECLARED_NON_IDENTITIES.items():
            assert Path(module).exists(), f"{module} no longer exists"
            assert len(reason.strip()) > 40, module

    def test_no_declared_non_identity_is_an_operational_one(self):
        """A declaration is a decision about scope, not an escape hatch.

        Anything a second component could form its own view about belongs in
        the context, whatever the reason field says.
        """
        operational = {"QUANTIFY_DATABASE_URL", "PILOT_DATA_POLICY",
                       "QUANTIFY_DEPLOYMENT_PROFILE", "ANTHROPIC_API_KEY",
                       "QUANTIFY_PARSER_MODEL", "QUANTIFY_MARKET_DATA_CACHE"}
        for read in environment_reads():
            if read.module in DECLARED_NON_IDENTITIES:
                assert not (read.names & operational), (
                    f"{read} is declared a non-identity but names an "
                    "operational one")


class TestTheSurfacesTakeTheContext:
    """The specific readers this slice removed, named so they stay removed."""

    @pytest.mark.parametrize("module", [
        "src/workspace/routes.py",           # ANTHROPIC_API_KEY, PARSER_MODEL
        "src/web/routes.py",
        "src/workspace/store.py",
        "src/db/engine.py",                  # QUANTIFY_DATABASE_URL
        "src/market_data/access.py",         # PILOT_DATA_POLICY
        "src/market_data/pilot_policy.py",
        "src/deploy/preflight.py",           # QUANTIFY_DEPLOYMENT_PROFILE
        "src/deploy/manifest.py",
        "src/api.py",
    ])
    def test_it_does_not_read_the_environment(self, module):
        assert module not in readers(), (
            f"{module} resolves a deployment fact for itself; it must take "
            "what `deploy.context.resolve` returned")


class TestThePreflightJudgesTheServedObject:
    """A preflight that resolved its own would be back to two answers."""

    def test_it_accepts_a_context(self):
        import inspect

        from src.deploy import preflight

        assert "context" in inspect.signature(preflight.run).parameters

    def test_it_judges_the_object_it_is_given(self):
        """Handed a context, the preflight must not go looking for another."""
        from src.deploy import context as context_module
        from src.deploy.preflight import Profile, run

        given = context_module.resolve({
            "QUANTIFY_DEPLOYMENT_PROFILE": "local",
            "QUANTIFY_DATABASE_URL": "sqlite:///given.db"})

        calls = []
        original = context_module.resolve
        try:
            context_module.resolve = lambda *a, **k: calls.append(1) or original(*a, **k)
            outcome = run(context=given)
        finally:
            context_module.resolve = original

        assert calls == [], "the preflight resolved a second context of its own"
        assert outcome.profile is Profile.LOCAL
        assert outcome.facts["database"]["engine"] == "sqlite"

    def test_create_app_serves_under_what_it_judged(self):
        """The object bound is the object preflighted, by identity."""
        import src.api as api
        from src.deploy import context as context_module

        judged = []
        original_run = api.__dict__.get("_run_probe")
        from src.deploy import preflight as preflight_module

        real_run = preflight_module.run

        def watched(*args, **kwargs):
            judged.append(kwargs.get("context"))
            return real_run(*args, **kwargs)

        preflight_module.run = watched
        try:
            api.create_app()
            served = context_module.bound()
        finally:
            preflight_module.run = real_run
            context_module.unbind()

        assert judged, "create_app did not run a preflight"
        assert judged[0] is not None, "the preflight was not given a context"
        assert served is judged[0], (
            "the application serves under a different object from the one the "
            "preflight judged")


class TestTheSecretsStayInTheContext:
    def test_the_model_key_is_not_a_public_field(self):
        from src.deploy.context import ModelTarget

        target = ModelTarget(_api_key="sk-not-a-real-key", model="m")
        assert "sk-not-a-real-key" not in repr(target)
        assert "sk-not-a-real-key" not in str(target.to_json())
        assert target.available is True

    def test_the_database_identity_is_loggable(self):
        from src.deploy.context import DatabaseTarget

        target = DatabaseTarget(
            url="postgresql://quantify:hunter2@db.internal:5432/quantify",
            dialect="postgresql")
        assert "hunter2" not in target.display
        assert "hunter2" not in str(target.to_json())

    def test_the_whole_context_is_loggable(self):
        from src.deploy.context import resolve

        context = resolve({
            "QUANTIFY_DATABASE_URL":
                "postgresql://quantify:hunter2@db.internal:5432/quantify",
            "ANTHROPIC_API_KEY": "sk-not-a-real-key",
            "PILOT_DATA_POLICY": "SYNTHETIC_ONLY"})
        rendered = str(context.to_json())
        assert "hunter2" not in rendered
        assert "sk-not-a-real-key" not in rendered
        # And still says enough to diagnose a deployment.
        assert "postgresql" in rendered
        assert "SYNTHETIC_ONLY" in rendered


class TestTheContextHoldsNothingRequestScoped:
    """A context that accepted an owner would become the next control plane,
    and its immutability would stop meaning anything."""

    def test_it_carries_no_request_fields(self):
        import dataclasses

        from src.deploy.context import DeploymentContext

        fields = {f.name for f in dataclasses.fields(DeploymentContext)}
        for forbidden in ("owner", "tenant", "request_id", "trace_id",
                          "user", "budget", "session"):
            assert forbidden not in fields, forbidden

    def test_it_is_frozen(self):
        import dataclasses

        from src.deploy.context import resolve

        context = resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"})
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.profile = "anything"                        # type: ignore


class TestResolutionDoesNotEscapeItsMapping:
    """An injected mapping is the whole environment, or the caller is being
    told about a deployment it did not describe."""

    def test_an_absent_url_does_not_fall_through_to_the_process(self,
                                                                monkeypatch):
        from src.deploy.context import resolve

        monkeypatch.setenv("QUANTIFY_DATABASE_URL",
                           "postgresql://real:real@real/real")
        resolved = resolve({"PILOT_DATA_POLICY": "SYNTHETIC_ONLY"})
        assert "real" not in resolved.database.url, (
            "resolution fell back to os.environ for a mapping that did not "
            "name the variable")
        assert resolved.database.configured is False

    def test_an_absent_policy_does_not_fall_through(self, monkeypatch):
        from src.deploy.context import resolve

        monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
        assert resolve({}).market_data.policy is None


class TestAManifestCannotReachADeploymentIdentity:
    """The declared non-identity read, fenced rather than merely explained."""

    @pytest.mark.parametrize("name", sorted([
        "QUANTIFY_DATABASE_URL", "PILOT_DATA_POLICY",
        "QUANTIFY_DEPLOYMENT_PROFILE", "ANTHROPIC_API_KEY"]))
    def test_a_reserved_reference_is_refused(self, name, monkeypatch):
        from src.market_data.loader import ReservedReference, _resolve

        monkeypatch.setenv(name, "a-value-a-manifest-should-not-reach")
        with pytest.raises(ReservedReference):
            _resolve("${%s}" % name)

    def test_an_ordinary_reference_still_expands(self, monkeypatch):
        from src.market_data.loader import _resolve

        monkeypatch.setenv("QUANTIFY_SNAPSHOT_BUCKET", "a-bucket")
        assert _resolve("${QUANTIFY_SNAPSHOT_BUCKET}") == "a-bucket"

    def test_the_reserved_list_covers_what_the_context_resolves(self):
        """Derived from the context, so a new identity is fenced by default."""
        import inspect

        from src.deploy import context as context_module
        from src.market_data.loader import RESERVED_NAMES

        source = inspect.getsource(context_module.resolve)
        named = {word.strip('"\'') for word in source.split()
                 if word.strip('"\',()').isupper() and "_" in word}
        resolved = {one.strip('"\',()') for one in named
                    if one.strip('"\',()').startswith(("QUANTIFY_", "PILOT_",
                                                       "ANTHROPIC_"))}
        missing = resolved - RESERVED_NAMES
        assert missing == set(), (
            f"the context resolves {sorted(missing)} but a manifest may still "
            "expand it")


class TestResolutionTerminates:
    """`resolve_target(None)` asks `current()`, which resolves. If the resolver
    ever calls it with no argument the pair recurses until the stack ends —
    and it would do so only in a deployment nobody configured, which is the
    one place a crash is least diagnosable."""

    def test_resolving_an_unconfigured_deployment_terminates(self):
        from src.deploy.context import resolve

        assert resolve({}).database.url.startswith("sqlite")

    def test_the_engine_default_reaches_the_context(self, monkeypatch):
        from src.db.engine import resolve_target
        from src.deploy.context import bind, resolve, unbind

        bind(resolve({"QUANTIFY_DATABASE_URL": "sqlite:///bound-here.db"}))
        try:
            assert resolve_target() == "sqlite:///bound-here.db"
        finally:
            unbind()
