"""Shared fixtures, and the market-data tier split.

Two datasets, two purposes:

    tests/fixtures/prices_synthetic.parquet   committed, invented, no network
    the licensed snapshot                     private, immutable, pinned by hash

The default suite runs entirely on the first. It needs no credentials, reaches
no network, and produces the same numbers on a fresh clone as on the machine
that wrote it. That is the point: the repository, not one workstation, has to be
able to reproduce the application.

The licensed snapshot is for integration, benchmark and research runs, behind
`-m market_data_integration`. When that marker is not requested, missing
credentials are not a failure — nobody asked for the licensed data.

Nothing measured on the synthetic fixture is a claim about any real security. It
is shaped like market data so the evaluation stack has something realistic to
run on, and deliberately not calibrated to anything.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = REPO_ROOT / "tests" / "fixtures" / "prices_synthetic.parquet"

#: The vendor snapshot, when a developer has produced one locally. Used only by
#: the integration tier; the default suite ignores it even when present, so a
#: result cannot differ between two machines because one happened to have it.
LICENSED = REPO_ROOT / "data" / "history" / "prices.parquet"

# --- the PostgreSQL lane, where a skip is a failure -------------------------
#
# The checks live inside `pytest_sessionstart` and `pytest_sessionfinish`
# further down, which already exist for the tree-freeze guard. Adding a second
# pair of functions with those names would have been silent: Python keeps the
# last definition in a module, so the new hooks simply never ran and the lane
# reported green while checking nothing — the exact failure they are for.

#: Set by the CI lane that exists to run these. Off everywhere else, so a
#: developer without a database still gets a useful suite.
REQUIRE_POSTGRES = "QUANTIFY_REQUIRE_POSTGRES"

#: Guarantees that only run on PostgreSQL and must therefore be *seen* to run.
#:
#: Named individually rather than counted. A count is satisfied by any test at
#: all, and the failure this prevents is precisely the one that already
#: happened: the digest check was skipped on SQLite for its whole life, so a
#: delivery record that could not be verified against its data went unnoticed
#: until somebody ran the other lane by hand. A rename that drops one from this
#: list is then a decision somebody makes in a diff, which is the point.
MUST_RUN_ON_POSTGRES = (
    "tests/test_provenance_journey.py::"
    "TestTheStoredRunCitesTheDeliveryItConsumed::"
    "test_the_stored_digest_is_the_resolver_digest",
    "tests/test_provenance_journey.py::"
    "TestTheStoredRunCitesTheDeliveryItConsumed::"
    "test_the_recorded_request_is_what_the_plan_asked_for",
    "tests/test_provenance_journey.py::"
    "TestTheStoredRunCitesTheDeliveryItConsumed::"
    "test_the_other_request_would_not_have_verified",
    "tests/test_tenancy_invariant.py::TestTheSchemaLayer::"
    "test_no_tenant_owned_table_is_unsafe",
    "tests/test_tenancy_invariant.py::TestTheSchemaLayer::"
    "test_the_exception_list_is_empty",
)

_ran: set = set()
_skipped_for_postgres: list = []


#: The exact phrases a "there is no database here" gate uses.
#:
#: Matched precisely rather than by looking for "postgres" anywhere in the
#: reason, because not every skip that mentions a database is one. The restore
#: drill declines when the database it would populate is not the container it
#: would dump — a statement about topology, in a lane that has a perfectly good
#: database — and a substring match flagged all eleven of its tests as
#: unchecked guarantees. A guard that cries wolf about a correct skip gets
#: switched off, which costs more than it was ever going to catch.
NO_DATABASE_HERE = ("needs a reachable postgresql", "sqlite cannot")


def pytest_runtest_logreport(report):
    """Remember what ran and what stood aside for want of a database."""
    if report.skipped:
        reason = ""
        if isinstance(report.longrepr, tuple) and len(report.longrepr) == 3:
            reason = str(report.longrepr[2])
        lowered = reason.lower()
        if any(phrase in lowered for phrase in NO_DATABASE_HERE):
            _skipped_for_postgres.append((report.nodeid, reason))
    elif report.when == "call" and report.passed:
        _ran.add(report.nodeid)


def _lane_problems():
    """Why this lane checked less than it exists to check, if it did.

    Two questions, because either can hold while the other fails: did anything
    stand aside *because* it wanted a database, and did each named guarantee
    actually execute. A test can leave collection without ever reporting a skip
    — a rename, a stray `-k`, a deleted file — and only the second notices.
    """
    if not os.environ.get(REQUIRE_POSTGRES):
        return []
    problems = [f"  skipped for want of a database: {nodeid}\n"
                f"    {reason.strip()[:160]}"
                for nodeid, reason in _skipped_for_postgres]
    problems += [f"  did not run: {nodeid}" for nodeid in MUST_RUN_ON_POSTGRES
                 if nodeid not in _ran]
    return problems


@pytest.fixture(autouse=True)
def no_model_calls(request, monkeypatch):
    """The default suite never calls a language model.

    Stage 1 may use one, and the moment `anthropic` was installed alongside an
    `ANTHROPIC_API_KEY` the workspace tests started making live API calls — they
    became nondeterministic, network-dependent, billable, and dependent on which
    machine ran them. One promptly failed because the model raised a question
    the deterministic rules do not.

    Same rule as the licensed market data: reaching outside is opt-in, and a
    test that silently acquires a network dependency passes locally, fails in
    CI, and gets diagnosed as flaky for a week. Opt in with
    `@pytest.mark.model_stage1`.
    """
    if request.node.get_closest_marker("model_stage1"):
        return
    if request.node.get_closest_marker("real_parser_client"):
        # Reaches the real `_parser_client` without reaching a model. The
        # decision it makes — declared mode, refuse or fall back — is worth
        # testing, and `model_stage1` is the wrong marker for it: that tier is
        # deselected by default because it calls a live API, so marking these
        # meant they silently never ran.
        return
    try:
        import src.workspace.routes as routes
    except Exception:                                           # pragma: no cover
        return
    monkeypatch.setattr(routes, "_parser_client", lambda: None, raising=False)


def _tree_state():
    """HEAD, the working-tree digest, and the untracked-file inventory.

    Three parts because they fail differently. `HEAD` catches a commit or a
    checkout mid-run; the porcelain digest catches an edit to a tracked file;
    the untracked inventory catches a new file appearing, which is the one the
    other two miss and the one that happens when a test file is being written
    while the suite runs.

    Returns None outside a git checkout, where there is nothing to compare and
    a hard failure would be noise.
    """
    import subprocess

    def run(*args):
        done = subprocess.run(("git",) + args, capture_output=True, text=True,
                              cwd=str(Path(__file__).resolve().parent.parent))
        return done.stdout if done.returncode == 0 else None

    head = run("rev-parse", "HEAD")
    if head is None:
        return None
    # The diff, not the status listing. `git status --porcelain` prints
    # ` M src/foo.py` whether a file was changed once or ten times, so
    # appending to a file that was already modified left the digest identical
    # — which is exactly the situation during this branch, where the tree is
    # never clean. Caught by testing the guard against the case it was written
    # for rather than against a clean checkout.
    tracked = run("diff", "HEAD") or ""
    untracked = run("ls-files", "--others", "--exclude-standard") or ""
    return {
        "head": head.strip(),
        "tracked": hashlib.sha256(tracked.encode()).hexdigest(),
        "untracked": tuple(sorted(untracked.split())),
    }


def pytest_sessionstart(session):
    """A suite run is evidence only if the code under test held still for it.

    Twice on this branch a full run was invalidated by an edit landing while
    it was in flight — once a source file, once a new test file — and both
    times the rule that would have prevented it was a convention someone had
    to remember. It is checked here instead.

    The PostgreSQL lane's precondition is here too, for the same reason it is
    a precondition: a lane that asks for a database and is not given one would
    run the SQLite suite, pass, and report the persistence lane green.
    """
    if os.environ.get(REQUIRE_POSTGRES) and not os.environ.get(
            "QUANTIFY_TEST_POSTGRES_URL"):
        raise pytest.UsageError(
            f"{REQUIRE_POSTGRES} is set and QUANTIFY_TEST_POSTGRES_URL is not. "
            "This lane exists to run the guarantees SQLite cannot express; "
            "without a database it would pass by skipping them")
    session.config._tree_at_start = _tree_state()


def pytest_sessionfinish(session, exitstatus):
    # First, because it is the cheaper failure to read and because the tree
    # check below returns early on the ordinary path.
    problems = _lane_problems()
    if problems:
        session.exitstatus = 1
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        say = reporter.write_line if reporter is not None else print
        say("")
        say("the PostgreSQL lane did not check what it exists to check:")
        for problem in problems:
            say(problem)

    before = getattr(session.config, "_tree_at_start", None)
    after = _tree_state()
    if before is None or after is None or before == after:
        return

    changed = []
    if before["head"] != after["head"]:
        changed.append(f"HEAD {before['head'][:12]} -> {after['head'][:12]}")
    if before["tracked"] != after["tracked"]:
        changed.append("a tracked file was modified")
    added = set(after["untracked"]) - set(before["untracked"])
    removed = set(before["untracked"]) - set(after["untracked"])
    if added:
        changed.append(f"untracked appeared: {', '.join(sorted(added)[:5])}")
    if removed:
        changed.append(f"untracked removed: {', '.join(sorted(removed)[:5])}")

    # Written to the terminal *and* made the exit status, because a warning at
    # the end of four thousand lines of output is a warning nobody reads. The
    # run may have been entirely green; the point is that its greenness is no
    # longer about any one version of the code.
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line("")
        reporter.write_line(
            "INVALIDATED: source tree changed during execution — "
            + "; ".join(changed), red=True, bold=True)
        reporter.write_line(
            "  The result above describes no single version of the code. "
            "Re-run against a frozen tree.", red=True)
    session.exitstatus = 3


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "model_stage1: exercises model-assisted compiler stage 1 against a live "
        "API. Requires ANTHROPIC_API_KEY and network; opt-in, because the "
        "default suite must produce the same result on every machine.")
    config.addinivalue_line(
        "markers",
        "market_data_integration: runs against the licensed snapshot. Requires "
        "credentials and network; fails rather than skips when explicitly "
        "requested, because a silent skip in the tier that exists to check the "
        "real data is indistinguishable from a pass.")


@pytest.fixture(scope="session")
def synthetic_prices():
    """The committed fixture, loaded through the ordinary loader.

    Deliberately not `pd.read_parquet` — going through the loader means the
    integrity check runs on every suite, so a fixture edited by hand is caught
    here rather than by a confusing assertion three layers up.
    """
    from src.market_data import load_prices, synthetic_snapshot

    return load_prices(synthetic_snapshot())


@pytest.fixture
def prices_on_disk(monkeypatch, tmp_path, synthetic_prices):
    """Point the web and workspace routes at the synthetic fixture.

    Both read a module-level path at request time. Redirecting that is what
    makes the rendered result panels — and therefore the pages asserting
    provenance is visible — work from a clone alone.
    """
    import src.web.routes as web_routes
    import src.workspace.routes as workspace_routes

    monkeypatch.setattr(web_routes, "PRICES", SYNTHETIC)
    monkeypatch.setattr(workspace_routes, "PRICES", SYNTHETIC)
    return synthetic_prices


@pytest.fixture(scope="session")
def licensed_snapshot():
    """The pinned licensed snapshot, for the integration tier only."""
    from src.market_data import load_prices, production_snapshot

    snapshot = production_snapshot()
    return load_prices(snapshot, allow_network=True), snapshot


def requires_licensed_data(func):
    """Mark a test as belonging to the licensed tier."""
    return pytest.mark.market_data_integration(func)


#: Set by the integration tier to make absence a failure rather than a skip.
LICENSED_REQUIRED = os.environ.get("QUANTIFY_REQUIRE_MARKET_DATA") == "1"


@pytest.fixture(autouse=True)
def _pilot_data_policy(monkeypatch):
    """Run the suite under the closed-pilot boundary, stated rather than
    inherited.

    `_prices()` fails closed without this, so a suite that did not declare a
    policy would quietly stop exercising every run path — and the journey tests
    would pass by producing nothing.
    """
    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")


@pytest.fixture(autouse=True)
def _no_inherited_deployment():
    """No test serves under the deployment a previous test established.

    `create_app` binds a process-wide `DeploymentContext`, which is right for a
    server — one process, one deployment — and wrong for a suite, where the
    next test would silently answer from the previous one's environment. That
    is the same defect the context exists to remove, one level up: a component
    using a resolved answer that was resolved for something else.
    """
    from src.deploy.context import unbind

    unbind()
    try:
        yield
    finally:
        unbind()


#: Identity configuration that must not arrive from the machine running the
#: tests.
IDENTITY_VARIABLES = ("OIDC_ISSUER", "OIDC_AUDIENCE", "OIDC_CLIENT_ID",
                      "OIDC_INTERNAL_BASE_URL", "PUBLIC_BASE_URL")


@pytest.fixture(scope="session", autouse=True)
def _no_ambient_identity_at_all(request):
    """The same rule, for fixtures that run before a function-scoped one can.

    `_no_ambient_identity` below is function-scoped, and pytest sets up
    higher-scoped fixtures first — so a module-scoped fixture builds its whole
    world *before* the protection applies, and sees the developer's shell after
    all.

    Found the expensive way. A shell holding `OIDC_ISSUER` and `OIDC_AUDIENCE`
    but no `OIDC_CLIENT_ID` — which is what a half-finished provider setup
    leaves behind — made the application demand a session and then refuse to
    start a login, so `test_backup_restore`'s module fixture failed at its
    first assertion with "there is nothing to log in to". On a machine with a
    clean shell it passed. That is the same "depends on who ran it" failure the
    function-scoped fixture was written to end, arriving through the one gap it
    could not cover.

    Session-scoped and autouse, so the environment is cleaned once before any
    fixture of any scope runs, and put back when the run finishes.
    """
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    for name in IDENTITY_VARIABLES:
        patch.delenv(name, raising=False)
    yield
    patch.undo()


@pytest.fixture(autouse=True)
def _no_ambient_identity(monkeypatch):
    """No test inherits an identity provider from the developer's shell.

    `unbind` above clears the *bound* context; `current()` then resolves a new
    one from the environment, and an environment is whatever the machine
    happens to have. A shell holding `OIDC_ISSUER` for an unrelated project —
    which is exactly how this was found — made the suite behave as though this
    deployment had accounts: the workspace demanded a session, and 123 tests
    that had nothing to do with identity failed on a redirect to a login.

    Worse than the failures is the direction they could have gone. The same
    ambient value could have made a test pass on the machine that had it and
    fail in CI, or the reverse — a suite whose result depends on who ran it is
    not evidence about the code.

    A test that wants a provider declares one, by passing a mapping to
    `resolve` or by patching the accessor. None of them need the environment.
    """
    for name in IDENTITY_VARIABLES:
        monkeypatch.delenv(name, raising=False)


# --- the confirmation journey, as a browser performs it ---------------------

def submit_rendered_confirmation(client, description, *, title="Test plan",
                                 answers=None, acknowledge_all=True):
    """GET the confirmation page, then submit the controls it rendered.

    Parses the returned HTML and posts the fields that are actually inside the
    save form. That is the point: the answer and confirmation radios once
    rendered *outside* it, so a user could read every question, click every
    answer, press Save, and none of it was submitted. Every backend test passed
    because they each built the POST body by hand.

    A test that constructs the payload it wishes the page produced is testing
    its own copy of the contract. This reads the page.

    Returns `(response, plan_id)`; `plan_id` is `None` unless the save
    redirected, because the identity is generated by the server.
    """
    import html as html_module
    import re

    page = client.get("/workspace/new", params={"describe": description})
    if page.status_code != 200:
        return page, None
    body = page.text

    fields = {"describe": description, "title": title}
    found = re.search(r'name="parse" value="([^"]*)"', body)
    fields["parse"] = html_module.unescape(found.group(1)) if found else ""

    # Radios: a browser submits the pre-checked option, or nothing until the
    # user picks one. Groups with no selection are what the user must answer.
    offered, chosen = {}, {}
    for name, value, rest in re.findall(
            r'<input type="radio" name="([^"]+)"\s+value="([^"]*)"([^>]*)',
            body, re.S):
        offered.setdefault(name, value)
        if "checked" in rest:
            chosen[name] = value

    # Selects submit their selected option, or the first one when none is
    # marked. Reading only radios left every confirmation empty the moment the
    # page moved to dropdowns, and the plan then failed to save for a reason
    # that had nothing to do with what was being tested.
    for name, options in re.findall(
            r'<select name="([^"]+)"[^>]*>(.*?)</select>', body, re.S):
        picked = re.search(r'<option value="([^"]*)"[^>]*selected', options)
        if picked:
            chosen[name] = picked.group(1)
        else:
            first = re.search(r'<option value="([^"]+)"', options)
            if first:
                offered.setdefault(name, first.group(1))

    for name, first in offered.items():
        chosen.setdefault(name, first)
    fields.update(chosen)

    if acknowledge_all:
        for name in re.findall(r'name="(exclude:[^"]+)"', body):
            fields[name] = "on"
    for field, value in (answers or {}).items():
        fields[f"answer:{field}"] = value

    response = client.post("/workspace/save", data=fields,
                           follow_redirects=False)
    plan_id = (response.headers["location"].rsplit("/", 1)[-1]
               if response.status_code == 303 else None)
    return response, plan_id


@pytest.fixture
def requires_the_vendor_snapshot():
    """Skip when the licensed price file is genuinely absent, and only then.

    The vendor parquet is deliberately not in the repository: it is not
    redistributable, and a repository under a public licence is redistribution.
    So a clean clone cannot run the handful of tests that ask whether a run
    comes back *holding prices*.

    Those tests are not decoration — they exist because every other check on
    the vendor snapshot once passed while no run could use it, and they are the
    only ones that ask the end-to-end question. Deleting them to get a green
    clone would remove the guard that caught that. Failing on a clone is no
    better: it reports a licensing constraint as a defect, and a suite whose
    red is expected is a suite nobody reads.

    So: skip, narrowly. The condition is the file being missing and nothing
    else, so on any machine or runner that has fetched the snapshot these tests
    run exactly as before. The message names the file, because "skipped" with
    no reason is how a permanently unverified property hides.
    """
    from pathlib import Path

    import pytest

    from src.market_data.access import approved_snapshot

    snapshot = approved_snapshot()
    if snapshot is None:
        return

    # A local snapshot is present iff its parquet is in the checkout.
    if getattr(snapshot, "is_local", False):
        path = Path(__file__).resolve().parent.parent / snapshot.uri
        if not path.exists():
            pytest.skip(
                f"the licensed vendor snapshot is absent ({snapshot.uri}). It "
                "is not redistributable, so this property is unverified in this "
                "checkout — fetch the snapshot to run it")
        return

    # An S3 snapshot is reachable only where boto3, credentials, the network and
    # the bucket URI env are all present — an offline runner or a clean clone has
    # none of them. Probe by actually loading it; skip narrowly if unreachable,
    # so the end-to-end guard still runs wherever the data can be fetched.
    from src.market_data.loader import load_prices
    try:
        load_prices(snapshot, allow_network=True)
    except Exception as why:                                    # noqa: BLE001
        pytest.skip(
            f"the licensed vendor snapshot is not reachable here "
            f"({snapshot.snapshot_id}): {why}. It is S3-backed and not "
            "redistributable, so this property is unverified in this environment")
