"""One account must not see another account's plans.

Every plan on this deployment belonged to the literal string `"pilot"`. That
was honest while one shared basic-auth password guarded the door — everybody
inside had been let in deliberately.

It stopped being honest the moment registration opened. The session gate asked
*whether* somebody was signed in and never *who they are*, so any stranger who
registered could read the plans of everybody who had. The two halves shipped
hours apart and the gap between them was the exposure.

These tests are the ones that would have caught it: two subjects, and neither
can see the other. The interesting assertion is the negative, as it is for
every boundary in this project.
"""
from __future__ import annotations

import os

import pytest

from src.deploy.identity import Identity
from src.workspace import owner as owner_module


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from src.deploy import context as deploy_context

    monkeypatch.setenv("PILOT_DATA_POLICY", "SYNTHETIC_ONLY")
    monkeypatch.setenv("QUANTIFY_PILOT_READER", "recorded")
    monkeypatch.setenv("QUANTIFY_PARSER_MODE", "RUNTIME")
    monkeypatch.setenv("QUANTIFY_PARSER_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    monkeypatch.setenv("QUANTIFY_DATABASE_URL", f"sqlite:///{tmp_path}/w.db")

    resolved = deploy_context.resolve(dict(os.environ))
    monkeypatch.setattr(deploy_context, "current", lambda: resolved)

    from src.workspace import pilot_store

    return pilot_store


def signed_in_as(subject):
    """Establish a viewer the way the middleware does."""
    from src.workspace.routes import _LOOKING

    return _LOOKING.set(Identity(subject=subject, email=f"{subject}@x.test"))


def store_a_plan(store, plan_id, text):
    connection = store._connect()
    try:
        connection.execute(
            "INSERT INTO pilot_plans "
            "(plan_id, owner, created_at, text, artifact) "
            "VALUES (?, ?, ?, ?, ?)",
            (plan_id, store.PILOT_OWNER(), "2026-08-14T00:00:00Z", text,
             '{"text": "' + text + '"}'))
        connection.commit()
    finally:
        connection.close()


class TestTwoAccountsAreTwoWorkspaces:
    def test_one_cannot_read_the_other_s_plan(self, workspace):
        from src.workspace.routes import _LOOKING

        token = signed_in_as("user-a")
        try:
            store_a_plan(workspace, "plan-1", "a plan belonging to A")
            assert workspace.load("plan-1") is not None
        finally:
            _LOOKING.reset(token)

        token = signed_in_as("user-b")
        try:
            assert workspace.load("plan-1") is None, (
                "a second account read the first account's plan by its id — "
                "which is what every registered account could do")
        finally:
            _LOOKING.reset(token)

    def test_the_owner_written_is_the_verified_subject(self, workspace):
        from src.workspace.routes import _LOOKING

        token = signed_in_as("user-a")
        try:
            store_a_plan(workspace, "plan-2", "another plan")
            connection = workspace._connect()
            try:
                row = connection.execute(
                    "SELECT owner FROM pilot_plans WHERE plan_id = ?",
                    ("plan-2",)).fetchone()
            finally:
                connection.close()
        finally:
            _LOOKING.reset(token)

        assert row["owner"] == "user-a", (
            f"the plan was written under {row['owner']!r} rather than the "
            "subject of the session that created it")

    def test_two_accounts_may_hold_the_same_plan_id(self, workspace):
        """The key is (owner, plan_id). A plan id derived from the same
        sentence collides across accounts, and without the owner in the key one
        would overwrite the other."""
        from src.workspace.routes import _LOOKING

        for subject, text in (("user-a", "A's sentence"), ("user-b", "B's")):
            token = signed_in_as(subject)
            try:
                store_a_plan(workspace, "same-id", text)
            finally:
                _LOOKING.reset(token)

        for subject, expected in (("user-a", "A's sentence"), ("user-b", "B's")):
            token = signed_in_as(subject)
            try:
                assert workspace.load("same-id")["text"] == expected
            finally:
                _LOOKING.reset(token)


class TestADeploymentWithNoProviderIsUnchanged:
    """The configuration the pilot ran under for months, and still may.

    One shared credential and one workspace is a legitimate deployment —
    nobody is pretending otherwise. Returning a subject where there is no
    identity would invent a tenant.
    """

    def test_the_owner_is_the_shared_workspace(self, workspace):
        from src.workspace.routes import _LOOKING

        token = _LOOKING.set(None)
        try:
            assert owner_module.current() == owner_module.SHARED
            assert workspace.PILOT_OWNER() == "pilot"
        finally:
            _LOOKING.reset(token)

    def test_rows_already_written_still_belong_to_it(self, workspace):
        """Existing plans were written under `"pilot"`. They are not orphaned
        by this change; they belong to the shared workspace, which is what
        they always did."""
        from src.workspace.routes import _LOOKING

        token = _LOOKING.set(None)
        try:
            store_a_plan(workspace, "old-plan", "written before accounts")
            assert workspace.load("old-plan") is not None
        finally:
            _LOOKING.reset(token)


class TestTheOwnerCannotBeAsserted:
    def test_it_is_read_from_the_session_and_nowhere_else(self):
        """An owner a request can state is an owner a request can choose.

        Checked structurally: `owner.current` reads the context variable the
        middleware sets from a verified token, and takes no argument a handler
        could pass.
        """
        import inspect

        assert not inspect.signature(owner_module.current).parameters, (
            "owner.current takes an argument, so a caller can name a "
            "workspace rather than being told which one they are in")

        # The code, not the docstring. This function's own prose says it
        # reads "never from a form field or a query parameter", so a grep over
        # the source matched the sentence explaining the rule — the third time
        # today a structural check has been failed by its own explanation.
        import ast

        tree = ast.parse(inspect.getsource(owner_module.current).lstrip())
        body = tree.body[0].body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)):
            body = body[1:]                                # drop the docstring
        code = "\n".join(ast.unparse(node) for node in body)

        assert "_LOOKING" in code, (
            "the owner is not read from the verified session")
        for controlled in ("request", "form", "query_params", "cookies",
                           "headers"):
            assert controlled not in code, (
                f"the owner is derived from {controlled!r}, which the request "
                "controls, so a caller could name somebody else's workspace")
