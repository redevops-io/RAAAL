"""Every path the workspace routers declare is behind the session gate.

`PRIVATE_PREFIXES` decides what requires a signed-in viewer, and it was a
hand-written tuple describing itself as "the whole private surface". It was not.
Two routers serve the workspace: `workspace.routes` carries `prefix="/workspace"`
and was covered, and `workspace.pilot_routes` carries no prefix at all, so
`/pilot`, `/pilot/answer`, `/pilot/save` and `/pilot/plans/{plan_id}` sat at the
root, outside the gate, on a live site that already held plans.

`/pilot/plans/{plan_id}` is why it mattered rather than being untidy. It
resolves a stored plan for `owner.current()`, which without a session is the
shared `pilot` workspace — so a plan id was enough to read a plan saved there,
while the same plan under `/workspace/plans/{plan_id}` sent the reader to sign
in. One route module, mounted twice, gated once.

**So this derives the surface instead of restating it.** The paths come from the
routers themselves — the decorator arguments in the source, plus the router's
own prefix — because a second hand-written list would be the same artifact that
failed, checked against itself. Read with `ast`: a grep for `@router.get` would
match this docstring.

What is deliberately public is named individually, with a reason. A prefix
tuple that could be widened to make this pass would be no check at all.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"

#: The router modules that serve the workspace, and nothing else.
WORKSPACE_ROUTERS = ("workspace/routes.py", "workspace/pilot_routes.py")

#: Public on purpose, each for a stated reason rather than by omission.
#:
#: `/auth/*` is the login itself: requiring a session to start a session cannot
#: work. It lives in `auth_routes.py`, which is not in the list above, and is
#: named here only so the rule is legible without opening that file.
DELIBERATELY_PUBLIC = {
    "/auth": "a login that required a login could not be started",
    "/evaluate": "the canonical public evaluator; trying a strategy needs no "
                 "account, an account is the price of keeping a plan (§3 of the "
                 "public strategy-lab plan). It is PUBLIC_EVALUATION in the "
                 "boundary manifest and delegates to the pilot flow.",
}


def declared_paths(relative: str):
    """Every path a router module declares, with its router prefix applied.

    Both halves are needed and neither is guessable from the other: the
    decorator gives `/plans/{plan_id}` and the `APIRouter(prefix=...)` decides
    whether that is served at `/workspace/plans/{plan_id}` or at the root. The
    bug was entirely in the second half.
    """
    source = (SRC / relative).read_text()
    tree = ast.parse(source)

    prefix = ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", "") != "APIRouter":
            continue
        for keyword in node.keywords:
            if keyword.arg == "prefix" and isinstance(keyword.value, ast.Constant):
                prefix = str(keyword.value.value)

    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not isinstance(function, ast.Attribute):
            continue
        if function.attr not in ("get", "post", "put", "delete", "patch"):
            continue
        if getattr(function.value, "id", "") != "router":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        path = str(node.args[0].value)
        found.add((prefix + path) or "/")
    return prefix, found


class TestTheGateCoversWhatTheRoutersServe:
    def test_every_declared_path_is_private(self):
        from src.api import PRIVATE_PREFIXES

        uncovered = []
        for relative in WORKSPACE_ROUTERS:
            _prefix, paths = declared_paths(relative)
            for path in sorted(paths):
                if any(path.startswith(one) for one in PRIVATE_PREFIXES):
                    continue
                if any(path.startswith(one) for one in DELIBERATELY_PUBLIC):
                    continue
                uncovered.append(f"{relative}: {path}")

        assert uncovered == [], (
            "these workspace paths are served without requiring a session:\n  "
            + "\n  ".join(uncovered)
            + "\nA workspace route outside PRIVATE_PREFIXES is readable by "
              "anybody, and under the shared owner that means readable plans")

    def test_the_reader_actually_finds_paths(self):
        """Without this the check above passes by finding nothing at all.

        The parse is specific — `router.get("...")` with a literal first
        argument — and a refactor to `add_api_route` or a computed path would
        make it silently return an empty set, which is indistinguishable from
        a codebase with no exposure.
        """
        for relative in WORKSPACE_ROUTERS:
            _prefix, paths = declared_paths(relative)
            assert len(paths) >= 3, (
                f"{relative}: found {len(paths)} routes, so this check is not "
                "reading the router any more")

    def test_it_notices_the_defect_it_was_written_for(self):
        """The mutation: the surface as it was, and the gate as it was.

        `/pilot/plans/{plan_id}` under a `("/workspace",)` gate must be
        reported. A check that could not fail on the exact configuration that
        shipped is not evidence that the configuration changed.
        """
        _prefix, paths = declared_paths("workspace/pilot_routes.py")
        assert "/pilot/plans/{plan_id}" in paths, (
            "the pilot plan route is not where this expects it")
        assert not any(path.startswith("/workspace") for path in paths), (
            "pilot_routes now carries a /workspace prefix; if that is the fix, "
            "this test should be rewritten rather than deleted")


class TestThePilotRouterHasNoPrefixOfItsOwn:
    def test_its_paths_are_absolute(self):
        """Why the omission was easy to make, recorded where it happened.

        `routes.py` declares `/plans/{plan_id}` and serves it at
        `/workspace/plans/{plan_id}`; `pilot_routes.py` declares
        `/pilot/plans/{plan_id}` and serves it exactly there. Reading either
        file alone tells you nothing about whether the gate covers it.
        """
        prefix, _paths = declared_paths("workspace/pilot_routes.py")
        assert prefix == "", (
            "pilot_routes has gained a router prefix; the gate must be checked "
            "against the new one rather than against the paths in the file")

        workspace_prefix, _ = declared_paths("workspace/routes.py")
        assert workspace_prefix == "/workspace"


class TestEveryPublicExceptionIsExplained:
    def test_each_carries_a_reason(self):
        for prefix, reason in DELIBERATELY_PUBLIC.items():
            assert len(reason) > 20, f"{prefix} is public with no reason given"
