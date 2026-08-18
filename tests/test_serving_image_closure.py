"""What the serving image installs is what the serving path can reach.

`requirements-core.txt` is what the production image installs, and it carried
`agentic-os@v0.2.3`, which pins `runtime-contracts@v0.2.2` while Quantify pins
`v0.2.4`. Unsatisfiable — `ResolutionImpossible` — and the reason the image
could not be built at all. The conflict had existed since the contracts pin
moved and nothing had noticed, because nothing had built the image.

It was removed rather than loosened, because the serving API does not use it:
an import-graph walk from `src/api.py` reaches no module under `src/agentic/`.

This asserts that reachability, so the day the request path does reach the
mission runtime the version conflict has to be *resolved* rather than
rediscovered by a failed build months later.

**On the import graph, not on the installed environment.** This checkout has
everything installed; the image does not, and the question is about the image.
"""
from __future__ import annotations

import ast
import collections
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
CORE = SRC.parent / "requirements-core.txt"


def _graph() -> dict:
    """Every intra-`src` import edge, plus the third-party names each module
    mentions."""
    edges = collections.defaultdict(set)
    for path in SRC.rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        module = ".".join(path.relative_to(SRC).with_suffix("").parts)
        package = module.split(".")[:-1]
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[:len(package) - node.level + 1]
                    target = (".".join((*base, node.module)) if node.module
                              else ".".join(base))
                else:
                    target = node.module or ""
                    if target.startswith("src."):
                        target = target[4:]
                edges[module].add(target)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    edges[module].add(alias.name[4:]
                                      if alias.name.startswith("src.")
                                      else alias.name)
    return edges


def _reachable_from(entry: str) -> set:
    edges = _graph()
    seen, stack = set(), [entry]
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        stack.extend(t for t in edges.get(module, ()) if t in edges)
    return seen


def _third_party_on(closure: set) -> set:
    edges = _graph()
    return {t for module in closure for t in edges.get(module, ())
            if t and t not in edges and not t.startswith(".")}


def test_the_serving_path_reaches_no_mission_runtime():
    """The property that made removing it safe."""
    closure = _reachable_from("api")
    assert closure, "the import walk found nothing; it is not measuring"

    inside = sorted(m for m in closure if m.startswith("agentic"))
    assert not inside, (
        f"`src/api.py` now reaches {inside}, which import `agentic_os`. That "
        "package is not in the serving image because it pins an older "
        "runtime-contracts; the conflict has to be resolved before the "
        "request path can use it.")

    outside = sorted(t for t in _third_party_on(closure)
                     if t.split(".")[0] == "agentic_os")
    assert not outside, f"the serving closure imports {outside}"


def test_the_serving_image_does_not_install_it():
    assert "agentic-os @ git" not in CORE.read_text(), (
        "agentic-os is back in requirements-core.txt; it pins "
        "runtime-contracts v0.2.2 against Quantify's v0.2.4 and the image "
        "build is unsatisfiable")


def test_the_walk_would_notice_if_it_came_back(tmp_path):
    """The mutation. A graph that finds nothing passes the test above."""
    closure = _reachable_from("api")
    assert "workspace.pilot" in closure, (
        "the walk does not reach the serving path it is supposed to be about")
    assert "discovery.adapter" in closure


def test_the_runtime_the_serving_path_does_use_is_installed():
    """The other direction: what it reaches must be in the image.

    Removing a dependency because nothing imports it is only safe if the
    things that *are* imported are present. `discovery_runtime` is installed
    from the vendored submodule and `runtime_contracts` by tag.
    """
    closure = _reachable_from("api")
    used = {t.split(".")[0] for t in _third_party_on(closure)}
    core = CORE.read_text()

    for package in ("discovery_runtime", "runtime_contracts"):
        if package not in used:
            pytest.fail(f"the serving path no longer imports {package}")

    assert "runtime-contracts @ git" in core
    assert "vendor/discovery-runtime" in (SRC.parent / "Dockerfile").read_text()
