"""Quantify may orchestrate Discovery. It may not re-implement it.

The deletion of `fusion.py` and `pipeline.py` removed the duplication. Nothing
stops it coming back — and it would not come back named `fusion.py` in a
directory called `discovery`. It would come back as a helper next to the call
site, doing a small obvious thing the runtime already does, because that is how
every one of these started.

So the guard is about *ownership of behaviour*, not about a path. Quantify is
still allowed application-specific orchestration: which reader runs, what a
dimension means, what `£2.5k` is worth, which decisions become open questions,
when to ask one. What it must not hold is an implementation of:

    reader fusion             several readings of one dimension -> an outcome
    comparison semantics      whether two written values are the same value
    reading merge             several readers -> one settled set
    intent drafting           a merged reading -> a draft intent
    sealing                   a draft becoming a sealed `VerifiedIntent`

**Sabotage, not inspection.** A static scan only recognises an implementation
it has been taught to name, and the next one will be named something else. So
each capability is proved by removing it: the owning symbol is replaced with
one that raises, a real production path runs, and the failure must come out. A
second implementation anywhere reachable shows up as a path that succeeds
without the owner — precisely the state this guard exists to detect and
precisely what no name-based check can see.

**Each capability names the path that exercises it**, because they are not the
same path and assuming one covers all of them produced three false alarms while
this was being written. `pilot.read` fuses and seals; it never calls
`merge_readings` or `draft_intent`, which the adapter's own drafting path uses.
Sabotaging a capability through a path that does not reach it looks exactly
like finding a duplicate implementation.

**What sabotage cannot see, stated so the green is not over-read.** It proves
no *reachable* second implementation on the paths named here. A duplicate that
nothing calls yet is invisible to it, which is what the static half is for: a
weaker check, deliberately kept, because dead code becomes live code.
"""
from __future__ import annotations

import ast
import os
import pathlib
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

DECLARED = {"QUANTIFY_PILOT_READER": "recorded",
            "QUANTIFY_PARSER_MODE": "RUNTIME",
            "QUANTIFY_PARSER_MODEL": "claude-sonnet-5",
            "ANTHROPIC_API_KEY": "unused",
            "PILOT_DATA_POLICY": "SYNTHETIC_ONLY"}

RECORDED = "invest $500 monthly into VTI"


class Sabotaged(BaseException):
    """Raised in place of a capability that has been removed.

    `BaseException`, not `Exception`, and that is load-bearing. `pilot._intent`
    wraps `seal()` in a bare `except Exception` — correctly: a refusal to seal
    is how the page learns it has a question to ask, not an error. A sabotage
    the production path is designed to swallow proves nothing, and would have
    reported sealing as un-owned when it is delegated correctly.
    """


@pytest.fixture()
def declared(monkeypatch):
    from src.deploy import context as deploy_context

    settings = deploy_context.resolve({**os.environ, **DECLARED})
    monkeypatch.setattr(deploy_context, "current", lambda: settings)
    return settings


def _pilot_read(with_syntax=False):
    """The production serving path, not a reconstruction of it."""
    from src.discovery.hosted_recording import RecordedHostedReader
    from src.discovery.schema import QUANTIFY_SCHEMA
    from src.discovery.syntax_stanza import RecordedReader
    from src.discovery.witnesses import BOTH, MODEL_ONLY
    from src.workspace.pilot import read

    return read(RECORDED, RecordedHostedReader(), schema=QUANTIFY_SCHEMA,
                profile=BOTH if with_syntax else MODEL_ONLY,
                syntax_reader=RecordedReader() if with_syntax else None)


def _adapter_draft():
    """The adapter's drafting path, which is where merging and drafting live."""
    from src.discovery import adapter
    from src.workspace import pilot_routes

    reader = adapter.ReaderAdapter(pilot_routes.configured_reader())
    return adapter.intent_from([reader], RECORDED)


#: Capability -> (owning symbols, the paths that exercise it).
#:
#: A capability lists every path it can be observed on. `same_value` appears
#: only on the two-witness lane and that is not an omission: with one witness
#: there is nothing to compare, and forcing it onto the model-only lane would
#: mean inventing a second reader to satisfy a guard.
OWNED = {
    "reader fusion": (
        [("discovery_runtime", "fuse"), ("discovery_runtime.fusion", "fuse")],
        ["pilot.read model-only", "pilot.read two-witness", "adapter.draft"]),
    "comparison semantics": (
        [("discovery_runtime", "same_value"),
         ("discovery_runtime.fusion", "same_value")],
        ["pilot.read two-witness"]),
    "reading merge": (
        [("discovery_runtime", "merge_readings"),
         ("discovery_runtime.reader", "merge_readings")],
        ["adapter.draft"]),
    "intent drafting": (
        [("discovery_runtime", "draft_intent"),
         ("discovery_runtime.intent", "draft_intent")],
        ["adapter.draft"]),
    "sealing": (
        [("runtime_contracts", "VerifiedIntent.seal")],
        ["pilot.read model-only", "pilot.read two-witness"]),
}

PATHS = {
    "pilot.read model-only": lambda: _pilot_read(False),
    "pilot.read two-witness": lambda: _pilot_read(True),
    "adapter.draft": _adapter_draft,
}


def _remove(monkeypatch, capability):
    """Replace a capability everywhere it is bound, not where it is defined.

    Patching the defining module alone is the mistake this project has now made
    twice. `from discovery_runtime import fuse` copies the object into the
    importing module's namespace, and a later patch on the original leaves
    every such copy pointing at the real thing — a sabotage that never happened
    reads exactly like a duplicate implementation.

    So the object is found first and every module holding *that object* under
    any name is rebound. Identity, not name: a module that imported it as
    `_fuse` is patched too.
    """
    import importlib

    def gone(*args, **kwargs):
        raise Sabotaged(capability)

    symbols, _ = OWNED[capability]
    originals = []
    replaced = []

    for module_name, symbol in symbols:
        module = importlib.import_module(module_name)
        if "." in symbol:                       # a method on a contract type
            type_name, attribute = symbol.split(".", 1)
            owner = getattr(module, type_name)
            assert hasattr(owner, attribute), f"{symbol} does not exist"
            monkeypatch.setattr(owner, attribute, gone)
            replaced.append((owner, attribute, getattr(owner, attribute)))
            continue
        found = getattr(module, symbol, None)
        if found is not None and not any(found is o for o in originals):
            originals.append(found)

    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            contents = list(vars(module).items())
        except TypeError:
            continue
        for name, value in contents:
            # Identity, never equality. `==` against an arbitrary attribute of
            # an arbitrary loaded module is not a boolean question: comparing
            # against a polars `Expr` returns another `Expr`, and `if` on it
            # raises. A scan across every module in the process has to touch
            # only operations that are total.
            if any(value is o for o in originals):
                monkeypatch.setattr(module, name, gone, raising=False)
                replaced.append((module, name, value))

    assert replaced, f"nothing holds {capability}; the sabotage would prove nothing"
    return replaced


@pytest.mark.parametrize("path", sorted(PATHS))
def test_the_path_works_before_anything_is_removed(declared, path):
    """The control.

    Without it a sabotage proves nothing: a path already broken for its own
    reasons raises, and the guard reads as green.
    """
    assert PATHS[path]() is not None


CASES = [(capability, path)
         for capability, (_, paths) in sorted(OWNED.items())
         for path in paths]


@pytest.mark.parametrize("capability,path", CASES,
                         ids=[f"{c}|{p}" for c, p in CASES])
def test_removing_a_capability_stops_the_path(declared, monkeypatch,
                                              capability, path):
    """Every capability, on every path that reaches it.

    Both pilot lanes appear because the two have twice diverged: last-wins on
    SET members was fixed on the single-witness lane and stayed broken on the
    two-witness one for a further sitting, found only by running both. A guard
    that checks one lane certifies half a codebase.
    """
    _remove(monkeypatch, capability)
    with pytest.raises(Sabotaged):
        PATHS[path]()


def test_the_paths_between_them_reach_every_capability():
    """No capability is listed with an empty path list.

    That is how a guard quietly stops checking something: the symbol stays in
    the table, the path that exercised it is removed, and the row goes on
    looking like coverage.
    """
    uncovered = [c for c, (_, paths) in OWNED.items() if not paths]
    assert not uncovered, f"{uncovered} are declared owned and never exercised"
    unknown = {p for _, paths in OWNED.values() for p in paths} - set(PATHS)
    assert not unknown, f"{sorted(unknown)} are named and not runnable"


def test_a_local_reimplementation_answers_the_same_question_differently(declared):
    """The mutation. Without it the tests above are a claim about my patching.

    The smallest plausible duplicate is written out — a dozen lines and a
    reasonable name, which is all the duplication ever was — and shown to
    answer the question the runtime answers, while answering it differently.
    That difference is why a second implementation is a defect rather than a
    redundancy.
    """
    import discovery_runtime
    from discovery_runtime.fusion import Decision, Fusion, Proposal

    from src.discovery.adapter import NORMALIZERS

    def local_fusion(dimension, proposals):
        """Quantify deciding for itself."""
        values = {str(getattr(p, "value", p)) for p in proposals}
        if len(values) <= 1:
            return Decision(dimension=dimension, outcome=Fusion.AGREE,
                            value=next(iter(values), None))
        return Decision(dimension=dimension, outcome=Fusion.DISAGREE, value=None)

    same = [Proposal(value="500", reader_id="a"),
            Proposal(value="500", reader_id="b")]
    assert local_fusion("amount", same).outcome is Fusion.AGREE, (
        "the stand-in does not answer the question, so it is not a stand-in "
        "for the duplication this guard forbids")

    # `$500` and `500` are the same amount to the runtime's NUMBER rule and two
    # different strings to anything that reimplements comparison casually.
    mixed = [Proposal(value="$500", reader_id="a"),
             Proposal(value="500", reader_id="b")]
    assert local_fusion("amount", mixed).outcome is Fusion.DISAGREE
    assert discovery_runtime.fuse(
        "amount", mixed, mode="NUMBER",
        normalizers=NORMALIZERS).outcome is Fusion.AGREE, (
        "the runtime and the local copy agree here, so this mutation does not "
        "show what a second implementation costs")


def test_a_fallback_defeats_the_sabotage(declared, monkeypatch):
    """And the sabotage is shown to be defeatable, which is the point.

    A path that answers fusion elsewhere when the runtime's is gone survives
    the removal — the exact state the parametrised test reports as a failure,
    demonstrated rather than asserted. Without this, a green sabotage could
    equally mean the mechanism is inert.

    The restoration reads the record `_remove` returns rather than scanning for
    what it did. Scanning was wrong twice over: it looked only for bindings
    still named `fuse`, so a module holding the same object under another name
    stayed sabotaged; and it compared `__name__` by equality across every
    loaded module, which raises the moment one of them is polars. A patch that
    knows what it changed does not need to go looking.
    """
    replaced = _remove(monkeypatch, "reader fusion")

    with pytest.raises(Sabotaged):
        _pilot_read()

    # Now put a second implementation back — here, the real one under its
    # original bindings, which is the mildest possible version of the defect —
    # and watch the sabotage stop detecting anything.
    for owner, name, original in replaced:
        monkeypatch.setattr(owner, name, original, raising=False)

    assert _pilot_read() is not None, (
        "with a second implementation in place the path still failed, so the "
        "sabotage is not what the parametrised test is measuring")


#: Symbols whose definition in a Discovery-participating module would mean the
#: behaviour came back. Weaker than the sabotage — a new duplicate arrives
#: under a new name and this list will not know it — and kept because an
#: unreachable duplicate is invisible to sabotage and becomes reachable later.
OWNED_SYMBOLS = {"fuse", "merge_readings", "draft_intent", "same_value",
                 "Fusion", "Decision", "VerifiedIntent", "seal_intent"}

#: Where the rule applies: modules that take part in Discovery, determined by
#: whether they import it. Not a directory — `src/discovery/` holds the domain
#: semantics and is entirely allowed to, while a helper under `src/workspace/`
#: that imports the runtime is squarely in scope.
#:
#: The scoping is also what makes the name list usable. `Decision` names an
#: unrelated class in `market_data`, `policy` and `telemetry`, and `same_value`
#: names decimal comparison in `db`. A guard that flagged those would be
#: reporting a name collision as an architectural violation, and would be
#: switched off within a week.
PARTICIPATES = ("discovery_runtime", "runtime_contracts", "src.discovery",
                ".discovery", "..discovery")

#: Modules allowed to define an owned name, with the reason. An entry is a
#: claim that the definition extends the runtime rather than replacing it, and
#: it has to say how.
PERMITTED = {
    "discovery/claims.py": (
        {"Decision"},
        "a carrier, not an outcome. It holds what Quantify needs beside a "
        "decision — materiality, the model proposal, the syntax evidence, the "
        "policy version — none of which the runtime's Decision has or should. "
        "The outcome it carries is the runtime's Fusion, produced by the "
        "runtime's fuse; nothing in this module decides one."),
}


def _participates(tree: ast.AST) -> bool:
    """Whether a module takes part in Discovery, from its imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = ("." * node.level) + (node.module or "")
            if any(module.startswith(p) for p in PARTICIPATES):
                return True
        elif isinstance(node, ast.Import):
            if any(a.name.startswith(p) for a in node.names for p in PARTICIPATES):
                return True
    return False


def local_implementations(root: pathlib.Path, *, scoped: bool = False) -> dict:
    """Owned names defined — not imported — under `root`, by file.

    Definitions only. `from discovery_runtime import fuse` binds the name and
    is the opposite of the defect: it is Quantify using the runtime's one.

    With `scoped`, only modules that take part in Discovery are examined.
    """
    found = {}
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        if scoped and not _participates(tree):
            continue
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                if node.name in OWNED_SYMBOLS:
                    defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in OWNED_SYMBOLS:
                        defined.add(target.id)
        if defined:
            found[str(path)] = sorted(defined)
    return found


def _unpermitted(root: pathlib.Path) -> dict:
    out = {}
    for path, names in local_implementations(root, scoped=True).items():
        allowed: set = set()
        for suffix, (permitted, _) in PERMITTED.items():
            if path.endswith(suffix):
                allowed = permitted
        remaining = sorted(set(names) - allowed)
        if remaining:
            out[path] = remaining
    return out


def test_quantify_defines_none_of_the_owned_names():
    found = _unpermitted(SRC)
    assert not found, (
        f"Quantify defines names owned by discovery-runtime: {found}. "
        "Orchestration around Discovery is fine; an implementation of its "
        "semantics is the duplication the migration removed.")


def test_every_permission_still_describes_something_that_exists():
    """A permission for a file that has gone is one nobody re-reads.

    It is also how an exception list starts covering things it was never
    argued for: the path stays and the content changes underneath it.
    """
    for suffix, (names, reason) in PERMITTED.items():
        path = SRC / suffix
        assert path.exists(), f"{suffix} is permitted and does not exist"
        defined = set(local_implementations(path.parent).get(str(path), ()))
        assert names <= defined, (
            f"{suffix} is permitted to define {sorted(names)} and defines "
            f"{sorted(defined)}; the permission has outlived its subject")
        assert len(reason.split()) >= 20, (
            f"{suffix} is permitted without an argument. A declared exception "
            "is not a justified one — it has to say what property requires it.")


def test_the_static_guard_catches_a_planted_implementation(tmp_path):
    """The mutation for the static half.

    A guard whose walk is wrong returns an empty dict and passes forever.
    """
    planted = tmp_path / "helpers.py"
    planted.write_text(
        "def fuse(dimension, proposals):\n"
        "    return proposals[0] if proposals else None\n")
    assert local_implementations(tmp_path) == {str(planted): ["fuse"]}


def test_an_import_of_an_owned_name_is_not_a_violation(tmp_path):
    """The other half, and the reason this is not a grep.

    `from discovery_runtime import fuse` contains the word and is Quantify
    doing exactly the right thing. A check that cannot tell a definition from a
    use forces every correct call site to be written around it.
    """
    using = tmp_path / "caller.py"
    using.write_text(
        "from discovery_runtime import fuse\n"
        "\n"
        "def decide(dimension, proposals):\n"
        "    return fuse(dimension, proposals)\n")
    assert local_implementations(tmp_path) == {}


def test_the_scoping_excludes_a_module_that_does_not_touch_discovery(tmp_path):
    """`Decision` is a common word."""
    unrelated = tmp_path / "unrelated.py"
    unrelated.write_text(
        "class Decision:\n"
        "    '''Whether to publish a benchmark version.'''\n")
    assert local_implementations(tmp_path, scoped=True) == {}
    assert local_implementations(tmp_path) == {str(unrelated): ["Decision"]}, (
        "the unscoped walk missed it, so the scoping is not what excluded it")


def test_the_scoping_covers_a_participant_outside_the_discovery_package(tmp_path):
    """The scope is participation, not a directory.

    A helper under `src/workspace/` that imports the runtime and then defines
    its own `fuse` is exactly the return path this guard exists for, and it is
    nowhere near `src/discovery/`.
    """
    participant = tmp_path / "workspace_helper.py"
    participant.write_text(
        "from discovery_runtime import merge_readings\n"
        "\n"
        "def fuse(dimension, proposals):\n"
        "    return proposals[0] if proposals else None\n")
    assert local_implementations(tmp_path, scoped=True) == {
        str(participant): ["fuse"]}


def test_the_limitation_is_recorded():
    """The guard says what it cannot do, in itself."""
    flat = " ".join(__doc__.split())
    assert "A duplicate that nothing calls yet is invisible to it" in flat
    assert "no *reachable* second implementation" in flat
