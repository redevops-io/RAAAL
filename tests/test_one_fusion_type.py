"""One authoritative `Fusion` on the serving path, and it is the runtime's.

Quantify declared its own with the same four member names and the same four
values. That is the shape that hides: an `is` / `is not` comparison against one
enum is always true for a value of the other, so

    outcome is not Fusion.AGREE                 held for every agreed field
    outcome is not Fusion.AMBIGUOUS_BY_LANGUAGE held for every ambiguity

and both went unnoticed until decisions began arriving from the runtime. The
first would have stored every agreed field as "AGREE" instead of which readers
accepted it; the second would have recorded every ambiguity as a disagreement.

Comparing by `.name` patches the call sites and leaves the duplication for the
next one to find somewhere less obvious. There is one type instead, so identity
comparisons are correct by construction.

**Values are asserted too, not only members.** A same-name duplicate is
dangerous in memory; a stale *serialized* value is the same problem after
persistence, and `provenance_of` writes `outcome.value` into stored plans.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"


def test_quantify_declares_no_fusion_of_its_own():
    """Structural, on the syntax tree.

    A grep would match this docstring and the name in every comment explaining
    the rule. The property is "no class statement with that name", which is a
    fact about the tree.
    """
    declared = []
    for file in SRC.rglob("*.py"):
        if "__pycache__" in str(file):
            continue
        try:
            tree = ast.parse(file.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "Fusion":
                declared.append(f"{file.relative_to(SRC.parent)}:{node.lineno}")
    assert not declared, (
        f"Fusion is declared inside Quantify at {declared}. Two types with the "
        "same member names make every identity comparison against one of them "
        "true for values of the other.")


def test_every_name_quantify_binds_is_the_runtimes():
    """Not one module's import — every module that holds the name.

    This used to check `src.discovery.fusion.Fusion`, which was fine while
    there was one place to check and became vacuous when that module was
    deleted. The property was never about a module: it is that no binding of
    the name `Fusion` anywhere in Quantify refers to anything but the runtime's
    type, since a single stale one restores the identity bug in whichever
    comparison happens to use it.

    Imported modules are walked, not files, because a file that declares
    nothing can still bind a wrong object through an alias.
    """
    import importlib
    import pkgutil
    import sys

    from discovery_runtime.fusion import Fusion as Runtime

    import src

    for info in pkgutil.walk_packages(src.__path__, prefix="src."):
        try:
            importlib.import_module(info.name)
        except Exception:                      # optional deps, deploy-only code
            continue

    wrong = []
    for name, module in list(sys.modules.items()):
        if not name.startswith("src.") or module is None:
            continue
        for attribute, value in list(vars(module).items()):
            if attribute == "Fusion" and value is not Runtime:
                wrong.append(f"{name}.{attribute}")
    assert not wrong, (
        f"{wrong} bind a Fusion that is not the runtime's. An `is` comparison "
        "against one enum is always true for a value of the other, which is "
        "how every agreed field was once stored as 'AGREE'.")


def test_the_walk_would_notice_a_wrong_binding(monkeypatch):
    """The mutation. A walk that imports nothing passes the test above."""
    import enum
    import sys

    import src.discovery.claims as claims

    class Fusion(enum.Enum):
        AGREE = "AGREE"

    monkeypatch.setattr(claims, "Fusion", Fusion, raising=False)

    from discovery_runtime.fusion import Fusion as Runtime

    wrong = [f"{n}.Fusion" for n, m in list(sys.modules.items())
             if n.startswith("src.") and m is not None
             and getattr(m, "Fusion", Runtime) is not Runtime]
    assert "src.discovery.claims.Fusion" in wrong


@pytest.mark.parametrize("member", ["AGREE", "DISAGREE",
                                    "INSUFFICIENT_RELATION",
                                    "AMBIGUOUS_BY_LANGUAGE"])
def test_each_outcome_serializes_to_its_own_name(member):
    """The persistence half.

    `witnesses.provenance_of` writes `outcome.value` into stored plans, and a
    reopened plan is read back against whatever the enum says today. A value
    that drifted from its name would make old plans describe outcomes that no
    longer exist — the in-memory duplication problem, one restart later.
    """
    from discovery_runtime.fusion import Fusion

    assert Fusion[member].value == member


def test_a_stored_outcome_round_trips():
    """Written as a string, read back as the same member."""
    from discovery_runtime.fusion import Fusion

    for outcome in Fusion:
        assert Fusion(outcome.value) is outcome, (
            f"{outcome.name} does not round-trip through its stored value")


def test_only_agreement_proceeds():
    """The one behavioural fact every caller depends on."""
    from discovery_runtime.fusion import Fusion

    assert Fusion.AGREE.proceeds
    for outcome in Fusion:
        assert outcome.proceeds is (outcome is Fusion.AGREE)


def test_no_fusion_comparison_uses_a_name_string():
    """`.name == "AGREE"` is the patch this file exists instead of.

    It works, and it works whichever enum the value came from — which is why it
    is worse: it makes the duplication survivable and therefore permanent.
    """
    offenders = []
    for file in SRC.rglob("*.py"):
        if "__pycache__" in str(file):
            continue
        text = file.read_text()
        for line_number, line in enumerate(text.splitlines(), 1):
            if "outcome.name" in line and ("==" in line or "!=" in line):
                offenders.append(f"{file.relative_to(SRC.parent)}:{line_number}")
    assert not offenders, (
        f"fusion outcomes compared by name string at {offenders}. There is one "
        "Fusion type now, so `is` is correct and a name comparison would hide "
        "a second type reappearing.")
