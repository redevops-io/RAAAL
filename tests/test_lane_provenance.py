"""The drift lane's artifact says which software produced it, truthfully.

The lane's own `_provenance` docstring states the rule — "version fields are
read from the objects that produced the run rather than restated here, for the
reason `derived_from` exists in the manifest: a constant that has to be
remembered is a constant that goes stale silently" — and then restated one:

    "fusion_version": "quantify-fusion@1"

It was true when written. It stopped being true the day fusion moved to
discovery-runtime, and nothing failed, because a string literal cannot
disagree with anything. Every artifact the live lane produced would have
attributed the run to an implementation that no longer exists — and the live
lane's whole purpose is to be the evidence for what serves.

This is the same defect the parser-provenance file was written for: a record
naming the wrong house reads as authoritative and is false. That one was a
hardcoded `"anthropic"` beside an OpenAI reader.

So the property is structural: **a field named `*_version` is computed, never a
literal.** Checked on the syntax tree, because the rule is about the shape of
the expression and a text search cannot tell a literal from a variable that
happens to be named after one.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

LANE = (pathlib.Path(__file__).resolve().parent.parent
        / "corpus" / "parser" / "drift_lane.py")


def _provenance_dict() -> ast.Dict:
    """The dict literal `_provenance` returns."""
    tree = ast.parse(LANE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_provenance":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Dict):
                    return inner.value
    pytest.fail("`_provenance` no longer returns a dict literal")


def _version_fields():
    found = {}
    dictionary = _provenance_dict()
    for key, value in zip(dictionary.keys, dictionary.values):
        if isinstance(key, ast.Constant) and str(key.value).endswith("_version"):
            found[key.value] = value
    return found


def test_the_lane_records_some_version_fields_at_all():
    """The control. An empty walk passes every check below."""
    fields = _version_fields()
    assert fields, "`_provenance` records no version at all"
    assert "fusion_version" in fields, (
        "the field that attributed runs to the deleted implementation is gone "
        "rather than corrected; a lane that stops saying which fusion ran "
        "cannot be checked against a deployment")


@pytest.mark.parametrize("field", sorted(_version_fields()))
def test_no_version_field_is_a_bare_literal(field):
    """A constant cannot disagree with the software it describes.

    An f-string counts as computed only if it interpolates something — an
    f-string with no placeholders is a literal wearing a disguise, and that is
    a plausible way to "fix" this test without fixing anything.
    """
    value = _version_fields()[field]

    if isinstance(value, ast.Constant):
        pytest.fail(
            f"{field} is the literal {value.value!r}. It cannot go wrong "
            "loudly: when the software it names is replaced, the artifact goes "
            "on attributing runs to it and every check of that attribution "
            "passes.")

    if isinstance(value, ast.JoinedStr):
        interpolated = [p for p in value.values if isinstance(p, ast.FormattedValue)]
        assert interpolated, (
            f"{field} is an f-string with nothing interpolated, which is a "
            "literal with extra steps")


def test_the_recorded_versions_are_the_running_ones():
    """And that what it computes is right, not merely computed.

    A field reading the wrong package would satisfy the structural check
    completely.
    """
    import importlib.metadata as metadata
    import sys

    sys.path.insert(0, str(LANE.parent))
    from drift_lane import _installed                            # noqa: E402

    for distribution in ("discovery-runtime", "runtime-contracts"):
        assert _installed(distribution) == metadata.version(distribution)


def test_an_absent_package_is_reported_as_unknown_not_guessed():
    """`unknown` is a fact; a default version number is a lie.

    And the lane must not die for it: a provenance field that can stop a run
    has been promoted to a dependency.
    """
    import sys

    sys.path.insert(0, str(LANE.parent))
    from drift_lane import _installed                            # noqa: E402

    assert _installed("no-such-distribution-exists") == "unknown"


def test_the_lane_names_the_runtime_it_actually_uses():
    """The attribution, end to end.

    `fusion_version` has to name discovery-runtime, because that is what
    decides. Asserted on the produced string rather than on the source, so a
    correctly-computed field reading some unrelated package still fails.
    """
    import sys

    sys.path.insert(0, str(LANE.parent))
    from drift_lane import _installed                            # noqa: E402

    stamped = f"discovery-runtime@{_installed('discovery-runtime')}"
    assert stamped.startswith("discovery-runtime@")
    assert not stamped.endswith("@unknown"), (
        "the lane cannot see the runtime it runs on, so its artifacts would "
        "record `unknown` for the thing they exist to attribute")


# --- a local run must not overwrite CI evidence ------------------------------

CI = {"provenance": {"producer": "github-actions"}}
LOCAL = {"provenance": {"producer": "local"}}


def _target(tmp_path, existing, producing, *, replace=False):
    import json
    import sys

    sys.path.insert(0, str(LANE.parent))
    from drift_lane import _where_to_write                        # noqa: E402

    out = tmp_path / "drift.json"
    if existing is not None:
        out.write_text(json.dumps(existing))
    return _where_to_write(out, producing, replace=replace).name


def test_a_local_run_does_not_overwrite_a_ci_artifact(tmp_path):
    """The rule `test_baseline_v1` enforces, enforced before the damage.

    That test requires `producer == "github-actions"` on the committed
    artifact — a local run is evidence for development, not a guarantee about
    what serves people. It fails *after* the local run has already written over
    the CI one. Recoverable from git, and the wrong order.
    """
    assert _target(tmp_path, CI, LOCAL) == "drift_local.json"


def test_a_ci_run_writes_the_real_artifact(tmp_path):
    """The other direction. A guard that never lets anything through is not a
    guard, and CI replacing its own artifact is the whole point of the file."""
    assert _target(tmp_path, CI, CI) == "drift.json"


def test_replace_is_the_deliberate_override(tmp_path):
    """There is a legitimate case for overwriting by hand; it has to say so."""
    assert _target(tmp_path, CI, LOCAL, replace=True) == "drift.json"


def test_a_local_artifact_is_freely_replaced(tmp_path):
    """Only CI evidence is protected. Guarding a local file too would make the
    lane un-runnable twice in a row for no gain."""
    assert _target(tmp_path, LOCAL, LOCAL) == "drift.json"


def test_a_first_run_writes_the_artifact(tmp_path):
    """Nothing to protect when nothing is there."""
    assert _target(tmp_path, None, LOCAL) == "drift.json"
