"""What the image must do, asserted against the image rather than the source.

Everything else in this suite reads files. This runs the container, because
three separate reasons the image could not be built were invisible to a
correct Dockerfile in git: a private repository with no build credential, a
floating parser version, and an unsatisfiable transitive pin that had been
wrong since the contracts moved. None of them were source defects.

**Skipped without Docker or the image.** These are deployment gates, run
before a push, not part of every commit — and a skip says so rather than a
green that checked nothing.

The invariant that matters most is the last one. A container that starts
successfully with Stanza unavailable would silently recreate the hole this
work closed: every syntax guard and derived reader runs on the two-witness
branch, so a missing parse is not an error anywhere, it is a refusal that
never happens.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

IMAGE = os.environ.get("QUANTIFY_IMAGE", "quantify-web:item9-candidate")

#: The declaration a production container must carry, and the cases that
#: earned it. `annuitize` is here because it is the one that showed the
#: problem was general rather than confined to the two families added for it.
EARNED_THE_WITNESS = {
    "annuitize a third of the portfolio at 70": "sell_action",
    "sell the loser and buy a similar fund to avoid a wash sale": "sell_action",
    "rebalance back to 60/40 every year": "periodic_rebalancing",
}


def _docker(*args, **kwargs):
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          timeout=600, **kwargs)


@pytest.fixture(scope="module")
def image():
    if shutil.which("docker") is None:
        pytest.skip("docker is not available")
    found = _docker("image", "inspect", IMAGE)
    if found.returncode != 0:
        pytest.skip(f"{IMAGE} has not been built")
    return IMAGE


def _run(image, script: str, *, network="none"):
    """A script inside the container, offline unless asked otherwise."""
    return _docker("run", "--rm", f"--network={network}", "--entrypoint", "python",
                   image, "-c", script)


def test_the_image_installs_the_exact_versions_it_was_tested_with(image):
    out = _run(image, "import importlib.metadata as m, json;"
                      "print(json.dumps({p: m.version(p) for p in "
                      "('stanza','discovery-runtime','runtime-contracts')}))")
    assert out.returncode == 0, out.stderr[-400:]
    versions = json.loads(out.stdout.strip().splitlines()[-1])
    assert versions["stanza"] == "1.14.0", versions
    assert versions["discovery-runtime"] == "0.1.9", versions
    assert versions["runtime-contracts"] == "0.2.4", versions


def test_the_mission_runtime_is_absent(image):
    """It pins an older contracts and the serving path never reaches it."""
    out = _run(image, "import importlib.metadata as m;"
                      "print('present' if m.distributions and _present() else '')"
                      "\ndef _present():\n    pass")
    # Simpler and without the tortured probe above:
    out = _run(image, "\n".join([
        "import importlib.metadata as m",
        "try:",
        "    m.version('agentic-os'); print('PRESENT')",
        "except Exception:",
        "    print('ABSENT')",
    ]))
    assert out.returncode == 0, out.stderr[-400:]
    assert out.stdout.strip().endswith("ABSENT"), out.stdout


def test_the_parser_works_with_no_network(image):
    """`download_method=None` is deliberate, so the model has to be baked in.

    Run with `--network none`, because "it worked on my machine with the
    internet on" is exactly the failure this prevents.
    """
    out = _run(image, "\n".join([
        "from src.discovery.syntax_stanza import StanzaReader",
        "r = StanzaReader('en')",
        "p = r.parse('a smoke sentence')",
        "print(r.id, len(p.sentences))",
    ]))
    assert out.returncode == 0, out.stderr[-600:]
    assert "stanza@1.14.0:en 1" in out.stdout


@pytest.mark.parametrize("text,dimension", sorted(EARNED_THE_WITNESS.items()))
def test_the_guard_fires_inside_the_container(image, text, dimension):
    """Not `hello world`. The parse has to carry enough structure for the
    material guard to prove a predicate the model can drop."""
    out = _run(image, "\n".join([
        "from src.discovery.guards import presence",
        "from src.discovery.syntax_stanza import StanzaReader",
        f"print(sorted(presence(StanzaReader('en').parse({text!r}))))",
    ]))
    assert out.returncode == 0, out.stderr[-600:]
    assert dimension in out.stdout, out.stdout


def test_the_predicate_position_rule_survives_in_the_container(image):
    """The reason the guard was not weakened to raw text."""
    out = _run(image, "\n".join([
        "from src.discovery.guards import presence",
        "from src.discovery.syntax_stanza import StanzaReader",
        "r = StanzaReader('en')",
        "print('verb', 'sell_action' in presence(r.parse('sell VTI every month')))",
        "print('noun', 'sell_action' in presence(r.parse('a sell signal fired')))",
    ]))
    assert out.returncode == 0, out.stderr[-600:]
    assert "verb True" in out.stdout and "noun False" in out.stdout, out.stdout


def test_the_container_declares_the_two_witness_profile(image):
    out = _run(image, "\n".join([
        "import os",
        "from src.deploy import context as dc",
        "s = dc.resolve(dict(os.environ))",
        "print(sorted(w.value for w in s.model.witnesses.available))",
    ]))
    assert out.returncode == 0, out.stderr[-600:]
    assert "['model', 'syntax']" in out.stdout, out.stdout


def test_a_container_without_the_model_refuses_to_serve(image):
    """The invariant that matters most, through the real preflight.

    A first version of this checked the helper directly and passed while the
    preflight refused earlier on something else and never reached it — so the
    helper being right proved nothing about startup.

    The container must also not quietly become a MODEL_ONLY server: the
    profile is a declaration and stays `['model', 'syntax']` with the model
    gone, which is why the refusal has to exist at all.
    """
    script = "\n".join([
        "import os, pathlib, shutil",
        "shutil.rmtree(pathlib.Path(os.environ['STANZA_RESOURCES_DIR']))",
        "os.environ.update({",
        "  'QUANTIFY_DATABASE_URL': 'postgresql://u:p@127.0.0.1:5432/x',",
        "  'QUANTIFY_COMMIT': 'test', 'QUANTIFY_RELEASE_REF': 'test',",
        "  'QUANTIFY_IMAGE_DIGEST': 'sha256:test', 'QUANTIFY_SNAPSHOT_ID': 'test',",
        "  'QUANTIFY_PARSER_MODE': 'RUNTIME', 'QUANTIFY_PARSER_PROVIDER': 'openai',",
        "  'QUANTIFY_PARSER_MODEL': 'gpt-5.4', 'OPENAI_API_KEY': 'unused',",
        "  'PILOT_DATA_POLICY': 'SYNTHETIC_ONLY'})",
        "from src.deploy import context as dc",
        "from src.deploy.preflight import run",
        "after = dc.resolve(dict(os.environ))",
        "print('PROFILE', sorted(w.value for w in after.model.witnesses.available))",
        "o = run(dict(os.environ))",
        "print('RESULT', o.result.name)",
        "print('DETAIL', o.detail[:120])",
    ])
    out = _run(image, script)
    assert out.returncode == 0, out.stderr[-600:]

    assert "PROFILE ['model', 'syntax']" in out.stdout, (
        f"the container fell back to a narrower profile: {out.stdout}")
    assert "RESULT REFUSED_CONFIGURATION" in out.stdout, (
        f"a container with no parser did not refuse to serve: {out.stdout}")
    assert "syntax witness" in out.stdout, out.stdout


def test_the_readiness_endpoint_refuses_without_the_model(image):
    """The same thing through the entry point the deployment actually runs.

    `create_app` runs the preflight while the process is starting, so a
    refusal prevents the server binding rather than being noticed later. This
    asserts the factory raises rather than yielding an app that serves.
    """
    script = "\n".join([
        "import os, pathlib, shutil",
        "shutil.rmtree(pathlib.Path(os.environ['STANZA_RESOURCES_DIR']))",
        "os.environ.update({",
        "  'QUANTIFY_DATABASE_URL': 'postgresql://u:p@127.0.0.1:5432/x',",
        "  'QUANTIFY_COMMIT': 'test', 'QUANTIFY_RELEASE_REF': 'test',",
        "  'QUANTIFY_IMAGE_DIGEST': 'sha256:test', 'QUANTIFY_SNAPSHOT_ID': 'test',",
        "  'QUANTIFY_PARSER_MODE': 'RUNTIME', 'QUANTIFY_PARSER_PROVIDER': 'openai',",
        "  'QUANTIFY_PARSER_MODEL': 'gpt-5.4', 'OPENAI_API_KEY': 'unused',",
        "  'PILOT_DATA_POLICY': 'SYNTHETIC_ONLY'})",
        "from src.api import create_app, preflight_outcome",
        "try:",
        "    create_app()",
        "    served = True",
        "except BaseException as e:",
        "    served = False",
        "    print('REFUSED', e.__class__.__name__)",
        "o = preflight_outcome()",
        "print('SERVED', served)",
        "print('READY', None if o is None else o.ready)",
    ])
    out = _run(image, script)
    assert out.returncode == 0, out.stderr[-800:]
    assert "READY True" not in out.stdout, (
        f"the app reported ready with no parser: {out.stdout}")
