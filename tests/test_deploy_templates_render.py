"""Every deployment template, rendered before it reaches a host.

`render-env.sh.j2` shipped with `${#VAR}` in it — bash's parameter-length
expansion, whose first two characters Jinja reads as the start of a comment. The
playbook passed `--syntax-check`, which checks the playbook and not the
templates it installs, so the failure arrived mid-deploy as "Missing end of
comment tag" against a file, naming neither the variable nor the line.

Then the comment written to explain the bug quoted the syntax it was warning
about and reintroduced it. Twice in one file is the argument for a test rather
than for care.

This renders each template with the variables an actual deploy passes. It is
not a check that the output is correct — that is what the deploy's own
assertions are for — only that the template is a template, which is the class of
failure that wastes a deploy.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2")

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "infra" / "ansible" / "roles" / "quantify" / "templates"

#: What Terraform hands Ansible. Kept here rather than read from a live
#: `terraform output`, so this runs with no cloud credentials — the values only
#: have to be present and plausible, since what is under test is whether the
#: template parses and resolves the names it uses.
VARIABLES = {
    "quantify_alb_dns": "internal-alb.us-east-1.elb.amazonaws.com",
    "quantify_build_commit": "0" * 40,
    "quantify_build_release_ref": "master",
    "quantify_build_snapshot_id": "syn-2026-08",
    "quantify_data_policy": "SYNTHETIC_ONLY",
    "quantify_database_host": "db.us-east-1.rds.amazonaws.com",
    "quantify_domain": "quantify.club",
    "quantify_identity_domain": "auth.quantify.club",
    "quantify_image": "registry/quantify@sha256:" + "0" * 64,
    "quantify_instance_id": "i-0",
    "quantify_log_group": "/quantify/test/application",
    "quantify_parser_fallback": "REFUSE",
    "quantify_parser_mode": "RUNTIME",
    "quantify_parser_model": "gpt-5.4-2026-03-05",
    "quantify_parser_prompt_version": "",
    "quantify_parser_provider": "OPENAI",
    "quantify_region": "us-east-1",
    "quantify_registry_host": "registry",
    "quantify_secret_basic_auth": "quantify-test/workspace-basic-auth",
    "quantify_secret_database": "quantify-test/database-password",
    "quantify_secret_identity_key": "quantify-test/identity-masterkey",
    "quantify_secret_model_key": "quantify-test/model-api-key",
    "quantify_secret_tunnel_token": "quantify-test/cloudflare-tunnel-token",
    "quantify_ssm_bucket": "quantify-test-deploy-transfer",
    "quantify_trace_retention_days": 90,
}


def environment():
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    env = Environment(loader=FileSystemLoader(str(TEMPLATES)),
                      undefined=StrictUndefined, keep_trailing_newline=True)
    # Ansible ships filters plain Jinja does not. Stubbed rather than skipped,
    # because a template using one still has to parse — and the parse is what
    # this file is about.
    env.filters["regex_search"] = lambda value, *a, **k: [""]
    env.filters["regex_replace"] = lambda value, *a, **k: str(value)
    env.filters["to_json"] = json.dumps
    env.filters["b64encode"] = lambda value, *a, **k: str(value)
    return env


def templates():
    return sorted(TEMPLATES.glob("*.j2")) if TEMPLATES.exists() else []


@pytest.mark.skipif(not TEMPLATES.exists(), reason="no ansible role here")
@pytest.mark.parametrize("template", [p.name for p in templates()])
def test_it_renders(template):
    environment().get_template(template).render(**VARIABLES)


@pytest.mark.skipif(not TEMPLATES.exists(), reason="no ansible role here")
@pytest.mark.parametrize("template", [p.name for p in templates()])
def test_it_renders_without_an_identity_provider(template):
    """The other configuration, which has to keep working.

    Everything about the provider is conditional on a hostname, and empty is
    what the pilot ran under for months. A template that only renders when
    identity is configured would break the deployment that has no accounts —
    and that is the one currently serving people.
    """
    environment().get_template(template).render(
        **{**VARIABLES, "quantify_identity_domain": ""})


@pytest.mark.skipif(not TEMPLATES.exists(), reason="no ansible role here")
def test_no_template_contains_a_bash_length_expansion():
    """The specific fault, named.

    The render test above catches it, and this says what it is. A future
    `${#var}` fails here with the file and the reason rather than with a Jinja
    parse error that mentions neither.
    """
    offenders = [p.name for p in templates() if "${#" in p.read_text()]
    assert not offenders, (
        f"{offenders} use bash's parameter-length expansion, whose first two "
        "characters open a Jinja comment. Use `wc -c` instead")
