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


class TestTheShellBlocksSurviveAnsiblesParser:
    """Two deploys have now been lost to a comment.

    The first put bash parameter-length expansion in a Jinja template, whose
    opening characters start a Jinja comment. The second put an apostrophe in a
    shell comment — Ansible splits a `shell:` block into arguments before bash
    sees it, so one unbalanced quote fails the whole playbook, and the error
    names the task rather than the word.

    Both are mechanical, both cost a round-trip to a host, and neither is the
    kind of thing care prevents twice.
    """

    ROLE = (Path(__file__).resolve().parent.parent / "infra" / "ansible"
            / "roles" / "quantify" / "tasks" / "main.yml")

    def blocks(self):
        """Every `shell:` block in the role, as text."""
        import re

        if not self.ROLE.exists():
            pytest.skip("no ansible role here")
        text = self.ROLE.read_text()
        return re.findall(r"ansible\.builtin\.shell:\s*\|\s*\n((?:[ ]{4,}.*\n|\n)+)",
                          text)

    def test_there_are_some(self):
        assert self.blocks(), "found no shell blocks; this file is stale"

    def test_no_block_has_an_unbalanced_single_quote(self):
        """Counted over the whole block, not per line.

        Per line was the first rule and it was wrong: `sh -c \'` legitimately
        opens a quote that closes several lines later, and the check failed on
        working code the moment it was written. Ansible balances across the
        argument it is splitting, so that is the unit — which still catches an
        apostrophe in prose, because one stray quote makes the whole block odd.
        """
        for block in self.blocks():
            assert block.count("'") % 2 == 0, (
                "a shell block has an unbalanced single quote. Ansible splits "
                "these into arguments before bash sees them, so an apostrophe "
                "in a comment fails the playbook:\n"
                + "\n".join(line for line in block.splitlines()
                            if line.count("'") % 2))

    def test_the_playbook_parses(self):
        """The general check behind the specific one. `--syntax-check` loads
        every task file and is the thing that would have caught both faults."""
        import shutil
        import subprocess

        if shutil.which("ansible-playbook") is None:
            pytest.skip("ansible is not installed here")
        root = Path(__file__).resolve().parent.parent / "infra" / "ansible"
        done = subprocess.run(
            ["ansible-playbook", "--syntax-check",
             "-i", str(root / "inventory.aws_ec2.yml"), str(root / "site.yml")],
            capture_output=True, text=True, timeout=300)
        assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]


class TestNothingDependsOnComposeInterpolation:
    """`${VAR}` in a compose file is substituted by compose, not by the shell.

    Compose resolves its env file relative to the working directory, and the
    deploy invokes it with `-f /opt/quantify/docker-compose.yml` from wherever
    the playbook happens to be. The substitution produced an empty string, the
    identity provider connected with a blank password, and it restarted every
    minute for ten minutes reporting "password authentication failed" — a
    message about credentials for a fault in file resolution.

    Values belong in the env file, which `env_file:` hands to the container
    verbatim. That is how the API service has always been configured, and it is
    why the API service never had this problem.
    """

    COMPOSE = (Path(__file__).resolve().parent.parent / "infra" / "ansible"
               / "roles" / "quantify" / "templates" / "docker-compose.yml.j2")

    def test_the_compose_file_interpolates_no_secrets(self):
        import re

        if not self.COMPOSE.exists():
            pytest.skip("no compose template here")
        found = re.findall(r"\$\{[A-Z_]+\}", self.COMPOSE.read_text())
        assert not found, (
            f"{sorted(set(found))} are interpolated by compose, which resolves "
            "its env file against the working directory. Put them in the env "
            "file and let `env_file:` deliver them")

    def test_every_service_reads_the_env_file(self):
        """The positive form. A service configured some other way is one whose
        configuration is not where anybody looks for it."""
        if not self.COMPOSE.exists():
            pytest.skip("no compose template here")
        rendered = environment().get_template(self.COMPOSE.name).render(**VARIABLES)
        import yaml

        services = yaml.safe_load(rendered)["services"]
        for name, service in services.items():
            if name in {"proxy", "tunnel"}:
                # Neither reads application configuration: one is a reverse
                # proxy over a file, the other takes a token on its command
                # line. Naming them is the point — a new service that skipped
                # the env file would have to be added here deliberately.
                continue
            assert service.get("env_file"), (
                f"service {name!r} does not read /opt/quantify/.env, so its "
                "configuration lives somewhere this deployment does not render")
