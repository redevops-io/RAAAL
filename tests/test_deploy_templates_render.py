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
    "quantify_secret_identity_admin": "quantify-test/identity-admin-password",
    "quantify_secret_model_key": "quantify-test/model-api-key",
    "quantify_secret_tunnel_token": "quantify-test/cloudflare-tunnel-token",
    "quantify_ssm_bucket": "quantify-test-deploy-transfer",
    "quantify_trace_retention_days": 90,
}


def environment():
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    # `trim_blocks=True` because that is how Ansible renders. Without it a
    # closing tag keeps the newline after it, and the concatenation fault below
    # does not reproduce — the check would pass here and fail on a host.
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)),
                      undefined=StrictUndefined, keep_trailing_newline=True,
                      trim_blocks=True)
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


class TestTheRenderScriptExportsWhatItSubstitutes:
    """`envsubst` replaces only variables that are exported.

    The masterkey line ended in an inline conditional, Ansible renders with
    trim_blocks, and the closing tag swallowed the newline — so the next
    `export` fused onto the variable name. Bash exported
    `QUANTIFY_IDENTITY_MASTERKEYexport`, the real variable was never exported,
    envsubst substituted nothing, and the identity provider refused to start
    with "masterkey must be 32 bytes, but is 0".

    Every layer was individually valid. The shell parsed, the template
    rendered, the deploy reported success, and the value was empty.
    """

    def rendered(self, **overrides):
        return environment().get_template("render-env.sh.j2").render(
            **{**VARIABLES, **overrides})

    def substituted(self, script):
        """Every variable named in an `envsubst` argument list."""
        import re

        lists = re.findall(r"render\s+\S+\s+\S+\s+\d+\s+'([^']*)'", script)
        assert lists, "found no render calls; this check is stale"
        return {name for group in lists
                for name in re.findall(r"\$\{([A-Z_]+)\}", group)}

    @pytest.mark.parametrize("identity", ["auth.quantify.club", ""])
    def test_every_substituted_variable_is_exported(self, identity):
        import re

        script = self.rendered(quantify_identity_domain=identity)
        exported = set()
        for line in script.splitlines():
            if line.startswith("export "):
                exported.update(line[len("export "):].split())

        for name in self.substituted(script):
            assert name in exported, (
                f"{name} is substituted into a rendered file and never "
                f"exported as a whole word. Exported names: {sorted(exported)}. "
                "envsubst replaces it with nothing, and the file it lands in "
                "looks complete")

    def test_no_export_line_ends_in_a_fused_word(self):
        """The specific shape, named. `export A Bexport C` is valid shell and
        exports a variable nobody meant."""
        script = self.rendered()
        for line in script.splitlines():
            if line.startswith("export "):
                assert "export" not in line[len("export "):], (
                    f"two export statements fused onto one line: {line!r}")


class TestNoCheckRunsABinaryInsideADistrolessImage:
    """A check that can never pass is worse than no check.

    The identity wait ran `docker compose exec identity curl`. That image is
    distroless — no shell, no curl, no printenv — so the command failed with
    "executable file not found" on every boot, healthy or not. It fired once
    while the provider was genuinely stuck on a migration, which read as the
    check working, and it went on failing after the provider recovered.

    The right question goes through the proxy, on the route a browser takes:
    that covers the service, the proxy's routing, and its header rewriting,
    rather than a port inside a container nobody can reach.
    """

    ROLE = (Path(__file__).resolve().parent.parent / "infra" / "ansible"
            / "roles" / "quantify" / "tasks" / "main.yml")

    #: Images with no userland. The application's own image is a normal Python
    #: base and `exec api python -c ...` is fine, which is why this is a list of
    #: services rather than a ban on `exec`.
    DISTROLESS = ("identity",)

    def commands(self):
        """Every shell/command body in the role, comments removed.

        Parsed rather than grepped. The first version of this test searched the
        file as text and matched the sentence above explaining the bug — a
        check asserting a structural property has to read the structure, or
        prose about the fault counts as the fault.
        """
        import yaml

        if not self.ROLE.exists():
            pytest.skip("no ansible role here")
        tasks = yaml.safe_load(self.ROLE.read_text()) or []
        bodies = []
        for task in tasks:
            for key in ("ansible.builtin.shell", "ansible.builtin.command",
                        "shell", "command"):
                body = task.get(key) if isinstance(task, dict) else None
                if isinstance(body, str):
                    bodies.append("\n".join(
                        line for line in body.splitlines()
                        if not line.strip().startswith("#")))
        assert bodies, "found no shell tasks; this check is stale"
        return bodies

    def test_no_task_execs_a_tool_into_one(self):
        import re

        for body in self.commands():
            for service in self.DISTROLESS:
                found = re.findall(
                    rf"exec\s+(?:-T\s+)?{service}\s+(?!true\b)(\S+)", body)
                assert not found, (
                    f"a deploy task runs {sorted(set(found))} inside the "
                    f"{service!r} container, whose image is distroless and has "
                    "no userland. The command fails identically whether the "
                    "service is healthy or dead. Ask through the proxy instead")


class TestTheProviderCommandsAreRealCommands:
    """Every flag the deploy passes the identity provider, checked against it.

    `init --masterkeyFromEnv` was written into the playbook from memory. `init`
    accepts no such flag — its only flag is `--help` — so the task would have
    failed on the host with "unknown flag", after a deploy, in the middle of
    the one sequence that had already cost several database recreates.

    Skipped when Docker cannot run, because this is the only check here that
    needs something outside the repository. That is worth it: the alternative
    is discovering the flag list one failed deploy at a time.
    """

    ROLE = (Path(__file__).resolve().parent.parent / "infra" / "ansible"
            / "roles" / "quantify" / "tasks" / "main.yml")
    COMPOSE = (Path(__file__).resolve().parent.parent / "infra" / "ansible"
               / "roles" / "quantify" / "templates" / "docker-compose.yml.j2")

    def image(self) -> str:
        import re

        if not self.COMPOSE.exists():
            pytest.skip("no compose template here")
        found = re.search(r"image:\s*(ghcr\.io/zitadel/zitadel:\S+)",
                          self.COMPOSE.read_text())
        if not found:
            pytest.skip("no identity image pinned here")
        return found.group(1)

    def invocations(self):
        """(subcommand, flags) for every provider command the deploy runs."""
        import re

        if not self.ROLE.exists():
            pytest.skip("no ansible role here")
        text = "\n".join(line for line in self.ROLE.read_text().splitlines()
                         if not line.strip().startswith("#"))
        found = []
        for line in re.findall(r"identity\s+((?:init|setup|start)\b[^\n\\]*)",
                               text):
            words = line.split()
            found.append((words[0], [w for w in words[1:]
                                     if w.startswith("--")]))
        assert found, "found no provider invocations; this check is stale"
        return found

    def help_for(self, image: str, subcommand: str) -> str:
        import subprocess

        done = subprocess.run(
            ["docker", "run", "--rm", image, subcommand, "--help"],
            capture_output=True, text=True, timeout=300)
        return done.stdout + done.stderr

    def test_every_flag_is_one_the_subcommand_accepts(self):
        import shutil
        import subprocess

        if shutil.which("docker") is None:
            pytest.skip("docker is not available here")
        image = self.image()
        try:
            subprocess.run(["docker", "image", "inspect", image],
                           capture_output=True, timeout=60, check=True)
        except Exception:  # noqa: BLE001 - not pulling in a test run
            pytest.skip(f"{image} is not present locally")

        for subcommand, flags in self.invocations():
            text = self.help_for(image, subcommand)
            for flag in flags:
                name = flag.split("=")[0]
                assert name in text, (
                    f"the deploy runs `zitadel {subcommand} {name}` and "
                    f"{subcommand} does not accept {name}. Its flags:\n"
                    + "\n".join(line for line in text.splitlines()
                                 if line.strip().startswith("-")))
