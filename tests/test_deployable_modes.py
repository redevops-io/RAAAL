"""What the application supports, and what the infrastructure can express.

These are two lists in two languages and nothing compared them. `ParserMode`
gained `RUNTIME` — the pilot interpreter, and the mode every pilot surface
branches on — and Terraform's validation still allowed only `MODEL_ASSISTED`
and `DETERMINISTIC`. There was no way to deploy the thing being built.

Nothing failed. `terraform apply` succeeded, Ansible succeeded, the deployment
identity proof passed, and `/workspace/new` served the legacy
`compile_scenario` path to every visitor while the strategy selector, the
clarification-convergence work and refusal-by-name from the capability manifest
sat behind a mode the infrastructure could not name. The symptom was a user
hard-refreshing the site and not finding a dropdown that had been shipped days
earlier.

This is the third defect of one shape in this deployment. `application_image`
drifted from `build_commit`; the environment template hardcoded
`ANTHROPIC_API_KEY` so no OpenAI model could ever have served; and the mode list
omitted the mode. Each time the infrastructure described a narrower product
than the code, and each time every check passed because the checks compared
configuration against configuration.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VARIABLES = ROOT / "infra" / "terraform" / "variables.tf"
TFVARS = ROOT / "infra" / "terraform" / "environments" / "test.tfvars"


def permitted(variable: str) -> set:
    """The values Terraform's validation allows for `variable`.

    Parsed from the configuration rather than restated here. A copy of the list
    in this file would be a third place for it to disagree.
    """
    if not VARIABLES.exists():
        pytest.skip("no terraform configuration in this checkout")
    text = VARIABLES.read_text()
    block = re.search(rf'variable "{variable}" \{{(.*?)\n\}}', text, re.S)
    assert block, f"no `{variable}` variable in variables.tf"
    condition = re.search(r'contains\(\[(.*?)\]', block.group(1), re.S)
    assert condition, f"`{variable}` has no contains() validation"
    return set(re.findall(r'"([^"]+)"', condition.group(1)))


def declared(variable: str) -> str:
    if not TFVARS.exists():
        pytest.skip("no environment tfvars in this checkout")
    found = re.search(rf'^{variable}\s*=\s*"([^"]+)"', TFVARS.read_text(),
                      re.M)
    assert found, f"{variable} is not declared in test.tfvars"
    return found.group(1)


class TestTheInfrastructureCanExpressWhatTheCodeSupports:
    def test_every_parser_mode_is_deployable(self):
        """A mode the code branches on and the infrastructure cannot name is a
        product nobody can ship."""
        from src.deploy.context import ParserMode

        missing = {m.value for m in ParserMode} - permitted("parser_mode")
        assert not missing, (
            f"{sorted(missing)} are parser modes the application supports and "
            "terraform will not accept. Code branches on them and no "
            "deployment can select them")

    def test_every_parser_provider_is_deployable(self):
        """The same check for the variable that had the same defect."""
        from src.deploy.context import ParserProvider

        missing = {p.value for p in ParserProvider} - permitted("parser_provider")
        assert not missing, (
            f"{sorted(missing)} are providers the application supports and "
            "terraform will not accept")


class TestTheDeploymentServesThePilot:
    """Not merely deployable — actually declared.

    The mode being *permitted* would have been enough to fix the immediate
    complaint and would have left the site serving the legacy compiler until
    somebody remembered to change the value too.
    """

    def test_the_environment_declares_the_runtime(self):
        assert declared("parser_mode") == "RUNTIME", (
            "this environment does not declare RUNTIME, so /workspace/new "
            "serves the legacy compile_scenario path and no pilot surface — "
            "the strategy selector included — is reachable")

    def test_the_pilot_surface_is_gated_on_exactly_that_mode(self):
        """The link between the two. If the route ever stops branching on
        RUNTIME, the assertion above stops meaning anything and this test says
        so rather than passing quietly."""
        from src.deploy.context import ParserMode
        from src.workspace import pilot_routes

        source = Path(pilot_routes.__file__).read_text()
        assert "ParserMode.RUNTIME" in source, (
            "the pilot no longer gates on ParserMode.RUNTIME; whatever it "
            "gates on now is what the environment must declare")
        assert ParserMode.RUNTIME.value == "RUNTIME"

    def test_the_provider_and_model_still_agree(self):
        """Declared together, checked together — the pair that could not both
        be right when one was hardcoded."""
        provider, model = declared("parser_provider"), declared("parser_model")
        prefix = {"OPENAI": "gpt-", "ANTHROPIC": "claude-"}[provider]
        assert model.startswith(prefix), (
            f"{provider} is declared with model {model!r}")
