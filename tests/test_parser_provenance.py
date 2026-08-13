"""What a plan records about the reader that interpreted it.

`ModelTarget.identity()` is what a stored plan carries, and its whole purpose is
to say how the sentence was read — so that reopening it later cannot re-read it
against a parser that has since moved. A provenance record naming the wrong
house is worse than one naming none: it reads as authoritative and is false.

It named the wrong house for as long as there was more than one. The provider
was the literal `"anthropic"` whenever the deployment was model-assisted,
written when Anthropic was the only option and left behind when a provider
became declarable. The deployment of d074390 printed

    PARSER={"mode": "MODEL_ASSISTED", "provider": "anthropic",
            "model": "gpt-5.4-2026-03-05", ...}

a contradiction inside a single object, while the reader that actually answered
was OpenAI's. Nothing caught it: the deploy asserts that the planned model
appears in the identity, and it did.
"""
from __future__ import annotations

import os

import pytest

from src.deploy.context import PROVIDER_KEY_VARS, ParserProvider, resolve

CASES = [
    ("OPENAI", "gpt-5.4-2026-03-05", "openai"),
    ("OPENAI", "gpt-4.1-2025-04-14", "openai"),
    ("ANTHROPIC", "claude-sonnet-5", "anthropic"),
]


def identity(provider: str, model: str) -> dict:
    environ = dict(os.environ,
                   QUANTIFY_PARSER_MODE="MODEL_ASSISTED",
                   QUANTIFY_PARSER_PROVIDER=provider,
                   QUANTIFY_PARSER_MODEL=model)
    return resolve(environ).model.identity()


class TestThePlanNamesTheReaderThatReadIt:
    @pytest.mark.parametrize("provider,model,expected", CASES)
    def test_the_declared_provider_is_recorded(self, provider, model, expected):
        assert identity(provider, model)["provider"] == expected

    @pytest.mark.parametrize("provider,model,expected", CASES)
    def test_the_model_is_recorded_beside_it(self, provider, model, expected):
        assert identity(provider, model)["model"] == model

    def test_two_providers_do_not_record_the_same_identity(self):
        """The discriminating case. A hardcoded provider passes every
        single-provider assertion above; only a comparison across two shows it,
        which is why this is the test that would have caught the original."""
        one = identity("OPENAI", "gpt-5.4-2026-03-05")
        other = identity("ANTHROPIC", "claude-sonnet-5")
        assert one["provider"] != other["provider"]


class TestTheIdentityCannotContradictItself:
    """The specific shape the defect took: a provider and a model inside one
    object naming different vendors. A deployment can be misconfigured that way
    and the plan must not describe it as coherent."""

    #: Which vendor a model name belongs to, by the prefix its publisher uses.
    #: Deliberately not a lookup of every model ever released — the point is to
    #: catch a provider and a model that cannot both be right.
    VENDOR_PREFIX = {"gpt-": "openai", "claude-": "anthropic"}

    @pytest.mark.parametrize("provider,model,expected", CASES)
    def test_the_provider_and_the_model_name_one_vendor(self, provider, model,
                                                         expected):
        recorded = identity(provider, model)
        implied = next((v for p, v in self.VENDOR_PREFIX.items()
                        if recorded["model"].startswith(p)), None)
        assert implied is not None, f"{recorded['model']!r} names no known vendor"
        assert recorded["provider"] == implied, (
            f"the plan records provider {recorded['provider']!r} beside model "
            f"{recorded['model']!r}, which belongs to {implied!r}. One of them "
            "is wrong and the plan cannot say which")

    def test_the_key_variable_follows_the_provider(self):
        """The other half of the same fact. If the identity says openai, the
        credential the deployment injected has to be the one OpenAI reads —
        otherwise the plan names a reader that could not have answered."""
        assert PROVIDER_KEY_VARS[ParserProvider.OPENAI] == "OPENAI_API_KEY"
        assert PROVIDER_KEY_VARS[ParserProvider.ANTHROPIC] == "ANTHROPIC_API_KEY"


class TestADeterministicDeploymentClaimsNoReader:
    def test_it_names_no_provider_and_no_model(self):
        """A deployment that reads nothing must not record a reader. An empty
        string here is a statement; a provider name would be a claim that a
        model was consulted when none was."""
        environ = dict(os.environ, QUANTIFY_PARSER_MODE="DETERMINISTIC")
        recorded = resolve(environ).model.identity()
        assert recorded["provider"] == ""
        assert recorded["model"] == ""
