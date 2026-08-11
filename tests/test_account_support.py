"""Three claims about an account, and why they must stay apart.

    RECOGNIZED   the compiler can read it and represent it
    COMPARABLE   a run pins a versioned account runtime
    ENFORCED     the declared behaviour actually executes

For a Roth 401(k) today the honest answer is yes, yes, no — and the third is the
one a user would assume from the first two. A single "supported" badge collapses
them.
"""
from __future__ import annotations

import pytest

from src.workspace.account_support import LABELS, Support, support_for


class TestTheThreeStatesAreIndependent:

    def test_roth_401k_is_recognised_comparable_and_partly_enforced(self):
        """Its contribution limit is applied; the limit it *shares* with the
        plan's traditional half is not, because `cap_contribution` caps one
        account and never sees the other."""
        support = support_for("ROTH_401K")
        assert support.recognized is Support.YES
        assert support.comparable is Support.YES
        assert support.enforced is Support.PARTIAL
        assert "shared-deferral-limit" in support.unenforced_behaviours

    def test_an_unrepresentable_account_is_recognised_and_nothing_more(self):
        """An inherited IRA is read from the text and has no runtime."""
        support = support_for("INHERITED_IRA")
        assert support.recognized is Support.YES
        assert support.comparable is Support.NO
        assert support.enforced is Support.NO

    def test_no_account_claims_nothing(self):
        support = support_for("NONE_APPLIED")
        assert support.recognized is Support.NO
        assert "not recognised" in support.summary

    def test_the_summary_states_all_three(self):
        summary = support_for("ROTH_401K").summary
        assert "identified as Roth 401(k)" in summary
        assert "pinned across comparisons" in summary
        assert "some of its rules are enforced" in summary


class TestEnforcementIsDerived:
    """From realization checks, never from a maintained support list. This
    project has been caught twice by a list that had to be remembered."""

    def test_it_moves_when_a_realization_arrives(self):
        """A derivation nobody has seen change is a derivation nobody has
        tested."""
        assert support_for("ROTH_401K", implemented=()).enforced is Support.NO
        assert support_for(
            "ROTH_401K", implemented=("cap_contribution",)
        ).enforced is Support.PARTIAL
        # YES additionally requires the limit figure to be verified against the
        # published source, which the shipped table is not. See
        # tests/test_account_limits.py::TestAnUncheckedFigureIsNotEnforcement.

    def test_partial_enforcement_is_distinguishable_from_none(self):
        """"Nothing works" and "half of it works" lead somewhere different."""
        from src.runtime import AccountKind, AccountRuntime

        runtime = AccountRuntime(name="account/roth_401k", version=1,
                                 account_kind=AccountKind.ROTH_401K,
                                 annual_contribution_limit=24500.0,
                                 employer_match_rate=0.5)
        declared = {a.name for a in runtime.assumptions}
        assert len(declared) > 2, "the fixture needs several behaviours"

        # `apply_match` realized, `cap_contribution` not.
        partial = [name for name in runtime.unrealized(("apply_match",))]
        assert partial and len(partial) < len(declared)

    def test_no_declared_rules_is_not_full_enforcement(self):
        """A taxable brokerage account has no contribution rule to enforce.
        Nothing declared is still not the same as everything enforced, so this
        reports NO rather than deriving certainty from an absence."""
        support = support_for("TAXABLE")
        assert support.declared_behaviours == ()
        assert support.enforced is Support.NO
        assert "none can be enforced" in support.unenforced_behaviours[0]

    def test_there_is_no_second_support_list(self):
        import inspect

        from src.workspace import account_support

        source = inspect.getsource(account_support)
        assert "ACCOUNT_IMPLEMENTED" in source
        assert "SUPPORTED_ACCOUNTS" not in source
        assert "FULLY_SUPPORTED" not in source


class TestItReachesTheScreens:

    def test_the_confirmation_card_carries_all_three(self):
        from src.mission.compiler import compile_scenario
        from src.workspace.confirmation import build

        compiled = compile_scenario(
            "I put $500 into SPY monthly in my Roth 401(k) and never sell.",
            name="p", version=1,
            benchmark_rule="benchmark-policy/public-default@1")
        card = build(compiled, text="").account
        assert card["support"]["recognized"] == "YES"
        assert card["support"]["comparable"] == "YES"
        assert card["support"]["enforced"] == "PARTIAL"

    def test_the_template_reads_three_fields_not_one(self):
        import re

        body = open("src/workspace/templates/_confirmation.html").read()
        for field in ("support.recognized", "support.comparable",
                      "support.enforced"):
            assert field in body, field

        # Comments are not rendered, and this file contains one explaining why a
        # single badge is wrong. Scanning the raw text caught that explanation.
        rendered = re.sub(r"\{#.*?#\}", " ", body, flags=re.S).lower()
        assert "supported" not in rendered.replace("unsupported", "")

    def test_every_label_is_human_readable(self):
        for declared, label in LABELS.items():
            assert label and label != declared
