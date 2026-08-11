"""A plan the account does not permit, caught before it runs.

The defect this closes: "$2,000 a month into my Roth IRA" compiled to $24,000 of
annual contributions, simulated all of it, and reported a balance built on
roughly three times the permitted amount. Nothing in the interpretation summary,
the modelling scope or the result said so.

Caught afterwards it is a number the user has already been shown and already
believes, which is why every assertion here is about the screen *before* the run.
"""
from __future__ import annotations

import pytest

from src.mission.compiler import compile_scenario
from src.workspace.confirmation import build

RULE = "benchmark-policy/public-default@1"


def view(text: str):
    return build(compile_scenario(text, name="p", version=1,
                                  benchmark_rule=RULE), text=text)


OVER = "I put $2,000 into SPY every month in my Roth IRA and never sell."
UNDER = "I put $400 into SPY every month in my Roth IRA and never sell."
TAXABLE = "I put $2,000 into SPY every month in my taxable brokerage and never sell."
DEFERRAL = "I put $1,500 into VTI every month in my Roth 401(k) and never sell."


def _render(rendered) -> str:
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader("src/workspace/templates"))
    return env.get_template("_confirmation.html").render(view=rendered)


def _block(rendered) -> str:
    """Just the over-limit panel, so an assertion about it cannot be satisfied
    by identical copy elsewhere on the page.

    Depth-counted rather than matched with a regex: the panel contains nested
    divs, and a non-greedy match would stop at the first inner `</div>` and
    silently return a fragment that happens to omit the very line under test.
    """
    import re

    html = _render(rendered)
    start = html.find('<div class="group conflict over-limit">')
    assert start != -1, "the over-limit panel did not render at all"

    depth, index = 0, start
    for tag in re.finditer(r"<(/?)div\b", html[start:]):
        depth += -1 if tag.group(1) else 1
        if depth == 0:
            index = start + tag.end()
            break
    return html[start:index]


class TestItRefusesRatherThanSimulating:

    def test_an_over_limit_plan_cannot_run(self):
        assert view(OVER).can_run is False

    def test_it_blocks_the_path(self):
        assert view(OVER).path == "BLOCKED"

    def test_the_headline_says_what_is_wrong(self):
        assert "more than the account allows" in view(OVER).headline

    def test_it_names_the_size_of_the_excess(self):
        """Being over by $16,500 is what tells a user they mis-stated the
        account rather than the cadence."""
        over = view(OVER).over_limit
        assert over["requested"] == 24_000.0
        assert over["refused"] == over["requested"] - over["permitted"]
        assert f"${over['refused']:,.0f}" in over["detail"]

    def test_nothing_is_silently_reduced(self):
        """The permitted figure is offered as a choice, never applied for the
        user. Quietly capping produces a balance they never described while
        every displayed number looks deliberate."""
        over = view(OVER).over_limit
        assert any(c["value"] == "reduce" for c in over["choices"])
        assert any(c["value"] == "change_account" for c in over["choices"])


class TestItDoesNotRefuseWhatIsAllowed:

    def test_a_plan_within_the_limit_runs(self):
        assert view(UNDER).over_limit is None
        assert view(UNDER).can_run is True

    def test_a_taxable_account_has_no_limit_to_exceed(self):
        """An account with no limit entered must not acquire one from the
        table's silence, in either direction."""
        assert view(TAXABLE).over_limit is None

    def test_an_employer_plan_uses_the_deferral_limit_not_the_ira_one(self):
        """$18,000 a year is over an IRA limit and under a 401(k) deferral
        limit. Using the wrong rule would refuse a legal plan."""
        assert view(DEFERRAL).over_limit is None

    def test_an_unknown_cadence_is_not_guessed_at(self):
        from src.workspace.confirmation import _annual_contribution

        class Schedule:
            cadence, amount = "whenever", 1_000.0

        assert _annual_contribution(Schedule()) is None


class TestTheRefusingFigureStatesItsOwnReliability:

    def test_the_figure_is_verified_and_says_so(self):
        over = view(OVER).over_limit
        assert over["limit_is_verified"] is True
        assert over["ruleset_ref"] == "account-rules/us-federal-2026@1"
        assert over["rule"] == "ira_contribution"

    def test_a_verified_figure_still_names_what_it_could_not_check(self):
        """The limit is shared with every other IRA. Being over it from one
        account is certain; the caveat is that the total was never visible."""
        over = view(OVER).over_limit
        assert "shared" in over["caveat"].lower()

    def test_the_caveat_reaches_the_template(self):
        html = _render(view(OVER))
        assert "More than this account allows" in html
        assert "$16,500" in html
        assert "shared" in _block(view(OVER)).lower()

    def test_a_caveat_free_refusal_shows_no_caveat_line(self):
        """Scoped to the refusal block, because the account card carries similar
        copy about the same figure — a different claim, in a different place."""
        rendered = view(OVER)
        assert "shared" in _block(rendered).lower()

        rendered.over_limit = {**rendered.over_limit, "caveat": None}
        assert "shared with" not in _block(rendered).lower()
        assert "More than this account allows" in _block(rendered)


class TestTheTemplateStillDecidesNothing:

    def test_the_block_renders_values_it_did_not_compute(self):
        """The page-level rule: this module computes, the template arranges. A
        limit recalculated in the template is a second implementation."""
        body = open("src/workspace/templates/_confirmation.html").read()
        for token in ("limit", "24500", "7500", "* 12", "sum("):
            assert f"{{{{ {token}" not in body
