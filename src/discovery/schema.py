"""The dimensions of meaning a reader is asked about.

**Wider than the capability manifest, on purpose.** Every value below is
something a person can mean. Several are things this engine refuses to run —
`inverse_volatility`, `risk_parity`, a stated 60/40 — and they are here anyway,
because a reader that cannot say them does not refuse them, it says the nearest
thing it can. That is how "allocate by inverse volatility" became an equal
split with a figure attached.

The rule, enforced by `tests/test_discovery_schema_is_wider.py`:

    schema     what can be meant     -> Discovery
    manifest   what can be run       -> Mission

The schema is allowed to be a superset. It must never be a subset, and the
overlap is never a permission.

Derived from the semantic dimensions this project learned the hard way, in the
order a reader meets them in a sentence.
"""
from __future__ import annotations

from .reader import Dimension, RelationSpec, Schema

QUANTIFY_SCHEMA = Schema(version="quantify-discovery-schema@2", dimensions=(

    Dimension(
        name="objective",
        describes="What the person wants to find out or do.",
        values=("evaluate_investment_strategy", "compare_strategies",
                "plan_contributions", "assess_withdrawal", "other"),
        examples=("evaluate this strategy", "should I convert to a Roth")),

    # ---- when money arrives -----------------------------------------
    Dimension(
        name="cadence",
        describes="How often money is contributed, when it is on a calendar.",
        values=("daily", "weekly", "biweekly", "monthly", "quarterly",
                "annual", "payroll", "once"),
        examples=("every month", "out of each paycheck", "a lump sum")),

    Dimension(
        name="day_rule",
        describes=("Which session within each period the money lands on. "
                   "'Monthly' names no day, and the day changes the result."),
        values=("first_session_of_period", "last_session_of_period",
                "calendar_first_rolled_forward"),
        examples=("last trading day of the month", "on the 1st")),

    Dimension(
        name="amount",
        compare_as="NUMBER",
        describes="How much, per contribution. A number, not a vocabulary."),

    Dimension(
        name="trigger_semantics",
        describes=(
            "When a condition is named, whether the purchase happens on the "
            "day the condition first becomes true, or on every day it stays "
            "true. These produce very different amounts of money and a "
            "sentence rarely says which explicitly."),
        values=("crossing_event", "persistent_condition"),
        examples=("when it crosses below", "whenever it is below",
                  "while it stays under")),

    Dimension(
        name="execution_timing",
        describes="Whether the order fills on the signal's close or the next open.",
        values=("same_session_close", "next_session_open"),
        examples=("at that day's close", "the next morning")),

    Dimension(
        name="conditional_amount",
        describes=("Whether the contributed amount varies with the condition "
                   "that triggered it."),
        examples=("double my contribution after a 20% fall", "step it up")),

    # ---- what it buys -------------------------------------------------
    Dimension(
        name="assets",
        compare_as="SET",
        describes=("What is bought, as the person wrote it. Tickers, fund "
                   "names, or a description like 'a US total market fund'. "
                   "Never resolved to a ticker here — 'SPX ETF' is not SPY, "
                   "and choosing one is a substitution nobody asked for.")),

    Dimension(
        name="observed_assets",
        compare_as="SET",
        describes=("What the condition watches, when it differs from what is "
                   "bought. 'Buy VOO whenever SPY crosses below' watches SPY "
                   "and holds VOO, and they do not cross on the same days.")),

    Dimension(
        name="allocation_method",
        describes="How money is divided between the holdings.",
        # Six of these seven are refused by the current engine. They are here
        # because people say them, and Mission's refusal is more useful than
        # Discovery's silence.
        values=("equal_weight_at_purchase", "stated_weights",
                "inverse_volatility", "risk_parity", "minimum_variance",
                "maximum_diversification", "volatility_target"),
        examples=("by inverse volatility", "risk parity", "equally")),

    Dimension(
        name="stated_weights",
        compare_as="SET",
        describes=("Explicit per-holding weights the person wrote, in the "
                   "order the holdings were named."),
        examples=("60/40", "55/35/10", "70% stocks and 30% bonds")),

    # ---- what happens after ------------------------------------------
    Dimension(
        name="periodic_rebalancing",
        describes=("Whether holdings are brought back to target, and how "
                   "often or on what drift."),
        examples=("rebalance quarterly", "when it drifts more than 5 points")),

    Dimension(
        name="sell_action",
        describes="Any selling, withdrawing, harvesting or converting.",
        examples=("sell the losers", "withdraw 4% a year",
                  "convert to a Roth")),

    Dimension(
        name="dividend_policy",
        describes="Whether dividends are reinvested or taken as cash.",
        values=("reinvested", "held_as_cash")),

    # ---- context ------------------------------------------------------
    Dimension(
        name="account_type",
        describes="Which account the money is in.",
        values=("TAXABLE", "TRADITIONAL_IRA", "ROTH", "TRADITIONAL_401K",
                "ROTH_401K", "HSA", "OTHER")),

    Dimension(
        name="evaluation_period",
        describes=(
            "The window to evaluate over, in a canonical form rather than as "
            "written. Use exactly one of:\n"
            "      trailing:<n>y   or  trailing:<n>m   — a duration back from now\n"
            "      since:<YYYY>    or  since:<YYYY-MM> — open-ended from a date\n"
            "      until:<YYYY>                        — open-ended to a date\n"
            "      range:<YYYY-MM>..<YYYY-MM>          — both ends given\n"
            "      rolling:<n>y                        — many windows, not one\n"
            "    Quote the words in source_span; the value is the canonical form."),
        examples=("'over the past 5 years' -> trailing:5y",
                  "'since January 2020' -> since:2020-01",
                  "'each month for the past five years' -> rolling:5y")),

    Dimension(
        name="moving_average_window",
        compare_as="NUMBER",
        describes="The length of any moving average named, in sessions."),
), relations=(

    # Added in schema@2. Both of these were forced by the shadow run: two
    # sentences where both readers read correctly and the schema made them
    # disagree, because it asked for one value where the sentence had two
    # entities in named roles.
    #
    # The rule for adding more, so this does not become a graph language:
    # a relation is warranted when meaning depends on which value belongs to
    # which participant, or on direction between participants. `60/40` across
    # two named holdings does not qualify — the mapping is unambiguous.

    RelationSpec(
        kind="portfolio_sleeves",
        describes=(
            "Holdings described as parts of a portfolio with different roles, "
            "where an allocation belongs to a particular part. Use this when a "
            "sentence describes a core plus one or more tilts, satellites or "
            "sleeves — not for a plain list of holdings."),
        roles=("core", "satellite"),
        required_roles=("core",),
        repeatable_roles=("satellite",),
        qualifiers={"allocation": "this sleeve's share, as written — '30%'"},
        ordered=False,
        examples=("'a core index fund and tilt 30% satellite into US value "
                  "ETF' -> core='core index fund', "
                  "satellite='US value ETF' allocation='30%'",)),

    RelationSpec(
        kind="account_transition",
        describes=(
            "Money moving between account types. The direction is the "
            "meaning: a conversion out of a traditional IRA into a Roth is not "
            "the same statement with the ends swapped."),
        roles=("from", "to"),
        required_roles=("from", "to"),
        attributes={"action": "convert | transfer | rollover | withdraw"},
        ordered=True,
        examples=("'convert my traditional IRA to a Roth' -> "
                  "from='traditional_ira', to='roth_ira', action='convert'",)),
))
