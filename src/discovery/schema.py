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

# @6: `selection_rule` and `holding_period`. The benchmark found "each month
# hold whichever of VTI and BND performed best" reading as two holdings and a
# monthly cadence, and *executing* — the selection, and the selling a rotation
# implies, were gone. Adding `rotate` and `whichever` to the syntax guard's
# lemma list would have been treating the witnesses as the semantic; the
# missing thing was a concept for Discovery to put the selection in.
#
# `holding_period` is the same shape, found by the same pair: "hold VTI for 200
# days" compiled identically to "buy VTI below its 200-day moving average",
# because neither the duration nor the lookback had anywhere to go.
#
# @5: `asset_location`. The last schema gap the strategy sweep left standing.
# `account_type` *was* read from "hold the bonds in the IRA and the stocks in
# the taxable account" — it returned TAXABLE — so the family scored as
# understood while the mapping, which is the whole request, was gone. A
# single-valued dimension cannot carry a mapping.
#
# @4: `reserve_policy` and `bucket_policy` relations, and a
# `leverage_multiplier` qualifier on `portfolio_sleeves`. The live drift lane
# found the same three families both silently reduced *and* execution-unstable,
# which is what a representational gap looks like: when the schema cannot state
# the thing, which fragment survives is decided by the draw, and one sentence
# yields several executable plans. None of the three is executable and all are
# refused by name in the manifest.
#
# @3: `objective` gained `assess_conversion` and `assess_debt_repayment`.
# Bumped because the *content* changed — a fingerprint that moves under an
# unchanged version makes two runs look comparable when they are not, which is
# the same rule `READER_VERSION` and `quantify-compiler@2` already follow.
# The shadow matrices were built under @2 and are stale; `corpus/shadow/STALE.md`
# says so and `test_phase3_exit_gate` checks that the declaration is current.
QUANTIFY_SCHEMA = Schema(version="quantify-discovery-schema@7", dimensions=(

    Dimension(
        name="objective",
        describes="What the person wants to find out or do.",
        values=("evaluate_investment_strategy", "compare_strategies",
                "plan_contributions", "assess_withdrawal",
                # Added after the strategy-family sweep. The examples below
                # already promised "should I convert to a Roth" while the
                # vocabulary had no value for it, so the model answered `other`
                # — correctly, and uselessly, because `other` is what a reader
                # says when a sentence names no objective at all. Mission then
                # executed it as an ordinary contribution plan.
                #
                # Neither is executable and both are here anyway, which is the
                # rule this whole file follows: the schema states what can be
                # meant and the manifest decides what can be run.
                "assess_conversion", "assess_debt_repayment",
                "other"),
        examples=("evaluate this strategy", "should I convert to a Roth",
                  "pay off the mortgage instead of investing")),

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
        name="selection_rule",
        describes=(
            "Choosing which holdings to own from a candidate set, "
            "periodically, by ranking them. Momentum, relative strength, "
            "'whichever performed best', 'the stronger of'. The candidates are "
            "in `assets`; this is the rule that picks among them."),
        examples=("hold whichever of VTI and BND performed best each month",
                  "rotate monthly into the stronger of the two",
                  "buy the top two by trailing 12-month return")),

    Dimension(
        name="holding_period",
        describes=(
            "A stated length of time a position is kept, as written. Distinct "
            "from `moving_average_window`, which is a lookback the market is "
            "measured over — one is how long you hold, the other is how far "
            "back you look."),
        examples=("hold for 200 days", "keep it for at least a year",
                  "hold the bonus shares for 90 days")),

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

    # --- families this build does not model ---------------------------------
    #
    # Recognised so they can be *refused by name*, which is the only reason
    # they exist. Nothing in Mission consumes either, so a sealed intent
    # carrying one strands and is refused as UNSUPPORTED_DIMENSION naming the
    # family — the mechanism that already refuses `observed_assets` and
    # `execution_timing`.
    #
    # They are dimensions rather than `allocation_method` values on purpose.
    # An unsupported value of a supported dimension refuses by that dimension,
    # so "we do not model factor tilts" and "we cannot compute risk parity"
    # would arrive as the same refusal identity — and the drift lane, which
    # identifies an outcome by the dimensions refused, could not tell them
    # apart. `age_based_allocation` is also not an allocation *method*: it is
    # an allocation that changes over time, and calling it one would describe
    # it as static.

    Dimension(
        name="factor_tilt",
        asked=False,
        describes=("A tilt toward a factor or style — value, size, quality, "
                   "momentum — rather than named holdings."),
        examples=("tilt 20% toward small cap value", "overweight value",
                  "add a quality tilt")),

    Dimension(
        name="age_based_allocation",
        asked=False,
        describes=("An allocation that changes with age or over time, rather "
                   "than one held for the whole evaluation."),
        examples=("hold my age in bonds", "increase bonds as I get older",
                  "reduce equity exposure over time")),
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
        qualifiers={
            "allocation": "this sleeve's share, as written — '30%'",
            # A multiplier belongs to one sleeve, which is the whole reason it
            # is a qualifier and not a dimension. "hold 2x leverage on the
            # equity sleeve" leveraged the equities and nothing else; a scalar
            # `leverage` field would say the portfolio was levered and lose
            # which part.
            "leverage_multiplier": (
                "gearing applied to this sleeve, as a number — '2' for 2x. "
                "Absent means unlevered; it is never assumed to be 1 for a "
                "sleeve the sentence did not describe that way"),
        },
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

    # The three families the live drift lane found were both silently reduced
    # *and* execution-unstable. That pairing is the evidence they are
    # representational rather than a tuning problem: when the schema cannot
    # state the thing, which fragment survives is decided by the draw, and one
    # sentence yields several executable plans.
    #
    # None of these is executable. They are here so Discovery can state them
    # faithfully enough for Mission to refuse them by name, which is the whole
    # rule this file follows.

    RelationSpec(
        kind="asset_location",
        describes=(
            "Which holding sits in which account. The mapping is the meaning: "
            "'bonds in the IRA and stocks in the taxable account' is not a "
            "list of two holdings beside a list of two accounts, and reading "
            "it that way loses the only thing the sentence was about. "
            "Requires both roles — naming an account you own is not placing "
            "anything in it."),
        roles=("holding", "account"),
        required_roles=("holding", "account"),
        repeatable_roles=("holding", "account"),
        ordered=False,
        examples=("'keep the REITs in the Roth' -> holding='REITs', "
                  "account='roth'",
                  "'hold the bonds in the IRA and the stocks in the taxable "
                  "account' -> two pairs, bonds->traditional_ira and "
                  "stocks->taxable")),

    RelationSpec(
        kind="reserve_policy",
        describes=(
            "Money deliberately held back from investment, sized against "
            "something other than a market value — months or years of "
            "expenses, or a stated sum kept in cash. The meaning is in what "
            "the reserve is measured against, so a bare 'keep some cash' is "
            "not this relation."),
        roles=("reserve",),
        required_roles=("reserve",),
        qualifiers={
            "amount_basis": "what the size is measured in — 'expenses', "
                            "'salary', or a currency amount",
            "duration": "how much of that basis — 'three years', 'six months'",
            "asset_role": "what the reserve is held in — 'cash'",
            "precedence": "whether it is funded before investing — "
                          "'before_investing' | 'alongside' | 'unstated'",
        },
        ordered=False,
        examples=("'keep six months of expenses in cash before investing "
                  "anything' -> reserve='cash', amount_basis='expenses', "
                  "duration='six months', precedence='before_investing'",)),

    RelationSpec(
        kind="bucket_policy",
        describes=(
            "Money split into pots by time horizon, each holding different "
            "assets, with rules about which pot is spent from and how it is "
            "refilled. The meaning is the mapping between horizon and "
            "holding, so this cannot be a list of assets and a list of "
            "durations side by side."),
        roles=("bucket",),
        required_roles=("bucket",),
        repeatable_roles=("bucket",),
        qualifiers={
            "horizon": "the period this bucket covers — 'three years'",
            "holding": "what this bucket is held in — 'cash', 'stocks'",
            "refilled_from": "which bucket tops this one up, if the sentence "
                             "says",
        },
        ordered=True,
        examples=("'keep three years of expenses in cash and the rest in "
                  "stocks' -> bucket[0] horizon='three years' holding='cash', "
                  "bucket[1] horizon='remainder' holding='stocks'",)),
))
