"""What each offered strategy states, as values rather than as a sentence.

A dropdown choice is structured evidence. The product knows which entry it
offered; sending that entry's sentence to a language model and asking what it
means throws that away and re-derives it, less reliably, on every request.

    CatalogSelection + CatalogDefaults + UserEdits -> VerifiedIntent

**Frozen, not inferred.** This table was produced once, from the recorded
reading of each entry's own sentence, and then committed. That is the whole
point: the reading happened at authoring time where it can be reviewed in a
diff, instead of at request time where nobody sees it. Changing what a
catalogue entry means is now an edit to this file.

**It is checked against the sentence, not trusted.**
`tests/test_catalog_structured_path.py` compiles every entry both ways — through
this table with no reader at all, and through the reader from the prose — and
requires the same execution identity. A row that drifts from the sentence it
describes fails there rather than quietly offering somebody a different
strategy than the one they read.

Values are canonical, in the form `discovery.canonical` produces: a cadence is
one of six names, an amount is a plain decimal, holdings are comma-separated.
They are re-canonicalised on the way in anyway, so a row written by hand in
ordinary words still works — it is simply not what the generator wrote.

An entry absent from this table has no structured evidence and falls back to
reading its sentence. `test_every_offered_strategy_has_structured_evidence`
fails on that rather than letting the fallback be silent, because a fallback
nobody measures is how half a feature ships.
"""
from __future__ import annotations

from typing import Mapping

#: entry key -> the dimensions that entry states, canonical.
STATES = {
    # --- money-in --------------------------------------------------
    'scheduled-funding': {'amount': '500', 'assets': 'VTI', 'cadence': 'monthly'},
    'event-triggered-funding': {'assets': 'VOO', 'moving_average_window': '200', 'observed_assets': 'SPY', 'trigger_semantics': 'crossing_event'},
    'percentage-of-income': {'account_type': 'TRADITIONAL_IRA', 'cadence': 'annual', 'objective': 'plan_contributions'},
    'percentage-of-paycheck': {'account_type': 'OTHER', 'objective': 'plan_contributions'},
    'weekly-funding': {'amount': '50', 'assets': 'VTI', 'cadence': 'weekly', 'objective': 'plan_contributions'},
    'annual-lump': {'amount': '10000'},
    'max-the-limit': {'account_type': 'OTHER', 'objective': 'plan_contributions'},
    'max-several-accounts': {'account_type': 'OTHER', 'cadence': 'annual', 'objective': 'plan_contributions'},
    'employer-match': {'account_type': 'HSA', 'objective': 'plan_contributions'},
    'split-across-accounts': {'amount': '50', 'cadence': 'monthly', 'objective': 'plan_contributions'},
    'escalating-contribution': {'cadence': 'monthly', 'conditional_amount': 'raise it to 150 EUR a month next year', 'objective': 'plan_contributions'},
    # --- allocation ------------------------------------------------
    'stated-weights': {'amount': '500', 'assets': 'VTI,BND', 'cadence': 'monthly', 'allocation_method': 'stated_weights', 'stated_weights': 'VTI=60,BND=40'},
    'rebalancing': {'amount': '500', 'assets': 'VTI,BND', 'cadence': 'monthly', 'allocation_method': 'stated_weights', 'stated_weights': 'VTI=60,BND=40', 'periodic_rebalancing': 'annual'},
    'risk-based-allocation': {'allocation_method': 'inverse_volatility'},
    'factor-tilt': {'objective': 'other', 'portfolio_sleeves': 'core=my portfolio, satellite=small cap value'},
    'glidepath': {'assets': 'bonds', 'cadence': 'annual', 'objective': 'plan_contributions', 'sell_action': 'shift 1% from stocks to bonds'},
    # A single fund is one holding, so it needs no split: the whole
    # contribution buys VOO. Named as a ticker the snapshot carries rather than
    # "an S&P 500 index fund", which the resolver refuses as a family.
    'single-fund': {'amount': '500', 'assets': 'VOO', 'cadence': 'monthly'},
    'three-way-split': {'amount': '600', 'assets': 'VTI,VXUS,BND', 'cadence': 'monthly', 'allocation_method': 'stated_weights', 'stated_weights': 'VTI=60,VXUS=30,BND=10'},
    'mean-variance': {'allocation_method': 'minimum_variance', 'assets': '30 holdings'},
    'volatility-target': {'allocation_method': 'volatility_target'},
    'fund-and-rebalance': {'amount': '600', 'assets': 'VTI,BND', 'cadence': 'quarterly', 'allocation_method': 'stated_weights', 'stated_weights': 'VTI=70,BND=30', 'periodic_rebalancing': 'quarterly', 'objective': 'evaluate_investment_strategy'},
    'holding-period': {'holding_period': '21 days', 'sell_action': 'close it'},
    # --- money-out -------------------------------------------------
    'safe-withdrawal-rate': {'objective': 'assess_withdrawal', 'sell_action': 'withdraw 4% of the portfolio each year, adjusted for inflation'},
    'withdrawal-ordering': {'objective': 'assess_withdrawal'},
    'required-minimum-distribution': {'objective': 'assess_withdrawal', 'sell_action': 'take the required minimum distribution'},
    'annuitisation': {'sell_action': 'annuitize a third of the portfolio at 70'},
    'dividend-income': {'dividend_policy': 'held_as_cash', 'sells_allowed': False},
    'fixed-dollar-withdrawal': {'amount': '20000', 'cadence': 'annual', 'objective': 'assess_withdrawal', 'sell_action': 'withdraw $20,000 from the portfolio each year'},
    'withdraw-to-clear-debt': {'amount': '18000', 'objective': 'assess_debt_repayment', 'sell_action': 'withdraw $18,000 from the portfolio'},
    # --- accounts --------------------------------------------------
    'asset-location': {'asset_location': 'holding=stocks, account=taxable account'},
    'roth-conversion': {'account_transition': 'from=traditional IRA, to=Roth', 'amount': '30000', 'cadence': 'annual', 'objective': 'assess_conversion', 'sell_action': 'convert'},
    'tax-loss-harvesting': {'conditional_amount': 'a position falls 10% below its cost basis', 'sell_action': 'harvest losses', 'trigger_semantics': 'persistent_condition'},
    'roth-deferral': {'account_type': 'OTHER', 'objective': 'plan_contributions'},
    'pretax-deferral': {'account_type': 'OTHER', 'objective': 'plan_contributions'},
    # --- other -----------------------------------------------------
    'cash-reserve': {'reserve_policy': 'reserve=cash'},
    'bucket-strategy': {'bucket_policy': 'bucket=cash, bucket=stocks'},
    'leverage': {},
    'option-income': {'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'sell_action': 'sell covered calls'},
    'non-market-alternative': {'objective': 'assess_debt_repayment', 'sell_action': 'pay off the mortgage'},
    'months-of-expenses': {'reserve_policy': 'reserve=cash'},
    'split-a-lump-sum': {'objective': 'plan_contributions', 'reserve_policy': 'reserve=emergency fund'},
    'payoff-versus-invest': {'amount': '10000', 'objective': 'assess_debt_repayment'},
    'margin-repayment': {'amount': '5000', 'objective': 'assess_debt_repayment'},
}


#: What each entry leaves open, frozen the same way and for the same reason.
#:
#: The first version of this file carried only what an entry *states*, so the
#: structured path sealed every selection outright — and a strategy whose
#: sentence names no holding went straight to "the intent names nothing to
#: hold" instead of asking what to hold. The prose path leaves those questions
#: open, the page asks them, and the family assumptions answer most of them.
#:
#: So what an entry does not say is part of what it means, and it is evidence
#: too. `(dimension, reason, detail, result_changing)`, in the shape
#: `Unresolved` takes.
OPEN = {
    'scheduled-funding': (
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'event-triggered-funding': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'percentage-of-income': (
        ('amount', 'UNRESOLVED_DISAGREEMENT', "'10%' was stated for amount and cannot be read as a number. Substituting a default here would produce a plan that looks like the one you asked for and is not", True),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'only syntax proposed a value, and syntax is evidence rather than authority', True),
    ),
    'percentage-of-paycheck': (
        ('amount', 'UNRESOLVED_DISAGREEMENT', "'8%' was stated for amount and cannot be read as a number. Substituting a default here would produce a plan that looks like the one you asked for and is not", True),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'UNRESOLVED_DISAGREEMENT', "'payroll' was stated for how often money moves and this build cannot place it on a calendar. It runs once, weekly, biweekly, monthly, quarterly, annual", True),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'only syntax proposed a value, and syntax is evidence rather than authority', True),
    ),
    'weekly-funding': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'annual-lump': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'UNRESOLVED_DISAGREEMENT', "the model read 'annual'; syntax argues otherwise ('once' scored +1). Not resolved by score: a parser can be confident and wrong", True),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'max-the-limit': (
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'UNRESOLVED_DISAGREEMENT', "the model read '22% of my salary'; syntax argues otherwise (Decimal('18000') scored +1). Not resolved by score: a parser can be confident and wrong", True),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'UNRESOLVED_DISAGREEMENT', 'only syntax proposed a value, and syntax is evidence rather than authority', True),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'only syntax proposed a value, and syntax is evidence rather than authority', True),
    ),
    'max-several-accounts': (
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'employer-match': (
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'UNRESOLVED_DISAGREEMENT', "the model read '400'; syntax argues otherwise (Decimal('100') scored +1). Not resolved by score: a parser can be confident and wrong", True),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'split-across-accounts': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'escalating-contribution': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'UNRESOLVED_DISAGREEMENT', "the model read '150'; syntax argues otherwise (Decimal('100') scored +1). Not resolved by score: a parser can be confident and wrong", True),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'stated-weights': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'rebalancing': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', "'rebalance' is used for both readings in attested writing (rebalance back to target | change the target allocation); a better parse cannot settle what the words do not", False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'risk-based-allocation': (
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'factor-tilt': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'only syntax proposed a value, and syntax is evidence rather than authority', True),
    ),
    'glidepath': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'UNRESOLVED_DISAGREEMENT', "'1%' was stated for amount and cannot be read as a number. Substituting a default here would produce a plan that looks like the one you asked for and is not", True),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'single-fund': (
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'three-way-split': (
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'mean-variance': (
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'volatility-target': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'fund-and-rebalance': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', "'rebalance' is used for both readings in attested writing (rebalance back to target | change the target allocation); a better parse cannot settle what the words do not", False),
    ),
    'holding-period': (
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'safe-withdrawal-rate': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'only syntax proposed a value, and syntax is evidence rather than authority', True),
    ),
    'required-minimum-distribution': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'annuitisation': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'only syntax proposed a value, and syntax is evidence rather than authority', True),
    ),
    'dividend-income': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'fixed-dollar-withdrawal': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'withdraw-to-clear-debt': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'tax-loss-harvesting': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'roth-deferral': (
        ('amount', 'UNRESOLVED_DISAGREEMENT', "'100%' was stated for amount and cannot be read as a number. Substituting a default here would produce a plan that looks like the one you asked for and is not", True),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'only syntax proposed a value, and syntax is evidence rather than authority', True),
    ),
    'pretax-deferral': (
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', "the model read '100/0'; syntax argues otherwise (Decimal('1') scored +1). Not resolved by score: a parser can be confident and wrong", True),
    ),
    'bucket-strategy': (
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'leverage': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'option-income': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'non-market-alternative': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'split-a-lump-sum': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'UNRESOLVED_DISAGREEMENT', "the model read '20000'; syntax argues otherwise (Decimal('10000') scored +1). Not resolved by score: a parser can be confident and wrong", True),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'payoff-versus-invest': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'margin-repayment': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
}


def states(entry_key: str) -> Mapping[str, str]:
    """What this entry states, or nothing when it is not described here."""
    return STATES.get(entry_key, {})


def unresolved(entry_key: str):
    """What this entry leaves open, as `Unresolved` values."""
    from runtime_contracts import OpenReason, Unresolved

    return tuple(
        Unresolved(dimension=dimension, reason=OpenReason(reason),
                   detail=detail, result_changing=material)
        for dimension, reason, detail, material in OPEN.get(entry_key, ()))
