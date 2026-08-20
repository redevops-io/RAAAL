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
    'scheduled-funding': {'amount': '500', 'assets': 'VTI', 'cadence': 'monthly', 'objective': 'plan_contributions'},
    'event-triggered-funding': {'assets': 'VOO', 'moving_average_window': '200', 'objective': 'evaluate_investment_strategy', 'observed_assets': 'SPY', 'trigger_semantics': 'crossing_event'},
    'percentage-of-income': {'account_type': 'TRADITIONAL_IRA', 'cadence': 'annual', 'objective': 'plan_contributions'},
    'percentage-of-paycheck': {'account_type': 'OTHER', 'objective': 'plan_contributions'},
    'weekly-funding': {'amount': '50', 'assets': 'VTI', 'cadence': 'weekly', 'objective': 'plan_contributions'},
    'annual-lump': {'amount': '10000', 'objective': 'plan_contributions'},
    'max-the-limit': {'account_type': 'OTHER', 'objective': 'plan_contributions'},
    'max-several-accounts': {'account_type': 'OTHER', 'cadence': 'annual', 'objective': 'plan_contributions'},
    'employer-match': {'account_type': 'HSA', 'objective': 'plan_contributions'},
    'split-across-accounts': {'amount': '50', 'cadence': 'monthly', 'objective': 'plan_contributions'},
    'escalating-contribution': {'amount': '100', 'cadence': 'monthly', 'conditional_amount': 'raise it to 150 EUR a month next year', 'objective': 'plan_contributions'},
    # --- allocation ------------------------------------------------
    'stated-weights': {'allocation_method': 'stated_weights', 'amount': '500', 'assets': 'VTI,BND', 'cadence': 'monthly', 'stated_weights': 'VTI=60,BND=40'},
    'rebalancing': {'allocation_method': 'stated_weights', 'amount': '500', 'assets': 'VTI,BND', 'cadence': 'monthly', 'stated_weights': 'VTI=60,BND=40'},
    'risk-based-allocation': {'allocation_method': 'inverse_volatility'},
    'factor-tilt': {'allocation_method': 'stated_weights', 'assets': 'small cap value', 'factor_tilt': 'factor_tilt', 'portfolio_sleeves': 'core=my portfolio, satellite=small cap value'},
    'glidepath': {'age_based_allocation': 'age_based_allocation', 'allocation_method': 'stated_weights', 'assets': 'stocks,bonds', 'cadence': 'annual', 'objective': 'plan_contributions'},
    'single-fund': {'amount': '500', 'assets': 'VOO', 'cadence': 'monthly'},
    'three-way-split': {'allocation_method': 'stated_weights', 'amount': '600', 'assets': 'VTI,VXUS,BND', 'cadence': 'monthly', 'stated_weights': 'VTI=60,VXUS=30,BND=10'},
    'mean-variance': {'allocation_method': 'minimum_variance'},
    'volatility-target': {'allocation_method': 'volatility_targeting'},
    'fund-and-rebalance': {'allocation_method': 'stated_weights', 'amount': '600', 'assets': 'VTI,BND', 'cadence': 'quarterly', 'stated_weights': 'VTI=70,BND=30'},
    'holding-period': {'holding_period': '21 days', 'sell_action': 'close it'},
    # --- computed-strategies ---------------------------------------
    'risk-parity-strategy': {'allocation_method': 'risk_parity', 'amount': '500', 'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'periodic_rebalancing': 'quarterly'},
    'minimum-variance-strategy': {'allocation_method': 'minimum_variance', 'amount': '500', 'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'periodic_rebalancing': 'quarterly'},
    'maximum-diversification': {'allocation_method': 'max_diversification', 'amount': '500', 'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'periodic_rebalancing': 'quarterly'},
    'equal-risk-contribution': {'allocation_method': 'equal_risk_contribution', 'amount': '500', 'cadence': 'monthly', 'periodic_rebalancing': 'quarterly'},
    'volatility-targeting': {'allocation_method': 'volatility_targeting', 'amount': '500', 'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'periodic_rebalancing': 'quarterly'},
    'time-series-momentum': {'allocation_method': 'time_series_momentum', 'amount': '500', 'cadence': 'monthly', 'periodic_rebalancing': 'quarterly'},
    'cross-sectional-momentum': {'allocation_method': 'cross_sectional_momentum', 'amount': '500', 'cadence': 'monthly', 'periodic_rebalancing': 'quarterly'},
    'dual-momentum': {'allocation_method': 'dual_momentum', 'amount': '500', 'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'periodic_rebalancing': 'quarterly'},
    'regime-momentum': {'allocation_method': 'regime_momentum', 'amount': '500', 'cadence': 'monthly', 'periodic_rebalancing': 'quarterly'},
    'relative-value': {'allocation_method': 'relative_value', 'amount': '500', 'cadence': 'monthly', 'periodic_rebalancing': 'quarterly'},
    'pairs-trading': {'allocation_method': 'pairs_trading', 'amount': '500', 'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'periodic_rebalancing': 'quarterly'},
    'stat-arb-credit': {'allocation_method': 'stat_arb_credit', 'amount': '500', 'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'periodic_rebalancing': 'quarterly'},
    'reversal': {'allocation_method': 'reversal', 'amount': '500', 'cadence': 'monthly', 'periodic_rebalancing': 'quarterly'},
    'ibs-hybrid-switch': {'allocation_method': 'ibs_hybrid_switch', 'amount': '500', 'cadence': 'monthly', 'periodic_rebalancing': 'quarterly'},
    'equity-factors': {'allocation_method': 'equity_factors', 'amount': '500', 'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'periodic_rebalancing': 'quarterly'},
    'macro-factors': {'allocation_method': 'macro_factors', 'amount': '500', 'cadence': 'monthly', 'periodic_rebalancing': 'quarterly'},
    'multi-factor-blend': {'allocation_method': 'multi_factor_blend', 'amount': '500', 'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'periodic_rebalancing': 'quarterly'},
    'sentiment-overlay': {'allocation_method': 'fomo_fobi_overlay', 'amount': '500', 'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'periodic_rebalancing': 'quarterly'},
    'sharpe-optimiser': {'allocation_method': 'sharpe_optimizer', 'amount': '500', 'cadence': 'monthly', 'periodic_rebalancing': 'quarterly'},
    'adaptive-rotation': {'allocation_method': 'adaptive_rotation', 'amount': '500', 'cadence': 'monthly', 'objective': 'evaluate_investment_strategy', 'periodic_rebalancing': 'quarterly'},
    'raaal-composite': {'allocation_method': 'raaal_composite', 'amount': '500', 'cadence': 'monthly', 'periodic_rebalancing': 'quarterly'},
    # --- money-out -------------------------------------------------
    'safe-withdrawal-rate': {'cadence': 'annual', 'objective': 'assess_withdrawal', 'sell_action': 'withdraw 4% of the portfolio each year, adjusted for inflation'},
    'withdrawal-ordering': {'objective': 'assess_withdrawal'},
    'required-minimum-distribution': {'objective': 'assess_withdrawal', 'sell_action': 'take the required minimum distribution'},
    'annuitisation': {'sell_action': 'annuitize a third of the portfolio at 70'},
    'dividend-income': {'dividend_policy': 'held_as_cash', 'sells_allowed': False},
    'fixed-dollar-withdrawal': {'amount': '20000', 'cadence': 'annual', 'objective': 'assess_withdrawal', 'sell_action': 'withdraw $20,000 from the portfolio each year'},
    'withdraw-to-clear-debt': {'amount': '18000', 'objective': 'assess_debt_repayment', 'sell_action': 'withdraw $18,000 from the portfolio'},
    # --- accounts --------------------------------------------------
    'asset-location': {'asset_location': 'holding=the stocks, account=the taxable account'},
    'roth-conversion': {'amount': '30000', 'cadence': 'annual', 'objective': 'assess_conversion', 'account_transition': 'from=traditional IRA, to=Roth'},
    'tax-loss-harvesting': {'conditional_amount': 'whenever a position falls 10% below its cost basis', 'sell_action': 'harvest losses', 'trigger_semantics': 'persistent_condition'},
    'roth-deferral': {'account_type': 'ROTH_401K', 'objective': 'plan_contributions', 'sell_action': 'Roth rather than pre-tax'},
    'pretax-deferral': {'account_type': 'OTHER', 'objective': 'plan_contributions'},
    # --- other -----------------------------------------------------
    'cash-reserve': {'reserve_policy': 'reserve=cash'},
    'bucket-strategy': {'bucket_policy': 'bucket=cash, bucket=stocks'},
    'leverage': {'assets': 'equity sleeve', 'portfolio_sleeves': 'core=equity sleeve'},
    'option-income': {'cadence': 'monthly', 'objective': 'other', 'sell_action': 'sell covered calls'},
    'non-market-alternative': {'objective': 'assess_debt_repayment'},
    'months-of-expenses': {'reserve_policy': 'reserve=cash'},
    'split-a-lump-sum': {'objective': 'plan_contributions', 'reserve_policy': 'reserve=emergency fund'},
    'payoff-versus-invest': {'amount': '10000', 'objective': 'assess_debt_repayment'},
    'margin-repayment': {'amount': '5000', 'objective': 'assess_debt_repayment', 'sell_action': 'pay down the margin loan'},
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
    # --- money-in --------------------------------------------------
    'scheduled-funding': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
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
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'percentage-of-income': (
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'no reader proposed a value', True),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'UNRESOLVED_DISAGREEMENT', "'10% of my after-tax salary' was stated for amount and cannot be read as a number. Substituting a default here would produce a plan that looks like the one you asked for and is not", True),
    ),
    'percentage-of-paycheck': (
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'no reader proposed a value', True),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'UNRESOLVED_DISAGREEMENT', "'8%' was stated for amount and cannot be read as a number. Substituting a default here would produce a plan that looks like the one you asked for and is not", True),
        ('cadence', 'UNRESOLVED_DISAGREEMENT', "'payroll' was stated for how often money moves and this build cannot place it on a calendar. It runs once, weekly, biweekly, monthly, quarterly, annual", True),
    ),
    'weekly-funding': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'annual-lump': (
        ('cadence', 'UNRESOLVED_DISAGREEMENT', "the words contradict it: the words say 'once' where the reading says 'annual'", True),
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'max-the-limit': (
        ('amount', 'UNRESOLVED_DISAGREEMENT', "the words contradict it: the words say Decimal('18000') where the reading says '22% of my salary'", True),
        ('cadence', 'UNRESOLVED_DISAGREEMENT', 'no reader proposed a value', True),
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'no reader proposed a value', True),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'max-several-accounts': (
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'employer-match': (
        ('amount', 'UNRESOLVED_DISAGREEMENT', "the words contradict it: the words say Decimal('100') where the reading says '400'", True),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'split-across-accounts': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'escalating-contribution': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    # --- allocation ------------------------------------------------
    'stated-weights': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
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
    'rebalancing': (
        ('periodic_rebalancing', 'NOT_ASKED', 'the words carry both readings: periodic_rebalancing | stated_weights', False),
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'risk-based-allocation': (
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'factor-tilt': (
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'no reader proposed a value', True),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'glidepath': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'UNRESOLVED_DISAGREEMENT', "'1%' was stated for amount and cannot be read as a number. Substituting a default here would produce a plan that looks like the one you asked for and is not", True),
    ),
    'single-fund': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'three-way-split': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
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
    'mean-variance': (
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'volatility-target': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'fund-and-rebalance': (
        ('periodic_rebalancing', 'NOT_ASKED', 'the words carry both readings: periodic_rebalancing | stated_weights', False),
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'holding-period': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    # --- computed-strategies ---------------------------------------
    'risk-parity-strategy': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'minimum-variance-strategy': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'maximum-diversification': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'equal-risk-contribution': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'volatility-targeting': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'time-series-momentum': (
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'cross-sectional-momentum': (
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'dual-momentum': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'regime-momentum': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'relative-value': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'pairs-trading': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'stat-arb-credit': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'reversal': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'ibs-hybrid-switch': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'equity-factors': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('conditional_amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'macro-factors': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'multi-factor-blend': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'sentiment-overlay': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'sharpe-optimiser': (
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'adaptive-rotation': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'raaal-composite': (
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    # --- money-out -------------------------------------------------
    'safe-withdrawal-rate': (
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'no reader proposed a value', True),
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'withdrawal-ordering': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'required-minimum-distribution': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'annuitisation': (
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'no reader proposed a value', True),
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'dividend-income': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'fixed-dollar-withdrawal': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'withdraw-to-clear-debt': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    # --- accounts --------------------------------------------------
    'asset-location': (),
    'roth-conversion': (
        ('sell_action', 'UNRESOLVED_DISAGREEMENT', 'syntax found this action stated in the sentence and the reader did not report it — this build only buys, so a sentence that disposes of something must not compile into one that accumulates', True),
    ),
    'tax-loss-harvesting': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'roth-deferral': (
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', 'no reader proposed a value', True),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'UNRESOLVED_DISAGREEMENT', "'100%' was stated for amount and cannot be read as a number. Substituting a default here would produce a plan that looks like the one you asked for and is not", True),
    ),
    'pretax-deferral': (
        ('stated_weights', 'UNRESOLVED_DISAGREEMENT', "the words contradict it: the words say '1' where the reading says '100/0'", True),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    # --- other -----------------------------------------------------
    'cash-reserve': (),
    'bucket-strategy': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'leverage': (
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
        ('moving_average_window', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('objective', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('observed_assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('selection_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('trigger_semantics', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'option-income': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('amount', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('execution_timing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('holding_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'non-market-alternative': (),
    'months-of-expenses': (),
    'split-a-lump-sum': (
        ('amount', 'UNRESOLVED_DISAGREEMENT', "the words contradict it: the words say Decimal('10000') where the reading says '20000'", True),
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('dividend_policy', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('periodic_rebalancing', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('stated_weights', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'payoff-versus-invest': (
        ('account_type', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('allocation_method', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('assets', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('cadence', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('day_rule', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('evaluation_period', 'NOT_ASKED', 'the reader was asked and did not answer', False),
        ('sell_action', 'NOT_ASKED', 'the reader was asked and did not answer', False),
    ),
    'margin-repayment': (),
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
