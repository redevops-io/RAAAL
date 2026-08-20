"""Which `allocation_method` values name a computed research strategy.

Kept deliberately free of any import from `src.strategies`, which pulls the full
research stack (scikit-learn, scipy, the regime ensemble). The pilot asks "is
this weighting a strategy?" on every compile and every coverage pass; those
paths must answer without loading that stack. Only the executor
(`rebalance.strategy_driven`) imports `run_capability`, and only when a strategy
actually runs.

The right-hand side is the capability id `strategies.run_capability` dispatches
on. `test_strategy_methods_match_registry` asserts every target exists in
`strategies.CAPABILITY_BY_ID`, so this table cannot drift from the engine.

`drl_portfolio` is deliberately absent: without a trained checkpoint (gymnasium
is not in the serving image) it falls back to a plain allocation, and offering
it as "deep RL" would name a mechanism the run does not use — the one thing this
whole boundary exists to prevent.
"""
from __future__ import annotations

from typing import Dict, Optional

#: allocation_method value -> the capability id that computes it. The value is
#: the canonical form the catalogue and the schema use; the three trailing
#: synonyms are what Discovery's reader still emits for the same intent.
STRATEGY_ALLOCATION_METHODS: Dict[str, str] = {
    # value == capability id
    "risk_parity": "risk_parity",
    "minimum_variance": "minimum_variance",
    "max_diversification": "max_diversification",
    "equal_risk_contribution": "equal_risk_contribution",
    "volatility_targeting": "volatility_targeting",
    "time_series_momentum": "time_series_momentum",
    "cross_sectional_momentum": "cross_sectional_momentum",
    "dual_momentum": "dual_momentum",
    "regime_momentum": "regime_momentum",
    "relative_value": "relative_value",
    "pairs_trading": "pairs_trading",
    "stat_arb_credit": "stat_arb_credit",
    "reversal": "reversal",
    "ibs_hybrid_switch": "ibs_hybrid_switch",
    "equity_factors": "equity_factors",
    "macro_factors": "macro_factors",
    "multi_factor_blend": "multi_factor_blend",
    "fomo_fobi_overlay": "fomo_fobi_overlay",
    "sharpe_optimizer": "sharpe_optimizer",
    "adaptive_rotation": "adaptive_rotation",
    "raaal_composite": "raaal_composite",
    # One Discovery synonym, kept because a distinct natural phrasing ("by
    # inverse volatility") reads to it. `maximum_diversification` and
    # `volatility_target` are intentionally not here: each would collide with a
    # canonical id (`max_diversification`, `volatility_targeting`), so Discovery
    # never offers them and nothing seals on them.
    "inverse_volatility": "risk_parity",
}


def strategy_capability(weighting: object) -> Optional[str]:
    """The capability id an allocation_method value runs, or None.

    None means the weighting is not a computed strategy —
    `equal_weight_at_purchase` and `stated_weights` stay with the simple
    executor and must not be routed through `run_capability`.
    """
    return STRATEGY_ALLOCATION_METHODS.get(str(weighting or ""))
