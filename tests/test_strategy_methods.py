"""The strategy-method table cannot drift from the engine that runs it.

`mission.strategy_methods` names which `allocation_method` values route to a
computed research strategy, and the capability id each one runs. It is kept free
of any import from `src.strategies` so the compile and coverage paths can ask
"is this a strategy?" without loading the research stack — which means nothing
in that module checks its right-hand side against the real registry. This does.
"""
from __future__ import annotations

from src.mission.strategy_methods import (
    STRATEGY_ALLOCATION_METHODS,
    strategy_capability,
)


def test_every_target_is_a_real_capability():
    """A value maps to a capability id the engine can actually dispatch."""
    from src.strategies import CAPABILITY_BY_ID

    unknown = {value: capability
               for value, capability in STRATEGY_ALLOCATION_METHODS.items()
               if capability not in CAPABILITY_BY_ID}
    assert unknown == {}, (
        f"these allocation methods route to capability ids the registry does "
        f"not define: {unknown}")


def test_drl_is_not_offered():
    """`drl_portfolio` falls back to a plain allocation without a trained model
    (gymnasium is absent in the serving image), so offering it would name a
    mechanism the run does not use."""
    assert "drl_portfolio" not in STRATEGY_ALLOCATION_METHODS.values()


def test_the_schema_can_say_every_canonical_method():
    """A method the table states as canonical must be sayable, or the catalogue
    could seal a value Discovery would reject on the prose path."""
    from src.discovery.schema import QUANTIFY_SCHEMA

    allocation = next(d for d in QUANTIFY_SCHEMA.dimensions
                      if d.name == "allocation_method")
    sayable = set(allocation.values)
    missing = sorted(set(STRATEGY_ALLOCATION_METHODS) - sayable)
    assert missing == [], (
        f"these strategy methods are not in the schema's allocation_method "
        f"values, so a catalogue entry sealing one has no prose it can match: "
        f"{missing}")


def test_equal_weight_and_stated_weights_are_not_strategies():
    """The two simple executors must stay off the strategy path: routing them
    through `run_capability` would compute a split for a plan that stated one."""
    assert strategy_capability("equal_weight_at_purchase") is None
    assert strategy_capability("stated_weights") is None
    assert strategy_capability("") is None
    assert strategy_capability(None) is None
