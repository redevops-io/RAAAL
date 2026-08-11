"""Runtimes a Mission executes inside, all sharing one lifecycle."""
from .account import AccountKind, AccountRuntime, TAXABLE_BROKERAGE
from .account import IMPLEMENTED as ACCOUNT_IMPLEMENTED
from .base import (
    REGISTERED_RULES,
    CompositionConflict,
    CompositionRuleSpec,
    Exclusion,
    ExecutionEnvironment,
    MissingRuntime,
    RuntimeArtifact,
    RuntimeAssumption,
    RuntimeLimitation,
    RuleCategory,
    Severity,
    canonical_hash,
    composition_rule,
)
from .cash_flow import SALARY_AND_VESTS, CashFlowRuntime, DayRule, FlowKind
from .cash_flow import IMPLEMENTED as FLOW_IMPLEMENTED
from .market_data import (
    YFINANCE_DAILY,
    AdjustmentPolicy,
    MarketDataRuntime,
    PointInTimePolicy,
    RealizedData,
    UniversePolicy,
)
from .market_data import IMPLEMENTED as MARKET_DATA_IMPLEMENTED
from .tax import PRE_TAX, TAX_DEFERRED, US_FEDERAL_WITHHOLDING, LotMethod, TaxRuntime
from .tax import IMPLEMENTED as TAX_IMPLEMENTED

#: Runtime kind -> type. Lets the comparison registry derive causal
#: dependencies from each runtime's own declarations instead of restating them.
def _runtime_types():
    """Built lazily: `TradingCalendar` lives in `src/calendars/` and importing it
    at module load would make the runtime package depend on pandas."""
    from ..calendars.calendar import TradingCalendar

    return {cls.kind: cls for cls in (
        TaxRuntime, AccountRuntime, MarketDataRuntime, CashFlowRuntime,
        TradingCalendar,
    )}


class _RuntimeTypes(dict):
    """A dict that fills itself on first access."""

    def _ensure(self):
        if not self:
            self.update(_runtime_types())

    def get(self, key, default=None):
        self._ensure()
        return dict.get(self, key, default)

    def __contains__(self, key):
        self._ensure()
        return dict.__contains__(self, key)

    def __getitem__(self, key):
        self._ensure()
        return dict.__getitem__(self, key)

    def keys(self):
        self._ensure()
        return dict.keys(self)


#: Runtime kind -> type. Lets the comparison registry derive causal
#: dependencies from each runtime's own declarations instead of restating them.
RUNTIME_TYPES = _RuntimeTypes()

__all__ = [
    "RUNTIME_TYPES",
    "ACCOUNT_IMPLEMENTED", "AdjustmentPolicy", "CompositionConflict",
    "Exclusion", "MARKET_DATA_IMPLEMENTED", "MarketDataRuntime",
    "PointInTimePolicy", "RealizedData", "Severity", "UniversePolicy",
    "YFINANCE_DAILY", "composition_rule", "REGISTERED_RULES",
    "CompositionRuleSpec", "RuleCategory", "CashFlowRuntime", "DayRule",
    "FLOW_IMPLEMENTED", "FlowKind", "SALARY_AND_VESTS", "AccountKind", "AccountRuntime", "ExecutionEnvironment",
    "LotMethod", "MissingRuntime", "PRE_TAX", "RuntimeArtifact",
    "RuntimeAssumption", "RuntimeLimitation", "TAXABLE_BROKERAGE",
    "TAX_DEFERRED", "TAX_IMPLEMENTED", "TaxRuntime", "US_FEDERAL_WITHHOLDING",
    "canonical_hash",
]
