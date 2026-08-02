"""What happens to money a sale actually produced.

    reconciled fill -> net proceeds -> policy -> orders -> fills -> residual

The invariant this module exists for:

> Allocation may consume only reconciled net proceeds from an actual fill. It
> may never allocate expected proceeds from an instruction.

An instructed sale has an expected price; a filled one has a price. Allocating
the first spends money that may never arrive, and the resulting portfolio looks
exactly like one that was funded.

**Proceeds are internal cash, not an external flow.** The only external flow was
the delivered in-kind value at vest. A reinvestment that added to the flow
series would count the same compensation twice, and the money-weighted return
would look plausible while being wrong.

**Funding is isolated by default.** A sale producing $10,000 against an
allocation requiring $12,000 must not quietly draw the difference from unrelated
account cash: the diversification plan would appear fully funded when it was
not. `SOURCE_PROCEEDS_ONLY` is the default and `ACCOUNT_CASH_ALLOWED` is a
declaration.

**Nothing renormalizes silently.** If one target of a 60/30/10 allocation cannot
be priced, the remainder does not become 67/33. The unfilled target stays
visible and the money stays as residual cash, because a plan that quietly
reweights is a different plan.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class AllocationStatus(str, Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    PARTIAL = "PARTIAL"
    """Some targets filled and some did not. Distinct from EXECUTED, because
    "all of it happened" and "most of it happened" lead somewhere different."""

    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"


TERMINAL = frozenset({AllocationStatus.EXECUTED, AllocationStatus.PARTIAL,
                      AllocationStatus.FAILED, AllocationStatus.SUPERSEDED})


class FundingScope(str, Enum):
    SOURCE_PROCEEDS_ONLY = "SOURCE_PROCEEDS_ONLY"
    """The narrower default. Orders may spend only this proceeds lot."""

    ACCOUNT_CASH_ALLOWED = "ACCOUNT_CASH_ALLOWED"
    """Declared, never assumed."""


class UnsupportedAllocation(ValueError):
    """A policy this system cannot compile into explicit weights.

    Raised rather than mapped to a plausible portfolio. "Invest conservatively"
    has no typed meaning here, and answering it with 60/40 would put a
    recommendation in the user's account under the description they wrote.
    """


class ProceedsAlreadyAllocated(ValueError):
    """One lot, allocated twice.

    The double-spend this ledger exists to prevent: two instructions against one
    sale invest money that existed once.
    """


class AllocationEventKind(str, Enum):
    SALE_FILL_RECONCILED = "SALE_FILL_RECONCILED"
    PROCEEDS_AVAILABLE = "PROCEEDS_AVAILABLE"
    ALLOCATION_EVALUATED = "ALLOCATION_EVALUATED"
    ALLOCATION_INSTRUCTION_CREATED = "ALLOCATION_INSTRUCTION_CREATED"
    PURCHASE_ORDERS_CREATED = "PURCHASE_ORDERS_CREATED"
    PURCHASE_FILLS_RECONCILED = "PURCHASE_FILLS_RECONCILED"
    RESIDUAL_CASH_RECORDED = "RESIDUAL_CASH_RECORDED"


ALLOCATION_ORDER = (
    AllocationEventKind.SALE_FILL_RECONCILED,
    AllocationEventKind.PROCEEDS_AVAILABLE,
    AllocationEventKind.ALLOCATION_EVALUATED,
    AllocationEventKind.ALLOCATION_INSTRUCTION_CREATED,
    AllocationEventKind.PURCHASE_ORDERS_CREATED,
    AllocationEventKind.PURCHASE_FILLS_RECONCILED,
    AllocationEventKind.RESIDUAL_CASH_RECORDED,
)


@dataclass(frozen=True)
class RealizedProceeds:
    """Money a fill actually produced. Generic: any sale can create one."""

    proceeds_id: str
    source_execution_id: str
    available_on: Any
    gross_proceeds: float
    transaction_costs: float = 0.0
    taxes_withheld_at_sale: float = 0.0
    currency: str = "USD"

    @property
    def net_proceeds(self) -> float:
        return (self.gross_proceeds - self.transaction_costs
                - self.taxes_withheld_at_sale)

    def to_json(self) -> Dict[str, Any]:
        return {"proceeds_id": self.proceeds_id,
                "source_execution_id": self.source_execution_id,
                "available_on": str(self.available_on),
                "gross_proceeds": self.gross_proceeds,
                "transaction_costs": self.transaction_costs,
                "taxes_withheld_at_sale": self.taxes_withheld_at_sale,
                "net_proceeds": self.net_proceeds, "currency": self.currency}


def proceeds_from(execution, *, transaction_costs: Optional[float] = None,
                  taxes_withheld_at_sale: float = 0.0,
                  log=None) -> RealizedProceeds:
    """Turn a reconciled sale into allocatable money.

    Refuses an unreconciled execution. Its `proceeds` is None — unknown, not
    zero — and allocating from an expectation spends money that may never
    arrive.
    """
    if not getattr(execution, "reconciled", False):
        raise ValueError(
            f"execution {execution.instruction_id} has no matching fill, so its "
            "proceeds are unknown. Allocating an expected price would spend "
            "money the sale may never have produced")

    if log is not None:
        log.record(AllocationEventKind.SALE_FILL_RECONCILED,
                   instruction_id=execution.instruction_id)

    lot = RealizedProceeds(
        proceeds_id=f"lot-{uuid.uuid4().hex[:16]}",
        source_execution_id=execution.instruction_id,
        available_on=execution.filled_on,
        gross_proceeds=float(execution.proceeds),
        # Defaults to the cost the sale actually incurred, not to zero. Gross
        # proceeds are not spendable, and sizing purchases against them draws
        # the difference from cash the sale never produced.
        transaction_costs=(getattr(execution, "fill_cost", None) or 0.0
                           if transaction_costs is None else transaction_costs),
        taxes_withheld_at_sale=taxes_withheld_at_sale)

    if log is not None:
        log.record(AllocationEventKind.PROCEEDS_AVAILABLE,
                   proceeds_id=lot.proceeds_id, net=lot.net_proceeds)
    return lot


@dataclass(frozen=True)
class AllocationInstruction:
    """Explicit weights against an explicit budget."""

    instruction_id: str
    source_proceeds_id: str
    policy_ref: str
    weights: Mapping[str, float]
    """Compiled before execution. `EQUAL_WEIGHT` over three assets is stored as
    three explicit thirds, so what executes is what can be read."""

    investable: float
    """Net proceeds less any declared cash reserve. Weights apply to this, not
    to gross — otherwise sale costs silently change the intended basis."""

    cash_reserve: float = 0.0
    funding_scope: FundingScope = FundingScope.SOURCE_PROCEEDS_ONLY
    earliest_execution_date: Any = None
    cost_rate: float = 0.001
    status: AllocationStatus = AllocationStatus.PENDING
    detail: str = ""

    def budgets(self) -> Dict[str, float]:
        """Order notional per target, sized so cost fits inside the lot.

        The engine charges the cost on top of the notional, so spending the
        whole investable amount on notional would need more cash than the sale
        produced — and under `SOURCE_PROCEEDS_ONLY` that shortfall is exactly
        what must not be drawn from elsewhere.
        """
        spendable = self.investable / (1.0 + self.cost_rate)
        return {asset: spendable * weight
                for asset, weight in self.weights.items()}

    def to_json(self) -> Dict[str, Any]:
        return {"instruction_id": self.instruction_id,
                "source_proceeds_id": self.source_proceeds_id,
                "policy_ref": self.policy_ref, "weights": dict(self.weights),
                "investable": self.investable,
                "cash_reserve": self.cash_reserve,
                "funding_scope": self.funding_scope.value,
                "cost_rate": self.cost_rate,
                "status": self.status.value, "detail": self.detail}


@dataclass(frozen=True)
class AllocationExecution:
    """What the allocation actually did, once its fills are reconciled."""

    instruction_id: str
    requested_allocation: Mapping[str, float]
    executed_allocation: Mapping[str, float] = field(default_factory=dict)
    unfilled_targets: Sequence[Dict[str, Any]] = ()
    invested_amount: float = 0.0
    purchase_costs: float = 0.0
    residual_cash: float = 0.0
    investable_base: float = 0.0
    """What the weights were applied to. Kept so realized weights can be
    measured against the same base the request was, rather than against
    whatever happened to be bought."""

    status: AllocationStatus = AllocationStatus.PENDING

    @property
    def realized_weights(self) -> Dict[str, float]:
        """What was actually bought, as a share of the *investable base*.

        Not as a share of what was bought. A 60/30/10 plan whose bond leg did
        not fill is not a 67/33 plan; it is a 60/30 plan with a tenth left over,
        and normalising over the executed total is exactly the silent
        renormalisation this module refuses to do — it would report a portfolio
        the user never asked for as though it were the one they did.
        """
        base = self.investable_base
        if base <= 0:
            return {}
        return {asset: value / base
                for asset, value in self.executed_allocation.items()}

    @property
    def unallocated_weight(self) -> float:
        """The share of the request that did not happen."""
        return max(0.0, 1.0 - sum(self.realized_weights.values()))

    def to_json(self) -> Dict[str, Any]:
        return {"instruction_id": self.instruction_id,
                "requested_allocation": dict(self.requested_allocation),
                "executed_allocation": dict(self.executed_allocation),
                "realized_weights": self.realized_weights,
                "unallocated_weight": self.unallocated_weight,
                "investable_base": self.investable_base,
                "unfilled_targets": [dict(one) for one in self.unfilled_targets],
                "invested_amount": self.invested_amount,
                "purchase_costs": self.purchase_costs,
                "residual_cash": self.residual_cash,
                "status": self.status.value}


# --- policy compilation ----------------------------------------------------

#: Phrasings with no typed meaning. Each would map to a plausible portfolio, and
#: that portfolio would be a recommendation delivered under the user's own
#: words.
UNSUPPORTED_PHRASINGS = (
    "invest conservatively", "best performing", "best-performing",
    "reduce risk", "tax optimal", "tax-optimal", "dynamic",
    "reduce concentration",
)

WEIGHT_TOLERANCE = 1e-9


def compile_policy(policy: Any, *, log=None) -> Dict[str, float]:
    """Turn a stated policy into explicit weights, or refuse.

    Accepts `{"HOLD_CASH"}`, an explicit weight mapping, a list of assets to
    equal-weight, or a `methodology/...@n` reference resolved by its caller.
    """
    if log is not None:
        log.record(AllocationEventKind.ALLOCATION_EVALUATED, policy=str(policy))

    if isinstance(policy, str):
        lowered = policy.lower()
        for phrase in UNSUPPORTED_PHRASINGS:
            if phrase in lowered:
                raise UnsupportedAllocation(
                    f"{policy!r} has no typed meaning here. Answering it with a "
                    "plausible portfolio would put a recommendation in the "
                    "account under the description the user wrote")
        if lowered == "hold_cash":
            return {}
        if policy.startswith("methodology/"):
            raise UnsupportedAllocation(
                f"{policy!r} must be resolved to explicit weights by the "
                "methodology it names before it reaches allocation. The "
                "reference is kept; the rules are not copied here")
        raise UnsupportedAllocation(f"unknown allocation policy {policy!r}")

    if isinstance(policy, Mapping):
        weights = {asset: float(weight) for asset, weight in policy.items()}
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise UnsupportedAllocation(
                f"weights sum to {total:.6f}, not 1. A near-miss silently "
                "rescaled is a different allocation than the one stated")
        return weights

    if isinstance(policy, Sequence):
        assets = list(policy)
        if not assets:
            return {}
        # Compiled to explicit thirds now, so what executes is what can be read
        # — and so the order the assets were named in cannot change the result.
        share = 1.0 / len(assets)
        return {asset: share for asset in sorted(assets)}

    raise UnsupportedAllocation(f"unrecognised allocation policy {policy!r}")


class ProceedsLedger:
    """Which proceeds lots have been spoken for.

    One lot funds one allocation. A second instruction against the same sale
    invests money that existed once, and both would look funded.
    """

    def __init__(self) -> None:
        self._claimed: Dict[str, str] = {}

    def claim(self, proceeds_id: str, instruction_id: str) -> None:
        held = self._claimed.get(proceeds_id)
        if held is not None and held != instruction_id:
            raise ProceedsAlreadyAllocated(
                f"proceeds {proceeds_id} are already allocated by {held}. "
                "Supersede that instruction before allocating them again")
        self._claimed[proceeds_id] = instruction_id

    def release(self, proceeds_id: str) -> None:
        """Freed by an explicit supersession, never by a failure."""
        self._claimed.pop(proceeds_id, None)

    def claimed_by(self, proceeds_id: str) -> Optional[str]:
        return self._claimed.get(proceeds_id)


def instruction_for(proceeds: RealizedProceeds, *, policy: Any,
                    ledger: Optional[ProceedsLedger] = None,
                    cash_reserve: float = 0.0,
                    funding_scope: FundingScope = FundingScope.SOURCE_PROCEEDS_ONLY,
                    cost_rate: float = 0.001,
                    log=None) -> Optional[AllocationInstruction]:
    """Compile a policy against one proceeds lot.

    `HOLD_CASH` returns None: holding is a policy, and it needs no instruction.
    """
    weights = compile_policy(policy, log=log)
    if cash_reserve > proceeds.net_proceeds:
        raise UnsupportedAllocation(
            f"the declared cash reserve ${cash_reserve:,.2f} exceeds the "
            f"${proceeds.net_proceeds:,.2f} this sale produced")

    if not weights:
        return None

    instruction = AllocationInstruction(
        instruction_id=f"alloc-{uuid.uuid4().hex[:16]}",
        source_proceeds_id=proceeds.proceeds_id,
        policy_ref=str(policy), weights=weights,
        investable=proceeds.net_proceeds - cash_reserve,
        cash_reserve=cash_reserve, funding_scope=funding_scope,
        earliest_execution_date=proceeds.available_on, cost_rate=cost_rate)

    if ledger is not None:
        ledger.claim(proceeds.proceeds_id, instruction.instruction_id)
    if log is not None:
        log.record(AllocationEventKind.ALLOCATION_INSTRUCTION_CREATED,
                   instruction_id=instruction.instruction_id,
                   weights=dict(weights))
    return instruction


def supersede(instruction: AllocationInstruction, *, reason: str,
              ledger: Optional[ProceedsLedger] = None
              ) -> AllocationInstruction:
    if instruction.status in TERMINAL:
        raise ValueError(
            f"instruction {instruction.instruction_id} is already "
            f"{instruction.status.value} and cannot be superseded")
    if ledger is not None:
        ledger.release(instruction.source_proceeds_id)
    return replace(instruction, status=AllocationStatus.SUPERSEDED,
                   detail=reason)


class AllocationSchedule:
    """Emits purchase orders for an allocation and reconciles their fills.

    Stateful for the same reason the disposition schedule is: an order that was
    placed and never filled has to remain visible afterwards, and a closure
    deciding per session cannot report that later.
    """

    def __init__(self, instructions: Sequence[AllocationInstruction] = (),
                 *, log=None) -> None:
        self.instructions: List[AllocationInstruction] = list(instructions)
        self.log = log
        self.executions: Dict[str, AllocationExecution] = {}
        self._placed: Dict[str, Dict[str, float]] = {}

    def program(self):
        from ..mission.accounting import Order

        def step(session, visible, holdings, cash):
            orders: List[Order] = []
            for index, instruction in enumerate(self.instructions):
                if instruction.status in TERMINAL \
                        or instruction.instruction_id in self._placed:
                    continue
                if instruction.earliest_execution_date is not None \
                        and session < instruction.earliest_execution_date:
                    # Proceeds are not available yet. An order placed before the
                    # sale filled would be funded by something else.
                    continue

                placed: Dict[str, float] = {}
                for asset, notional in instruction.budgets().items():
                    if asset not in visible.columns or not len(visible):
                        continue
                    price = float(visible.iloc[-1][asset])
                    if price != price or price <= 0:
                        continue
                    orders.append(Order(session, asset, notional,
                                        reason=f"allocate {instruction.policy_ref}"))
                    placed[asset] = notional

                self._placed[instruction.instruction_id] = placed
                self.instructions[index] = instruction
                if self.log is not None:
                    self.log.record(AllocationEventKind.PURCHASE_ORDERS_CREATED,
                                    instruction_id=instruction.instruction_id,
                                    targets=sorted(placed))
            return orders

        return step

    def reconcile(self, fills, *, proceeds: Mapping[str, RealizedProceeds]
                  ) -> Dict[str, AllocationExecution]:
        """Attach what was actually bought, and account for what was not.

        Nothing is renormalised. A target that could not be priced or could not
        fill stays in `unfilled_targets` at its requested weight, and its money
        stays as residual cash.
        """
        purchases = [one for one in fills if one.shares > 0]

        for instruction in self.instructions:
            if instruction.status in TERMINAL:
                continue
            lot = proceeds.get(instruction.source_proceeds_id)
            placed = self._placed.get(instruction.instruction_id, {})

            executed: Dict[str, float] = {}
            costs = 0.0
            for asset in instruction.weights:
                match = next((one for one in purchases if one.ticker == asset),
                             None)
                if match is None:
                    continue
                purchases.remove(match)
                spent = match.shares * match.price
                executed[asset] = spent
                costs += getattr(match, "cost", 0.0)

            unfilled = [
                {"asset": asset, "requested_weight": weight,
                 "why": ("no order was placed — the asset could not be priced"
                         if asset not in placed
                         else "the order was placed and did not fill")}
                for asset, weight in instruction.weights.items()
                if asset not in executed]

            invested = sum(executed.values())
            residual = (lot.net_proceeds if lot else instruction.investable) \
                - invested - costs

            status = (AllocationStatus.EXECUTED if not unfilled
                      else AllocationStatus.PARTIAL if executed
                      else AllocationStatus.FAILED)

            self.executions[instruction.instruction_id] = AllocationExecution(
                instruction_id=instruction.instruction_id,
                requested_allocation=dict(instruction.weights),
                executed_allocation=executed, unfilled_targets=tuple(unfilled),
                invested_amount=invested, purchase_costs=costs,
                residual_cash=residual,
                investable_base=instruction.investable, status=status)

            if self.log is not None:
                self.log.record(AllocationEventKind.PURCHASE_FILLS_RECONCILED,
                                instruction_id=instruction.instruction_id,
                                invested=invested)
                self.log.record(AllocationEventKind.RESIDUAL_CASH_RECORDED,
                                instruction_id=instruction.instruction_id,
                                residual=residual)
        return self.executions


#: Currency tolerance for the conservation identity.
CONSERVATION_TOLERANCE = 1e-6


def conserved(execution: AllocationExecution, proceeds: RealizedProceeds,
              tolerance: float = CONSERVATION_TOLERANCE) -> bool:
    """net proceeds == invested + purchase costs + residual cash."""
    return abs(proceeds.net_proceeds - execution.invested_amount
               - execution.purchase_costs - execution.residual_cash) <= tolerance
