"""What happens to delivered shares, as a lifecycle rather than a side effect.

    vest -> evaluate policy -> instruction -> defer while ineligible
                                           -> execute on the first eligible
                                              session

The invariant this module exists for:

> A disposition instruction survives until it executes, expires by declared
> policy, or is explicitly superseded. It is never dropped because the vest date
> was ineligible.

A sale silently discarded because the vest landed inside a blackout converts a
diversification plan into a hold — a different strategy, with a different
result, that the user never chose. The failure is invisible precisely because
the portfolio still looks reasonable afterwards.

**A sale is not an external flow.** The only external flow is the delivered
in-kind value at vest. Counting the proceeds again would credit the same
compensation twice in the money-weighted return, and the number would look
plausible.

Kept separate from `VestEvent` on purpose. A vest is a fact about compensation;
a disposition is a decision about it. One can be certain while the other is
still pending, and merging them makes the pending one look settled.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class DispositionStatus(str, Enum):
    PENDING = "PENDING"
    """Created, not yet eligible. Still owed."""

    ELIGIBLE = "ELIGIBLE"
    EXECUTED = "EXECUTED"
    EXPIRED = "EXPIRED"
    """Reached its declared expiry without executing. A stated outcome, not a
    disappearance."""

    SUPERSEDED = "SUPERSEDED"
    """Replaced by a later approved policy. Both survive in the history."""

    FAILED = "FAILED"
    """Could not execute, and why. Retained so the gap is inspectable rather
    than showing up as shares that simply never sold."""


#: Statuses that are settled. A settled instruction's status is never
#: recomputed from dates: an executed sale does not become PENDING again
#: because a later blackout window covers its date.
TERMINAL = frozenset({DispositionStatus.EXECUTED, DispositionStatus.EXPIRED,
                      DispositionStatus.SUPERSEDED, DispositionStatus.FAILED})


class UnsupportedPolicy(ValueError):
    """A disposition this system cannot size or schedule.

    Raised rather than approximated. "Sell enough to get employer stock under
    20%" is not "sell half", and delivering the second while the user asked for
    the first is a different portfolio with no visible sign of substitution.
    """


@dataclass(frozen=True)
class DispositionInstruction:
    """One pending sale, with everything needed to decide when it may happen."""

    instruction_id: str
    grant_ref: str
    created_from_vest: str
    asset: str
    quantity: float
    policy: str

    earliest_eligible_date: Any = None
    """Never earlier than the delivery that created it. Execution before
    delivery would sell shares the account does not hold."""

    blackout_ref: Sequence[tuple] = ()
    execution_lag: int = 1
    expires_at: Any = None
    status: DispositionStatus = DispositionStatus.PENDING
    detail: str = ""

    sizing_policy: Any = None
    """A `ConcentrationPolicy` when the quantity is solved from portfolio state.

    Present, `quantity` is not known at creation and must not be. Both the
    employer price and the rest of the portfolio move between the vest and the
    first eligible session, so a quantity fixed at vest solves yesterday's
    problem."""

    employer_asset: str = ""
    sized_at: Any = None
    sizing_plan: Any = None

    @property
    def sizes_from_portfolio(self) -> bool:
        return self.sizing_policy is not None

    def to_json(self) -> Dict[str, Any]:
        return {"instruction_id": self.instruction_id,
                "grant_ref": self.grant_ref,
                "created_from_vest": self.created_from_vest,
                "asset": self.asset, "quantity": self.quantity,
                "policy": self.policy,
                "earliest_eligible_date": (str(self.earliest_eligible_date)
                                           if self.earliest_eligible_date
                                           is not None else None),
                "blackout_ref": [list(w) for w in self.blackout_ref],
                "execution_lag": self.execution_lag,
                "expires_at": (str(self.expires_at)
                               if self.expires_at is not None else None),
                "status": self.status.value, "detail": self.detail}


@dataclass(frozen=True)
class DispositionEligibility:
    """Whether an instruction may execute on a session, and if not, why not."""

    eligible: bool
    session: Any = None
    blocked_by: Sequence[str] = ()

    def to_json(self) -> Dict[str, Any]:
        return {"eligible": self.eligible,
                "session": str(self.session) if self.session is not None else None,
                "blocked_by": list(self.blocked_by)}


@dataclass(frozen=True)
class DispositionExecution:
    """A sale that was instructed, and — once reconciled — what it actually did.

    The instruction session and the fill session are different facts. The engine
    applies the declared execution lag, so an order placed on the first eligible
    session fills on a later one at a price nobody knew when deciding. Recording
    the expected price as though it were the fill would state an outcome from
    the decision that produced it, which is the whole reason execution lag
    exists.
    """

    instruction_id: str
    instructed_on: Any
    shares: float
    expected_price: float

    filled_on: Any = None
    fill_price: Optional[float] = None
    proceeds: Optional[float] = None
    """Gross proceeds — shares times fill price, before the sale's own cost.
    None until reconciled against the engine's fills. Not zero: an unreconciled
    sale has unknown proceeds, and zero is a number."""

    fill_cost: Optional[float] = None
    """The transaction cost the sale itself incurred.

    Carried because gross proceeds are not spendable. Allocating the gross
    figure sizes purchases against money the sale did not produce, and the
    shortfall then comes from somewhere else — which is the funding leak the
    narrow scope exists to prevent."""

    @property
    def net_proceeds(self) -> Optional[float]:
        if self.proceeds is None:
            return None
        return self.proceeds - (self.fill_cost or 0.0)

    @property
    def reconciled(self) -> bool:
        return self.filled_on is not None

    def to_json(self) -> Dict[str, Any]:
        return {"instruction_id": self.instruction_id,
                "instructed_on": str(self.instructed_on),
                "shares": self.shares, "expected_price": self.expected_price,
                "filled_on": (str(self.filled_on)
                              if self.filled_on is not None else None),
                "fill_price": self.fill_price, "proceeds": self.proceeds,
                "reconciled": self.reconciled}


class VestEventKind(str, Enum):
    """The vest-session sequence, recorded so its order can be asserted.

    Semantic order, not source-line order. A test that reads the source proves
    only that the lines are arranged a certain way; this proves the events
    happened in that relation.
    """

    VEST_VALUED = "VEST_VALUED"
    WITHHOLDING_APPLIED = "WITHHOLDING_APPLIED"
    SHARES_DELIVERED = "SHARES_DELIVERED"
    EXTERNAL_FLOW_RECORDED = "EXTERNAL_FLOW_RECORDED"
    DISPOSITION_EVALUATED = "DISPOSITION_EVALUATED"
    DISPOSITION_DEFERRED = "DISPOSITION_DEFERRED"
    SALE_INSTRUCTION_CREATED = "SALE_INSTRUCTION_CREATED"
    SALE_EXECUTED = "SALE_EXECUTED"


#: The order these events must occur in. Later entries may be absent — a hold
#: never reaches a sale — but none may precede an earlier one.
CANONICAL_ORDER = (
    VestEventKind.VEST_VALUED,
    VestEventKind.WITHHOLDING_APPLIED,
    VestEventKind.SHARES_DELIVERED,
    VestEventKind.EXTERNAL_FLOW_RECORDED,
    VestEventKind.DISPOSITION_EVALUATED,
    VestEventKind.SALE_INSTRUCTION_CREATED,
    VestEventKind.SALE_EXECUTED,
)


@dataclass
class EventLog:
    """What happened, in the order it happened."""

    entries: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, kind: VestEventKind, **detail: Any) -> None:
        self.entries.append({"kind": kind, **detail})

    def kinds(self) -> List[VestEventKind]:
        return [entry["kind"] for entry in self.entries]

    def in_order(self, sequence: Sequence[Any],
                 ignoring: Sequence[Any] = ()) -> bool:
        """Whether the recorded events respect a declared sequence.

        Events not named in `sequence` are skipped rather than failing, so one
        log can carry several sequences — a vest's and an allocation's — and
        each be checked against its own.
        """
        positions = []
        for kind in self.kinds():
            if kind in ignoring or kind not in sequence:
                continue
            positions.append(list(sequence).index(kind))
        return all(earlier <= later
                   for earlier, later in zip(positions, positions[1:]))

    def in_canonical_order(self) -> bool:
        """The vest-to-sale sequence.

        Deferrals are excluded: an instruction may be deferred any number of
        times, between any two stages.
        """
        return self.in_order(CANONICAL_ORDER,
                             ignoring=(VestEventKind.DISPOSITION_DEFERRED,))


def new_instruction_id() -> str:
    return f"disp-{uuid.uuid4().hex[:16]}"


# --- policy ----------------------------------------------------------------

#: Fraction of delivered shares each policy sells. Concentration-targeted
#: policies are deliberately absent: sizing one needs the whole portfolio, and
#: substituting a fixed fraction would answer a different question silently.
_FRACTIONS = {"HOLD": 0.0, "SELL_ALL_AND_DIVERSIFY": 1.0,
              "SELL_HALF_AND_DIVERSIFY": 0.5}

CONCENTRATION_TARGETED = ("REDUCE_CONCENTRATION_BELOW",)


def instruction_for(*, vest_ref: str, grant_ref: str, asset: str,
                    delivered_shares: float, policy: str,
                    delivery_session, blackouts: Sequence[tuple] = (),
                    execution_lag: int = 1, expires_at=None,
                    sizing_policy: Any = None,
                    log: Optional[EventLog] = None
                    ) -> Optional[DispositionInstruction]:
    """The instruction a policy produces, or None where it produces no sale.

    `HOLD` yields no instruction at all rather than a zero-quantity one. A
    zero-share sale in the history would read as an attempt that failed.
    """
    name = str(policy or "").upper()
    if log is not None:
        log.record(VestEventKind.DISPOSITION_EVALUATED, policy=name)

    if any(name.startswith(prefix) for prefix in CONCENTRATION_TARGETED):
        if sizing_policy is None:
            raise UnsupportedPolicy(
                f"policy {policy!r} sizes the sale from portfolio "
                "concentration and no concentration policy was supplied. "
                "Approximating it as a fixed fraction would sell a different "
                "number of shares than was asked for, and nothing in the "
                "result would show the substitution")

        # Quantity deliberately zero at creation. It is solved at the first
        # eligible session against the portfolio as it is then.
        instruction = DispositionInstruction(
            instruction_id=new_instruction_id(), grant_ref=grant_ref,
            created_from_vest=vest_ref, asset=asset, quantity=0.0, policy=name,
            earliest_eligible_date=delivery_session,
            blackout_ref=tuple(blackouts), execution_lag=execution_lag,
            expires_at=expires_at, sizing_policy=sizing_policy,
            employer_asset=asset)
        if log is not None:
            log.record(VestEventKind.SALE_INSTRUCTION_CREATED,
                       instruction_id=instruction.instruction_id,
                       quantity=None)
        return instruction

    if name not in _FRACTIONS:
        raise UnsupportedPolicy(f"unknown disposition policy {policy!r}")

    fraction = _FRACTIONS[name]
    if fraction <= 0:
        return None

    instruction = DispositionInstruction(
        instruction_id=new_instruction_id(), grant_ref=grant_ref,
        created_from_vest=vest_ref, asset=asset,
        quantity=delivered_shares * fraction, policy=name,
        # Never earlier than the delivery. Selling before delivery would trade
        # shares the account does not hold.
        earliest_eligible_date=delivery_session,
        blackout_ref=tuple(blackouts), execution_lag=execution_lag,
        expires_at=expires_at)

    if log is not None:
        log.record(VestEventKind.SALE_INSTRUCTION_CREATED,
                   instruction_id=instruction.instruction_id,
                   quantity=instruction.quantity)
    return instruction


# --- scheduling ------------------------------------------------------------


def eligibility(instruction: DispositionInstruction, session, *,
                held_shares: float, price: Optional[float]
                ) -> DispositionEligibility:
    """Whether this instruction may execute on this session.

    Every blocking reason is named. An instruction that simply does not fire
    leaves shares unsold with nothing saying why.
    """
    blocked: List[str] = []

    if instruction.earliest_eligible_date is not None \
            and session < instruction.earliest_eligible_date:
        blocked.append("before the shares were delivered")

    for start, end in instruction.blackout_ref:
        import pandas as pd

        if pd.Timestamp(start) <= session <= pd.Timestamp(end):
            blocked.append(f"inside blackout {start}..{end}")

    if price is None or price != price or price <= 0:
        blocked.append("no usable price on this session")

    if held_shares + 1e-9 < instruction.quantity:
        blocked.append(
            f"only {held_shares:g} shares held, {instruction.quantity:g} required")

    return DispositionEligibility(eligible=not blocked, session=session,
                                  blocked_by=tuple(blocked))


def advance(instruction: DispositionInstruction, session, *,
            held_shares: float, price: Optional[float],
            log: Optional[EventLog] = None) -> DispositionInstruction:
    """Move an instruction one session forward, returning its new state.

    A settled instruction is returned untouched. Recomputing status from dates
    would let an executed sale become pending again because a later blackout
    covers its date.
    """
    if instruction.status in TERMINAL:
        return instruction

    if instruction.expires_at is not None and session > instruction.expires_at:
        return replace(instruction, status=DispositionStatus.EXPIRED,
                       detail="reached its declared expiry without executing")

    verdict = eligibility(instruction, session, held_shares=held_shares,
                          price=price)
    if verdict.eligible:
        return replace(instruction, status=DispositionStatus.ELIGIBLE,
                       detail="")

    if log is not None:
        log.record(VestEventKind.DISPOSITION_DEFERRED,
                   instruction_id=instruction.instruction_id,
                   blocked_by=verdict.blocked_by)
    return replace(instruction, status=DispositionStatus.PENDING,
                   detail="; ".join(verdict.blocked_by))


def supersede(instruction: DispositionInstruction, *, reason: str
              ) -> DispositionInstruction:
    """Replace an instruction with a later decision, keeping both.

    The superseded one stays in the history: a worksheet that showed only the
    surviving instruction would hide that a plan was changed.
    """
    if instruction.status in TERMINAL:
        raise ValueError(
            f"instruction {instruction.instruction_id} is already "
            f"{instruction.status.value} and cannot be superseded")
    return replace(instruction, status=DispositionStatus.SUPERSEDED,
                   detail=reason)


def fail(instruction: DispositionInstruction, *, reason: str
         ) -> DispositionInstruction:
    return replace(instruction, status=DispositionStatus.FAILED, detail=reason)


class DispositionSchedule:
    """Carries instructions across sessions and emits sales when eligible.

    Stateful on purpose: the whole point is that an instruction outlives the
    session that created it. A closure that decided per session could not
    report, afterwards, that a sale was owed and never happened.
    """

    def __init__(self, instructions: Sequence[DispositionInstruction] = (),
                 *, log: Optional[EventLog] = None) -> None:
        self.instructions: List[DispositionInstruction] = list(instructions)
        self.log = log if log is not None else EventLog()
        self.executions: List[DispositionExecution] = []

    def add(self, instruction: Optional[DispositionInstruction]) -> None:
        if instruction is not None:
            self.instructions.append(instruction)

    @property
    def outstanding(self) -> List[DispositionInstruction]:
        return [one for one in self.instructions if one.status not in TERMINAL]

    def program(self):
        """An event program the simulation engine can run.

        Emits at most one sale order per instruction. The engine applies the
        execution lag and the transaction cost, because the sale *is* a trade —
        unlike the delivery that created the shares.
        """
        from ..mission.accounting import Order

        def step(session, visible, holdings, cash):
            orders: List[Order] = []
            for index, instruction in enumerate(self.instructions):
                if instruction.status in TERMINAL:
                    continue

                held = float(holdings.get(instruction.asset, 0.0))
                price = None
                if instruction.asset in visible.columns and len(visible):
                    candidate = float(visible.iloc[-1][instruction.asset])
                    price = candidate if candidate == candidate else None

                # Sized here, against the portfolio as it is now. Solved at the
                # vest it would answer a question about a portfolio that has
                # since changed.
                if instruction.sizes_from_portfolio:
                    instruction = self._size(instruction, session, visible,
                                             holdings, cash, price=price)
                    self.instructions[index] = instruction
                    if instruction.status in TERMINAL or not instruction.quantity:
                        continue

                moved = advance(instruction, session, held_shares=held,
                                price=price, log=self.log)
                self.instructions[index] = moved
                if moved.status is not DispositionStatus.ELIGIBLE:
                    continue

                orders.append(Order(session, instruction.asset,
                                    -instruction.quantity * (price or 0.0),
                                    reason=f"disposition {instruction.policy}"))
                self.instructions[index] = replace(
                    moved, status=DispositionStatus.EXECUTED,
                    detail=f"sold on {session}")
                self.executions.append(DispositionExecution(
                    instruction_id=instruction.instruction_id,
                    instructed_on=session, shares=instruction.quantity,
                    expected_price=price or 0.0))
                self.log.record(VestEventKind.SALE_EXECUTED,
                                instruction_id=instruction.instruction_id,
                                session=session)
            return orders

        return step

    def _size(self, instruction, session, visible, holdings, cash, *, price):
        """Solve a concentration-targeted quantity from the live portfolio."""
        from .concentration import Feasibility, assess, solve

        if price is None:
            return instruction

        latest = visible.iloc[-1] if len(visible) else {}
        prices = {asset: float(latest[asset])
                  for asset in getattr(visible, "columns", ())
                  if asset in latest}
        assessment = assess(holdings=dict(holdings), prices=prices, cash=cash,
                            employer_asset=instruction.employer_asset or
                            instruction.asset,
                            policy=instruction.sizing_policy,
                            measured_at=session)
        plan = solve(assessment, price=price,
                     held_shares=float(holdings.get(instruction.asset, 0.0)),
                     policy=instruction.sizing_policy)

        if plan.feasibility is Feasibility.UNCOMPUTABLE:
            # Refused, not approximated. A denominator missing an unpriced
            # holding sizes the sale too small and reports success.
            return replace(instruction, status=DispositionStatus.PENDING,
                           sized_at=session, sizing_plan=plan,
                           detail="; ".join(plan.unresolved_inputs)
                           or plan.detail)
        if plan.feasibility is Feasibility.ALREADY_SATISFIED:
            return replace(instruction, status=DispositionStatus.EXPIRED,
                           sized_at=session, sizing_plan=plan,
                           detail="the declared cap was already satisfied")

        return replace(instruction, quantity=plan.shares_to_sell,
                       sized_at=session, sizing_plan=plan,
                       detail=plan.detail)

    def reconcile(self, fills) -> List[Dict[str, Any]]:
        """Attach what actually happened to each instructed sale.

        An instruction the engine never filled is marked FAILED with the reason,
        not left looking executed. A sale that was decided and did not happen is
        the failure this whole lifecycle exists to make visible.
        """
        available = [f for f in fills if f.shares < 0]
        for index, execution in enumerate(self.executions):
            match = next((f for f in available
                          if f.ticker == self._asset_of(execution.instruction_id)
                          and abs(abs(f.shares) - execution.shares) < 1e-6
                          and f.date >= execution.instructed_on), None)
            if match is None:
                self._mark_failed(execution.instruction_id,
                                  "the order was instructed and never filled")
                continue
            available.remove(match)
            self.executions[index] = replace(
                execution, filled_on=match.date, fill_price=match.price,
                proceeds=abs(match.shares) * match.price,
                fill_cost=getattr(match, "cost", 0.0))
        return self.unsettled_report()

    def _asset_of(self, instruction_id: str) -> str:
        for one in self.instructions:
            if one.instruction_id == instruction_id:
                return one.asset
        return ""

    def _mark_failed(self, instruction_id: str, reason: str) -> None:
        for index, one in enumerate(self.instructions):
            if one.instruction_id == instruction_id:
                self.instructions[index] = fail(
                    replace(one, status=DispositionStatus.PENDING),
                    reason=reason)

    def unsettled_report(self) -> List[Dict[str, Any]]:
        """Instructions that never executed, and why.

        The report that stops a dropped sale from looking like a decision not
        to sell.
        """
        return [{"instruction_id": one.instruction_id, "asset": one.asset,
                 "quantity": one.quantity, "status": one.status.value,
                 "why": one.detail}
                for one in self.instructions
                if one.status is not DispositionStatus.EXECUTED]
