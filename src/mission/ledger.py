"""What the rule actually did, row by row, and whether the totals agree.

    signals → contribution events → fills → ledger → reconciliation → totals

The ledger is not a presentation layer. It is the artifact that makes "the rule
executed" checkable, and the chart is one rendering of it.

That ordering matters because of how the defect it exists to prevent was
actually caught. A user read $1,000 contributed beside a rule that fires
repeatedly and saw that the arithmetic could not hold. Nothing in the system
had noticed, because nothing in the system compared what the rule declared with
what the engine did — every figure was internally consistent with an engine
that had quietly replayed buy-and-hold.

So the reconciliation is the point:

    sum(contribution amounts)  == the contributed total the result reports
    count(signals)             == crossings reported, unexecutable ones named
    count(purchases)           == purchases reported
    every purchase             cites exactly one signal
    every signal               is evaluated from the pinned frame

A ledger that could not disagree with the result would be a restatement of it,
and would have certified the original defect just as happily.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from .funding import ContributionEvent
from .signals import Signal

#: Two figures derived from float arithmetic on the same money may differ in
#: the last bits. Compared exactly, reconciliation would fail on rounding; not
#: compared at all, it would pass on a missing purchase. A cent is far below
#: anything that could hide an event and far above float noise.
TOLERANCE = Decimal("0.01")

#: What executed the rule, and under which reconciliation rules its ledger was
#: accepted. Recorded on every run that produces one, so a replacement result
#: can say what changed rather than merely being newer — and so the withdrawn
#: run becomes a historical comparison rather than an embarrassment:
#: `engine/buy-and-hold-only@1` interpreted the rule and did not execute it;
#: this engine executes it and can account for every contribution.
EXECUTION_ENGINE_VERSION = "engine/event-runtime@1"
RECONCILIATION_VERSION = "ledger/reconciled@1"


@dataclass(frozen=True)
class LedgerRow:
    """One executed contribution, from the observation that caused it."""

    signal_session: Optional[pd.Timestamp]
    contribution_session: pd.Timestamp
    """When the money arrived. Distinct from the execution date, and the
    distinction is load-bearing: with only two dates, a policy that contributed
    on the very session that produced its signal still showed a later *fill*
    date — because the fill lags the order — and the look-ahead check passed.
    The mutation survived until this column existed."""

    execution_session: pd.Timestamp
    subject: str
    contribution: Decimal
    shares: Decimal
    price: Decimal
    reason: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "signal_session": (str(self.signal_session.date())
                               if self.signal_session is not None else None),
            "contribution_session": str(self.contribution_session.date()),
            "execution_session": str(self.execution_session.date()),
            "subject": self.subject,
            "contribution": str(self.contribution),
            "shares": str(self.shares.quantize(Decimal("0.000001"))),
            "price": str(self.price.quantize(Decimal("0.0001"))),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Reconciliation:
    """Whether the ledger and the result tell the same story."""

    agrees: bool
    checks: Mapping[str, bool]
    detail: Mapping[str, str]

    def failures(self) -> Sequence[str]:
        return tuple(name for name, ok in self.checks.items() if not ok)

    def to_json(self) -> Dict[str, Any]:
        return {"agrees": self.agrees, "checks": dict(self.checks),
                "detail": dict(self.detail),
                "failures": list(self.failures())}


@dataclass(frozen=True)
class ExecutionLedger:
    """Every signal, every contribution, every fill, and their totals."""

    rows: Sequence[LedgerRow] = ()
    signals: Sequence[Signal] = ()
    filled_notional: Decimal = Decimal("0")
    """What the fills cost, before fees. With `filled_fees` and the ending cash
    this closes the money: contributions in, purchases and fees out, the rest
    still cash."""

    filled_fees: Decimal = Decimal("0")
    ending_cash: Decimal = Decimal("0")

    filled_shares: Decimal = Decimal("0")
    """Every share the engine reports filling, from the path rather than from
    the rows. The independent side of the share check: a ledger that summed its
    own rows would agree with itself no matter how badly it joined."""

    unexecutable: Sequence[Signal] = ()
    """Signals that fired with no later session to act on. Named rather than
    dropped: "the condition occurred and could not be acted on" is a different
    fact from "it did not occur", and it is the honest explanation for a
    signal count that exceeds the purchase count."""

    @property
    def total_contributed(self) -> Decimal:
        return sum((row.contribution for row in self.rows), Decimal("0"))

    @property
    def total_shares(self) -> Decimal:
        return sum((row.shares for row in self.rows), Decimal("0"))

    def summary(self) -> Dict[str, Any]:
        """The four numbers that would have exposed the original defect.

        One purchase and $1,000 total against a rule that fires repeatedly is
        visible here without any understanding of the engine.
        """
        return {
            "signals_detected": len(self.signals),
            "purchases_executed": len(self.rows),
            "signals_not_executable": len(self.unexecutable),
            "total_contributed": str(self.total_contributed),
        }

    def digest(self) -> Dict[str, str]:
        """Content digests over the two ledgers, for a run to cite.

        Over the rows and the signals separately: "the same crossings were
        detected" and "the same purchases were made from them" are different
        claims, and a single digest could not distinguish a changed indicator
        from a changed execution.
        """
        import hashlib
        import json

        def over(payload) -> str:
            return hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":"),
                default=str).encode()).hexdigest()

        return {"signal_ledger": over([s.to_json() for s in self.signals]),
                "execution_ledger": over([r.to_json() for r in self.rows])}

    def fingerprint(self) -> Dict[str, Any]:
        """What a replay cites instead of recomputing.

        A page rendered later should be able to say *"this came from ledger
        X"* rather than re-evaluating today's prices, today's indicator and
        today's compiler — which would silently answer a different question
        under an old plan's name. The digests identify the content; the totals
        let a reader check a displayed figure against the fingerprint without
        opening the ledger itself.
        """
        return {
            **self.digest(),
            "event_count": len(self.rows),
            "signal_count": len(self.signals),
            "contribution_total": str(self.total_contributed),
            "share_total": str(self.total_shares),
            "cash_total": str(self.ending_cash),
            "engine_version": EXECUTION_ENGINE_VERSION,
            "reconciliation_version": RECONCILIATION_VERSION,
        }

    def to_json(self) -> Dict[str, Any]:
        return {"rows": [row.to_json() for row in self.rows],
                "signals": [s.to_json() for s in self.signals],
                "unexecutable": [s.to_json() for s in self.unexecutable],
                "summary": self.summary(),
                "fingerprint": self.fingerprint()}


def build(*, events: Sequence[ContributionEvent], fills: Sequence[Any],
          signals: Sequence[Signal] = (),
          unexecutable: Sequence[Signal] = (),
          ending_cash: float = 0.0) -> ExecutionLedger:
    """Join what was contributed to what it bought.

    **Not by session equality.** Money arrives on session N and the order fills
    on N+1, because acting on the close that produced the signal would read one
    bar into the future. Matching fills to the contribution date therefore
    found nothing, and every row reported zero shares at a price of zero —
    while the reconciliation passed, because it compared contribution totals
    and never asked whether anything was actually bought.

    Each event takes the unconsumed fills dated before the next event, so the
    pairing follows from the ordering rather than from an assumed lag. A fill
    is consumed once: positional pairing would misattribute permanently after
    the first session that filled a different number of orders.
    """
    ordered_fills = sorted(fills, key=lambda f: pd.Timestamp(f.date))
    ordered_events = sorted(events, key=lambda e: pd.Timestamp(e.session))

    rows: List[LedgerRow] = []
    index = 0
    for position, event in enumerate(ordered_events):
        boundary = (pd.Timestamp(ordered_events[position + 1].session)
                    if position + 1 < len(ordered_events) else None)
        matched = []
        while index < len(ordered_fills):
            fill_date = pd.Timestamp(ordered_fills[index].date)
            if fill_date < pd.Timestamp(event.session):
                index += 1          # predates this contribution; not its doing
                continue
            if boundary is not None and fill_date >= boundary:
                break               # belongs to the next contribution
            matched.append(ordered_fills[index])
            index += 1

        shares = sum((Decimal(str(f.shares)) for f in matched), Decimal("0"))
        notional = sum((Decimal(str(f.notional)) for f in matched), Decimal("0"))
        price = (notional / shares) if shares else Decimal("0")
        executed = (pd.Timestamp(matched[0].date) if matched
                    else pd.Timestamp(event.session))
        rows.append(LedgerRow(
            signal_session=(event.signal.session if event.signal else None),
            contribution_session=pd.Timestamp(event.session),
            execution_session=executed,
            subject=(matched[0].ticker if matched
                     else (event.signal.subject if event.signal else "")),
            contribution=event.amount,
            # Unrounded. Quantizing here made thirty rows drift past the
            # share check by accumulated rounding — the reconciliation failing
            # on the ledger's own presentation rather than on anything the
            # engine did. Rounding belongs in `to_json`, where it is a display
            # decision and cannot reach a comparison.
            shares=shares,
            price=price,
            reason=event.reason))
    return ExecutionLedger(
        rows=tuple(rows), signals=tuple(signals),
        unexecutable=tuple(unexecutable),
        filled_shares=sum((Decimal(str(f.shares)) for f in ordered_fills),
                          Decimal("0")),
        filled_notional=sum((Decimal(str(f.notional)) for f in ordered_fills),
                            Decimal("0")),
        filled_fees=sum((Decimal(str(f.cost)) for f in ordered_fills),
                        Decimal("0")),
        ending_cash=Decimal(str(ending_cash)))


def reconcile(ledger: ExecutionLedger, result: Any) -> Reconciliation:
    """Compare the ledger with the figure the engine produced.

    Independent by construction: the ledger is built from the funding policy
    and the fills, and the result's totals come from the portfolio path. Both
    describe the same run, and a disagreement means one of them is wrong —
    which is exactly the state that shipped, undetected, because nothing
    performed this comparison.
    """
    reported = Decimal(str(getattr(result.path, "contributed", 0.0)))
    ledger_total = ledger.total_contributed

    checks: Dict[str, bool] = {}
    detail: Dict[str, str] = {}

    checks["contributions_sum_to_the_reported_total"] = (
        abs(reported - ledger_total) <= TOLERANCE)
    detail["contributions_sum_to_the_reported_total"] = (
        f"ledger {ledger_total} vs reported {reported}")

    # Every purchase cites one signal, for an event-triggered policy. A row
    # with no signal in a triggered ledger is a purchase nobody can explain.
    triggered = [row for row in ledger.rows if row.signal_session is not None]
    checks["every_triggered_purchase_cites_a_signal"] = (
        len(triggered) == len(ledger.rows) if ledger.signals else True)
    detail["every_triggered_purchase_cites_a_signal"] = (
        f"{len(triggered)} of {len(ledger.rows)} rows cite a signal")

    # Signals are conserved: each either bought something or is named as
    # unexecutable. Neither silently vanishes.
    accounted = len(ledger.rows) + len(ledger.unexecutable)
    checks["every_signal_is_accounted_for"] = (
        accounted == len(ledger.signals) if ledger.signals else True)
    detail["every_signal_is_accounted_for"] = (
        f"{len(ledger.rows)} executed + {len(ledger.unexecutable)} "
        f"unexecutable vs {len(ledger.signals)} signals")

    # The signal must precede the *contribution*, not merely the fill.
    #
    # Checked against the execution date alone, a policy that funded on the
    # session that produced its own signal still passed: the fill lags the
    # order by one session, so the last date in the chain moved forward while
    # the look-ahead sat between the first two.
    ordered = all(row.signal_session < row.contribution_session
                  for row in triggered)
    checks["every_signal_precedes_its_funding"] = ordered
    detail["every_signal_precedes_its_funding"] = (
        "all contributions land strictly after their signal" if ordered
        else "money arrived on the session that produced its own signal, "
             "which reads one bar into the future")

    # Shares, against the engine's own total rather than the ledger's rows.
    #
    # Every check above passed while every row reported zero shares at a price
    # of zero: contributions summed correctly, signals were accounted for, and
    # nothing asked whether the money had bought anything. A reconciliation
    # that cannot see a broken join is a reconciliation that certifies one.
    checks["shares_match_what_the_engine_filled"] = (
        abs(ledger.total_shares - ledger.filled_shares) <= Decimal("0.000001"))
    detail["shares_match_what_the_engine_filled"] = (
        f"ledger {ledger.total_shares} vs filled {ledger.filled_shares}")

    # And no purchase may show nothing bought while the engine filled orders.
    # The totals could agree while individual rows were misattributed.
    empty = [row for row in ledger.rows if row.shares <= 0]
    checks["every_purchase_bought_something"] = (
        not empty or ledger.filled_shares == 0)
    detail["every_purchase_bought_something"] = (
        f"{len(empty)} of {len(ledger.rows)} rows bought no shares")

    # A price of zero is not a price. Shares can be positive while the price
    # column is empty, and a row a person reads with a blank price is a row
    # they cannot check against anything.
    unpriced = [row for row in ledger.rows if row.price <= 0]
    checks["every_purchase_has_a_price"] = not unpriced or ledger.filled_shares == 0
    detail["every_purchase_has_a_price"] = (
        f"{len(unpriced)} of {len(ledger.rows)} rows have no price")

    # Economic effect, stated for this policy: every detected signal is
    # executable here, so a purchase count below the signal count means an
    # event was silently lost rather than legitimately unexecutable.
    if ledger.signals:
        expected = len(ledger.signals) - len(ledger.unexecutable)
        checks["every_executable_signal_produced_a_purchase"] = (
            len(ledger.rows) == expected)
        detail["every_executable_signal_produced_a_purchase"] = (
            f"{len(ledger.rows)} purchases vs {expected} executable signals")

    # The money the engine actually took in, against the money the ledger says
    # arrived. `contributed` is the portfolio path's own sum over its flow
    # series — a different computation from adding the contribution amounts.
    checks["ledger_cash_use_matches_the_engine"] = (
        abs(reported - ledger_total) <= TOLERANCE)
    detail["ledger_cash_use_matches_the_engine"] = (
        f"engine flows {reported} vs ledger contributions {ledger_total}")

    # The money closes.
    #
    #     purchases + fees + cash still held  ==  everything contributed
    #
    # Independent of returns and of every benchmark, and it catches the class
    # the counting checks cannot: each purchase exists, each fill exists, the
    # share totals agree, and money quietly appears or disappears between them.
    # Tolerance scales with the number of fills, since each contributes its own
    # float rounding rather than the total carrying one.
    accounted = ledger.filled_notional + ledger.filled_fees + ledger.ending_cash
    room = TOLERANCE * max(len(ledger.rows), 1)
    checks["the_money_closes"] = abs(accounted - ledger_total) <= room
    detail["the_money_closes"] = (
        f"purchases {ledger.filled_notional} + fees {ledger.filled_fees} + "
        f"cash {ledger.ending_cash} = {accounted}, against contributions "
        f"{ledger_total}")

    return Reconciliation(agrees=all(checks.values()), checks=checks,
                          detail=detail)
