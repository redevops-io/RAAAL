# Quantify Money-Weighted Return — the definition

Frozen before any Lean was written, deliberately. A proof written first would
have formalised whatever the root finder happened to do, and "our solver
returned a number" is a theorem that looks like success and means nothing.

> **Quantify Money-Weighted Return is the unique annualized rate, on an
> Actual/365 Fixed basis, that sets the present value of investor external cash
> flows plus terminal portfolio value to zero. If no unique admissible rate
> exists, Quantify reports no MWR rather than selecting an arbitrary numerical
> root.**

## What is a cash flow

Only money crossing the boundary between the investor and the portfolio.

    external contribution        negative
    external withdrawal          positive
    terminal portfolio value     positive, once, on the end date

Everything inside the portfolio is invisible to this calculation:

    buy or sell inside the portfolio      ignored
    dividend retained in the portfolio    ignored
    rebalance                             ignored
    cash moved between holdings           ignored

This is formal rather than a convention, because a rebalance that counted as a
flow would manufacture cash movements and change the reported return. The
theorem that internal activity cannot move MWR is what makes the number a
statement about the investor's experience rather than about how the strategy
happened to trade.

## The equation

For dated flows `(dayᵢ, cashᵢ)` measured from the evaluation start:

    Σ cashᵢ / (1 + r) ^ (dayᵢ / 365) = 0

Actual/365 Fixed. Chosen because it is deterministic and states in one line;
"a year" is not. It also happens to be the basis Excel's XIRR uses, so a later
compatibility claim needs no second convention.

## When a rate may be reported

All five, or no rate:

1. every external flow is dated and inside the evaluation interval;
2. the terminal portfolio value appears exactly once, on the end date;
3. the signed flows contain both investment and recovery — economically, both
   negative and positive terms;
4. a solution exists with `r > -1`;
5. **the solution is unique** in that domain.

## What is reported

    RATE(r)                     all five hold
    NO_SOLUTION                 no admissible root
    NON_UNIQUE                  more than one admissible root
    INSUFFICIENT_CASH_FLOWS     conditions 1–3 not met

Publishing a number for `NON_UNIQUE` would mean picking whichever root the
solver reached first, and two runs of the same portfolio could then report
different returns with equal confidence.

## What the definition is not

It is not the algorithm. The production solver is an *implementation* that must
return a root satisfying the predicate; it does not decide what MWR means. Lean
proves the reporting contract — that a rate may be published only when it is
the unique admissible root — independently of how the root is found.

    financial definition   ≠   numerical algorithm

## Known non-conformance — Closed

`mission.accounting.money_weighted_return` does not implement the fourth
outcome. It returns `Optional[float]`, so `NO_SOLUTION` and
`INSUFFICIENT_CASH_FLOWS` both arrive as `None` and `NON_UNIQUE` has no
representation at all.

Found by running it, not by reading it:

    flows      -100 at session 0, +450 at session 1
    terminal   450 at session 2

    f(g) = -100g² + 450g - 450     roots at g = 1.5 and g = 3.0
    admissible rates               50% and 200%

    solver returns                 0.4999999999
    contract says                  NON_UNIQUE, report nothing

The solver's docstring justifies uniqueness by Descartes' rule — one sign
change in the coefficient sequence gives one positive root. That is true for a
series of contributions plus a terminal value, and false for a series
containing a withdrawal. Nothing checks that the series has the shape the
argument assumes, so bisection returns whichever root its opening bracket
happens to straddle and reports it with no qualification.

`tests/test_mwr_conformance.py` asserts this is still the behaviour, so the
record cannot go stale without a test failing.

**Closed.** `money_weighted_return` now returns `MWRResult`, and on this series
reports `NON_UNIQUE` with no rate. The implementation is split the way the
contract is — validate the cash flows, search for roots, classify what was
found — and uniqueness is *established* rather than assumed: one sign change in
the coefficient sequence means exactly one positive root by Descartes and a
rate may be published; more than one means the rule permits several, so the
search decides and refuses to publish when it cannot tell.

The implementation carries one state the financial contract does not:
`INDETERMINATE`, for when the search cannot establish which of the four
applies. A bounded scan detects crossings, so a root that touches zero without
crossing or sits beyond the searched range is invisible to it — and reporting
`NO_SOLUTION` there would turn "could not establish" into "established". It is
an implementation state and not a fifth outcome of the definition.
