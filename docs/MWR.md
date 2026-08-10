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
