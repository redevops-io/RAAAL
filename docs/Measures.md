# What the figures mean

Every reported number, defined once.

> Consolidated from `Measures.md`, `Measures.md`, `Measures.md`.
>
> A figure defined in one file and computed in another drifts. These are the definitions the code is checked against.


---

## Quantify Money-Weighted Return — the definition

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


---

## Quantify drawdown — the definition

> **Drawdown at a session is how far the portfolio sits below its own
> high-water mark to that point, as a fraction of that mark. Maximum drawdown
> is the largest such value over the reported evaluation period.**

Non-negative: a quarter below the peak is `0.25`. The implementation returns
the same magnitude negated, and `tests/test_drawdown_conformance.py` converts
once rather than letting the two conventions meet by accident.

The high-water mark includes the opening level. A portfolio that falls on its
first session has fallen from where it started, and a peak that begins at the
first *move* has already absorbed that fall.

Maximum drawdown is not current drawdown. `100 → 120 → 90 → 110 → 130` ends at
a new high with a current drawdown of zero and a maximum of 25%. Recovery does
not erase history.

Warm-up sessions may feed an indicator and may not contribute a reported
drawdown, the same boundary `Window.lean` proves for contributions.

## Semantics versions

    drawdown@1   opening equity level missing from the curve
                 falls from the opening level understated, a first-session
                 crash invisible; historical results potentially understated

    drawdown@2   opening level included; conforms to the definition above

`@1` built the equity curve as `(1 + returns).cumprod()`, which starts at the
first *return*. The opening level was never in the curve, so `cummax` began at
the post-first-move value.

    _max_drawdown([-0.5])            0.0     a portfolio that halved
    100 → 75 → 50                    1/3     the definition says 1/2
    100 → 50 → 100 → 101 → 102       0.0     the definition says 1/2

Any fall from the opening level is understated, and a crash in the first
session vanishes entirely.

**Fixed in `@2`.** The curve now begins at 1.0 and every fixture the definition
exposed conforms.

Stored results are not recomputed. Every persisted drawdown was produced under
`@1`, and regenerating them silently would present old evidence as though it
had always been measured this way — so new evaluations carry
`drawdown_semantics: drawdown@2` beside the number and older artifacts remain
historical evidence unless explicitly regenerated.

Which fixture caught it is worth recording. `100 → 120 → 90 → 110 → 130`, the
series chosen to separate a correct implementation from three plausible wrong
ones, conformed under `@1` — it rises first, so the missing opening level never
becomes the peak. It took a series that falls immediately.


---

## Formal Financial Core v1

> **Quantify's financial core is formally verified with Lean 4: ledger
> conservation, cash flows, positions, cadence, trigger semantics, evaluation
> windows, ordering, valuation, TWR, MWR, drawdown, and headline accounting
> figures are tied to proved deterministic operators; undefined returns are
> surfaced explicitly rather than rendered as zero.**

Every claim in that sentence is checked by `tests/test_formal_core.py` against
the theorem tree, because a milestone statement is the thing most likely to
outlive what it describes.

## What is proved

| Module | Covers |
|---|---|
| `Types` | exact money, prices, shares; scaled integers, no floating point |
| `Ledger` | cash conservation, position conservation, portfolio valuation |
| `Summary` | contributed, withdrawn, net, gain — all from one flow authority |
| `Cadence` | once / monthly / annual; N periods → N contributions → N × A |
| `Triggers` | crossing versus persistent, and the ratio between them |
| `Ordering` | causality, no look-ahead, fill association by identity |
| `Window` | warm-up may feed an indicator and may not be reported |
| `MovingAverage` | the threshold series, and where an off-by-one hides |
| `Composition` | the operators wired in the order Python wires them |
| `Returns/Simple` | simple return |
| `Returns/TimeWeighted` | TWR, and that a boundary flow is not performance |
| `Returns/MoneyWeighted` | the MWR reporting contract |
| `Returns/Drawdown` | drawdown, and that recovery does not erase the maximum |

Mathlib is required only under `Returns/`. The core builds from a bare
toolchain, and a test asserts that no core module imports it.

## What "verified" means here, precisely

Three different things, kept apart:

    proved            a Lean theorem, general over its inputs
    fixture-checked   a Lean `#guard` on concrete numbers
    conformance       the Python implementation agrees with the definition

A theorem about an operator says nothing about whether the engine runs that
operator. Each conformance lane checks the implementation separately, and where
it disagreed the disagreement is recorded rather than absorbed:

- MWR reported one root of a two-root series — closed, and the return type
  widened so `NON_UNIQUE` can be said at all.
- Drawdown omitted the opening level, so a first-session crash reported zero —
  closed under `drawdown@2`, with the old semantics named rather than
  retrofitted onto stored results.
- TWR conformed; the defect was in presentation, below.

## Undefined is not zero

Three templates rendered an undefined return as `+0.00%`, which reads as "broke
even" for a number that does not exist. Both return bases had it. A test reads
the templates so the substitution cannot return.

## The boundary this does not cross

    Discovery   what the user means
    Mission     whether the approved intent is faithfully representable
    Quantify    deterministic financial execution
    Lean        proof that those calculations obey their contract

`coverage` and `modelling_scope` stay in the runtime-contract layer. They ask
whether a specification faithfully covers what a person approved, which is a
question about interpretation; pulling it into Lean would drag semantic
judgement back into the formal core.

Volatility is absent for the same reason inverted: nothing in the engine
computes one, and adding a metric in order to have something to verify would
reverse the dependency. Lean certifies semantics that already travel a governed
result path.
