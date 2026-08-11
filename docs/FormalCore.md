# Formal Financial Core v1

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
