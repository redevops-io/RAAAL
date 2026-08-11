# Quantify drawdown — the definition

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
