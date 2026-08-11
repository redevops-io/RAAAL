import Lake
open Lake DSL

package quantify

/-- Mathlib, required only for `Quantify.Returns`.

    Return metrics are where the mathematics changes kind. Everything proven so
    far is discrete — integers, lists, counts, ordering, identity — and scaled
    integers were the right representation for it. A time-weighted return is a
    product of ratios, and stating it over fixed-point integers would mean
    proving facts about a rounding policy instead of about the return.

    The dependency is contained at the module boundary: nothing under
    `Quantify/` outside `Returns/` imports it, and `test_lean_ledger_conformance`
    asserts that structurally. The core stays buildable from a bare toolchain. -/
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.14.0"

/-- The lightweight core. No Mathlib. -/
@[default_target]
lean_lib Quantify where
  roots := #[`Quantify.Types, `Quantify.Ledger, `Quantify.Cadence,
             `Quantify.Triggers, `Quantify.Ordering, `Quantify.Window,
             `Quantify.MovingAverage, `Quantify.Composition,
             `Quantify.Fixtures]

/-- Return metrics, over exact rationals.

    A second target, and `@[default_target]` on both. A bare `lake build` that
    built only one of them would be the "green CI over unbuilt proofs" defect
    again — which this repository has already hit once, when the library had no
    default target at all and two mutations passed. -/
@[default_target]
lean_lib QuantifyReturns where
  roots := #[`Quantify.Returns.Simple, `Quantify.Returns.TimeWeighted, `Quantify.Returns.MoneyWeighted, `Quantify.Returns.Drawdown]
