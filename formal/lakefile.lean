import Lake
open Lake DSL

package quantify

/-- `@[default_target]`, and it is not decoration.

    Without it `lake build` with no argument builds nothing and exits zero. The
    CI step ran exactly that, and a mutation — a `#guard` changed to a wrong
    number, and `- s.fees` deleted from `endingCash` — passed twice. A
    verification lane that verifies nothing is worse than an absent one,
    because the badge is green. -/
@[default_target]
lean_lib Quantify where
  roots := #[`Quantify.Types, `Quantify.Ledger, `Quantify.Cadence, `Quantify.Fixtures]
