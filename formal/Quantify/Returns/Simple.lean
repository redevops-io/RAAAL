/-
  Simple return, over exact rationals.

  `ℚ`, not fixed-point integers and not `Float`. A return is a ratio, and
  stating it over scaled integers would mean every theorem carried a rounding
  policy — proving facts about truncation rather than about the return. Lean's
  own `Float` models finite machine arithmetic with rounding and special
  values, and is opaque at the logic level; formalising it here would make the
  verification target harder than the finance.

  The engine still computes in floating point. That is deliberate and not a
  contradiction: this states what the number *means*, the conformance lane
  checks the implementation agrees to a declared tolerance, and the two stay
  separate questions.
-/

import Mathlib.Data.Rat.Defs
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace Quantify
namespace Returns

/-- The return of a value that went from `start` to `finish`.

    `start` must be non-zero, and it is a hypothesis rather than a guard
    returning zero: a portfolio with nothing in it has no return, and reporting
    `0` would say it broke even. -/
def simpleReturn (start finish : ℚ) (h : start ≠ 0) : ℚ :=
  (finish - start) / start

/-- **No change is no return.** -/
theorem no_change_is_zero (v : ℚ) (h : v ≠ 0) : simpleReturn v v h = 0 := by
  -- `h` is unused in the proof and required in the statement: the function
  -- will not accept a zero start, and a version taking one would be saying a
  -- portfolio with nothing in it broke even.
  simp [simpleReturn]

/-- 100 → 110 is a tenth. -/
theorem ten_percent : simpleReturn 100 110 (by norm_num) = 1 / 10 := by
  norm_num [simpleReturn]

/-- A loss is negative, and exactly as large as it should be. -/
theorem ten_percent_down : simpleReturn 100 90 (by norm_num) = -1 / 10 := by
  norm_num [simpleReturn]

/-- **The value implied by a return recovers the value.** The formula inverts,
    which rules out a sign or denominator error that the fixtures above would
    not catch on their own. -/
theorem return_determines_the_finish (start finish : ℚ) (h : start ≠ 0) :
    start * (1 + simpleReturn start finish h) = finish := by
  unfold simpleReturn
  field_simp

end Returns
end Quantify
