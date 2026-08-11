/-
  Drawdown: how far below its own high-water mark the portfolio went.

  Formalised because `evaluation/runner.py::_max_drawdown` already computes it
  on a real result path. That order matters and is the rule this file follows:

      Lean certifies semantics that already participate in a Quantify result.
      It does not create product semantics because a metric would be useful.

  Volatility is the counter-example. Nothing in the engine computes one, so a
  proof of it would be a definition with nothing to conform to — and adding the
  metric in order to have something to verify would reverse the dependency.

  **Sign.** Drawdown here is a non-negative magnitude: a quarter below the peak
  is `1/4`. The implementation returns `min(curve/cummax - 1)`, which is `-1/4`
  for the same series. Neither is wrong and the conformance lane converts
  explicitly rather than letting the two conventions meet by accident.
-/

import Mathlib.Data.Rat.Defs
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

namespace Quantify
namespace Returns

/-- The high-water mark at each session, inclusive of that session. -/
def runningPeaks : List ℚ → List ℚ
  | []          => []
  | v :: rest   => v :: (runningPeaks rest).map (max v)

/-- How far below its peak one session sat, as a fraction of the peak.

    Zero when the peak is not positive. A portfolio worth nothing has not
    fallen by a percentage of nothing, and the alternative is a division that
    every theorem below would have to carry a hypothesis about. -/
def drawdownAt (peak value : ℚ) : ℚ :=
  if peak ≤ 0 then 0 else (peak - value) / peak

/-- Every session's drawdown, in order. -/
def drawdowns (values : List ℚ) : List ℚ :=
  (runningPeaks values).zipWith drawdownAt values

/-- The worst of them. Zero for an empty series, which is the only honest
    answer: nothing happened, so nothing was lost. -/
def maxDrawdown (values : List ℚ) : ℚ :=
  (drawdowns values).foldl max 0

/-- A running maximum never drops below where it started. Generalised over the
    accumulator, because `foldl` carries one and an induction that fixed it at
    zero cannot take the step — the same shape as the ledger's
    `foldl_add_nonneg`. -/
theorem le_foldl_max :
    ∀ (xs : List ℚ) (acc : ℚ), acc ≤ xs.foldl max acc
  | [],        acc => by simp
  | x :: rest, acc => by
    have step := le_foldl_max rest (max acc x)
    exact le_trans (le_max_left acc x) step

/-- **Never negative.** Stated because a drawdown reported as a gain is exactly
    the sign error this file's two conventions make possible. -/
theorem maxDrawdown_nonneg (values : List ℚ) : 0 ≤ maxDrawdown values :=
  le_foldl_max (drawdowns values) 0

/-- The current drawdown: the last session's, or zero for an empty series.

    Distinct from `maxDrawdown` on purpose. They are different questions — how
    far below the peak the portfolio sits *now*, and how far below it ever
    went — and an implementation that answered the first while being asked the
    second would look right on any series ending at its high. -/
def currentDrawdown (values : List ℚ) : ℚ :=
  ((drawdowns values).getLast?).getD 0

/-! ## The discriminating series

    `100 → 120 → 90 → 110 → 130`

    Peak 120, trough 90, so the worst drawdown is 25% — and the series ends at
    a new high. Three wrong implementations all give something else here:

        loss from the start        0, the series ended up
        terminal drawdown          0, it finished at its peak
        current drawdown           0, same reason

    A series that merely fell and stayed down would not separate them.
-/

def recovers : List ℚ := [100, 120, 90, 110, 130]

#guard runningPeaks recovers == [100, 120, 120, 120, 130]

/-- **A quarter below the peak**, on a series that finished higher than it
    started. -/
theorem worst_drawdown_is_a_quarter : maxDrawdown recovers = 1 / 4 := by
  norm_num [maxDrawdown, currentDrawdown, drawdowns, runningPeaks,
            drawdownAt, List.zipWith, List.foldl, List.getLast?, recovers]

/-- **Recovery does not erase it.** The series ends at a new high and the
    maximum still stands. -/
theorem recovery_does_not_erase_the_maximum :
    currentDrawdown recovers = 0 ∧ maxDrawdown recovers = 1 / 4 := by
  constructor <;> norm_num [maxDrawdown, currentDrawdown, drawdowns, runningPeaks,
            drawdownAt, List.zipWith, List.foldl, List.getLast?, recovers]

/-- **A new high resets the current drawdown and not the maximum.** The two
    numbers disagree here, which is what makes the fixture worth having. -/
theorem a_new_high_separates_the_two :
    currentDrawdown recovers ≠ maxDrawdown recovers := by
  norm_num [maxDrawdown, currentDrawdown, drawdowns, runningPeaks,
            drawdownAt, List.zipWith, List.foldl, List.getLast?, recovers]

/-- **Rising monotonically means no drawdown.** -/
def climbs : List ℚ := [100, 110, 120, 130]

theorem a_rising_series_has_none : maxDrawdown climbs = 0 := by
  norm_num [maxDrawdown, currentDrawdown, drawdowns, runningPeaks,
            drawdownAt, List.zipWith, List.foldl, List.getLast?, climbs]

/-- And falling has all of it. The converse guard: an implementation returning
    zero unconditionally satisfies the theorem above. -/
def falls : List ℚ := [100, 75, 50]

theorem a_falling_series_has_half : maxDrawdown falls = 1 / 2 := by
  norm_num [maxDrawdown, currentDrawdown, drawdowns, runningPeaks,
            drawdownAt, List.zipWith, List.foldl, List.getLast?, falls]

/-- **Loss from the start is a different number.** Stated so the fixture cannot
    be satisfied by an implementation measuring the wrong baseline. -/
theorem the_peak_is_not_the_starting_value :
    maxDrawdown recovers ≠ (100 - 90) / 100 := by
  norm_num [maxDrawdown, currentDrawdown, drawdowns, runningPeaks,
            drawdownAt, List.zipWith, List.foldl, List.getLast?, recovers]

/-! ## Window boundaries

    Warm-up sessions feed indicators and must not contribute a reported
    drawdown. The composition is the same one `Window.lean` proves for
    contributions: the metric is computed over the reported slice, never over
    the frame.
-/

/-- A crash inside the warm-up, and a quiet reported period. -/
def crashThenCalm : List ℚ := [100, 50, 100, 101, 102]

/-- Over the whole frame the worst drawdown is a half. -/
theorem the_frame_saw_a_crash : maxDrawdown crashThenCalm = 1 / 2 := by
  norm_num [maxDrawdown, currentDrawdown, drawdowns, runningPeaks,
            drawdownAt, List.zipWith, List.foldl, List.getLast?, crashThenCalm]

/-- Over the reported slice alone it is nothing — the crash is data the
    indicator may use and not a loss the report may claim. -/
theorem the_reported_slice_saw_none :
    maxDrawdown (crashThenCalm.drop 2) = 0 := by
  norm_num [maxDrawdown, currentDrawdown, drawdowns, runningPeaks,
            drawdownAt, List.zipWith, List.foldl, List.getLast?, crashThenCalm]

theorem the_window_changes_the_answer :
    maxDrawdown crashThenCalm ≠ maxDrawdown (crashThenCalm.drop 2) := by
  norm_num [maxDrawdown, currentDrawdown, drawdowns, runningPeaks,
            drawdownAt, List.zipWith, List.foldl, List.getLast?, crashThenCalm]

end Returns
end Quantify
