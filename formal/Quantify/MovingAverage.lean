/-
  The threshold series, and nothing about signals.

      price series
        → movingAverage n
        → threshold series          consumed by `Triggers.lean`

  Kept apart deliberately. `Triggers` proves what a crossing means, `Window`
  proves which sessions may count, `Ordering` proves when a fill may happen,
  and this proves the number they compare against is the right number. One
  theorem spanning all four would be true, unreadable, and impossible to
  diagnose when it broke.

  **The window is a count of observations, not a span of days.** A 200-day
  moving average over a series of sessions is 200 sessions; weekends and
  holidays are absent from the series before this operator sees it. Conflating
  the two is how an off-by-one becomes a plausible-looking number.
-/

import Quantify.Types

namespace Quantify

namespace MA

/-- The observations a moving average at session `t` is computed from: the
    `n` ending at `t`, inclusive.

    `none` before enough history exists. Not zero, and not a shorter average —
    a mean over four observations is a different statistic from a mean over
    five, and returning one where the other was asked for is the defect this
    file exists to make impossible. -/
def windowAt (n : Nat) (prices : List Int) (t : Nat) : Option (List Int) :=
  if t + 1 < n ∨ n = 0 then none
  else some ((prices.drop (t + 1 - n)).take n)

/-- The average of those observations, in minor units, truncating toward zero
    exactly as the engine's integer arithmetic does. -/
def movingAverage (n : Nat) (prices : List Int) (t : Nat) : Option Int :=
  match windowAt n prices t with
  | none    => none
  | some xs => if xs.length = n then some (xs.foldl (· + ·) 0 / (n : Int))
               else none

/-- **No value before the warm-up is complete.** -/
theorem undefined_before_enough_history
    (n : Nat) (prices : List Int) (t : Nat) (h : t + 1 < n) :
    movingAverage n prices t = none := by
  simp [movingAverage, windowAt, h]

/-- **The first defined session is exactly `n - 1`**, given enough prices.
    One earlier is undefined; that one is not. -/
theorem first_value_is_at_n_minus_one
    (n : Nat) (prices : List Int) (hn : 0 < n) (hlen : n ≤ prices.length) :
    movingAverage n prices (n - 1) ≠ none := by
  have hnot : ¬((n - 1) + 1 < n ∨ n = 0) := by omega
  simp only [movingAverage, windowAt, if_neg hnot]
  have : ((prices.drop ((n - 1) + 1 - n)).take n).length = n := by
    simp [Nat.sub_self]
    omega
  simp [this]

/-- **The window holds exactly `n` observations** wherever it is defined. -/
theorem window_length (n : Nat) (prices : List Int) (t : Nat)
    (xs : List Int) (h : windowAt n prices t = some xs)
    (hlen : t < prices.length) : xs.length = n := by
  simp only [windowAt] at h
  split at h
  · exact absurd h (by simp)
  · rename_i hnot
    injection h with h
    subst h
    simp
    omega

/-- **Only the observations inside the window can matter.** Two series whose
    windows at `t` coincide have the same average there, whatever they do
    elsewhere.

    Stated over the window rather than over indices because that is what the
    operator reads: a price outside it is not consulted, and a theorem phrased
    on indices would be proving a fact about list addressing instead. -/
theorem depends_only_on_its_window
    (n : Nat) (a b : List Int) (t : Nat)
    (h : windowAt n a t = windowAt n b t) :
    movingAverage n a t = movingAverage n b t := by
  simp only [movingAverage, h]

/-- Summing a flat run. Generalised over the accumulator, because `foldl`
    carries one and an induction that fixed it at zero cannot take the step. -/
theorem foldl_replicate (c : Int) :
    ∀ (k : Nat) (acc : Int),
      (List.replicate k c).foldl (· + ·) acc = acc + (k : Int) * c
  | 0,     acc => by simp
  | k + 1, acc => by
    simp only [List.replicate_succ, List.foldl_cons, foldl_replicate c k]
    -- `ring` is a Mathlib tactic and this build has no Mathlib. Distributing
    -- by hand leaves `k * c` as an atom, which `omega` can then treat
    -- linearly.
    push_cast
    rw [Int.add_mul, Int.one_mul]
    omega

/-- **A flat series averages to its own level**, at every defined session. -/
theorem constant_series_averages_to_the_constant
    (n : Nat) (c : Int) (t : Nat) (hn : 0 < n) (hfit : n ≤ t + 1) :
    movingAverage n (List.replicate (t + 1) c) t = some c := by
  have hnot : ¬(t + 1 < n ∨ n = 0) := by omega
  simp only [movingAverage, windowAt, if_neg hnot]
  have hw : ((List.replicate (t + 1) c).drop (t + 1 - n)).take n
              = List.replicate n c := by
    rw [List.drop_replicate, List.take_replicate]
    congr 1
    omega
  rw [hw]
  simp only [List.length_replicate, if_pos rfl, foldl_replicate c n 0,
             Int.zero_add]
  have hn0 : (n : Int) ≠ 0 := by exact_mod_cast Nat.pos_iff_ne_zero.mp hn
  rw [Int.mul_ediv_cancel_left c hn0]
  simp

/-! ## Where the window bites

The theorems above say a moving average reads only its own window. These show
that the window is the one asked for — an off-by-one satisfies every theorem
above while computing a different statistic.
-/

/-- Ten sessions. The last three are 400; everything before is 100. -/
def series : List Int :=
  [100, 100, 100, 100, 100, 100, 100, 400, 400, 400]

-- A 3-session average at the last session sees only the 400s.
#guard movingAverage 3 series 9 == some 400

-- A 5-session average at the same point sees two 100s as well.
#guard movingAverage 5 series 9 == some 280

/-- **Window length is not a detail.** Two averages over one series at one
    session, and they are different numbers — so a strategy written against one
    and executed with the other is a different strategy. -/
theorem window_length_changes_the_threshold :
    movingAverage 3 series 9 ≠ movingAverage 5 series 9 := by decide

/-- **Off by one is a different statistic** — where the extra observation
    differs.

    Asserted at session 8, not 9, and the reason is the finding. At session 9
    the window sits wholly inside a flat run of 400s, so a 2-, 3- or 4-session
    average are all exactly 400 and an off-by-one is *invisible*. `decide`
    disproved the first version of this theorem, which had claimed otherwise.

    That is not a fixture detail. It is why a 200-day average computed over 199
    observations can run for months looking correct: on quiet stretches it
    agrees with itself, and it only diverges where the boundary observation is
    different from the rest — which is exactly where a crossing happens. -/
theorem off_by_one_is_a_different_average :
    movingAverage 3 series 8 = some 300 ∧
    movingAverage 2 series 8 = some 400 ∧
    movingAverage 4 series 8 = some 250 := by
  refine ⟨?_, ?_, ?_⟩ <;> decide

/-- And the invisibility itself, stated rather than left as a gap: on a flat
    stretch the wrong window gives the right answer. -/
theorem off_by_one_hides_on_a_flat_stretch :
    movingAverage 3 series 9 = movingAverage 2 series 9 := by decide

/-- Changing a price *outside* the window leaves the average alone. -/
def changedOutside : List Int :=
  [999, 100, 100, 100, 100, 100, 100, 400, 400, 400]

theorem a_price_outside_the_window_cannot_move_it :
    movingAverage 3 changedOutside 9 = movingAverage 3 series 9 := by decide

/-- Changing one *inside* it does. Without this the theorem above is satisfied
    by an operator that ignores its input entirely. -/
def changedInside : List Int :=
  [100, 100, 100, 100, 100, 100, 100, 400, 400, 700]

theorem a_price_inside_the_window_moves_it :
    movingAverage 3 changedInside 9 ≠ movingAverage 3 series 9 := by decide

-- Nothing before the warm-up completes, and something exactly at it.
#guard movingAverage 5 series 3 == none
#guard movingAverage 5 series 4 == some 100

end MA
end Quantify
