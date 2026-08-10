/-
  Which events are allowed to count.

  Causality says when an event *may* execute. This says which executions are
  allowed to affect what the strategy reports — and the two are different
  questions with different answers for the same session.

      requested   what the person asked to evaluate
      warm-up     sessions before it, present because an indicator needs them
      frame       warm-up ++ requested: everything the engine loads
      reported    what the result is computed from

  The rule, in one line:

      Data may exist before the evaluation period because computation needs it.
      Economic events may not.

  That is the old "three months returned ten years" defect stated precisely. It
  is not that the engine loaded too much history — it has to. It is that money
  which moved during the warm-up was counted in a result the person asked to be
  about three months.
-/

import Quantify.Types

namespace Quantify

/-- The sessions a run may look at, and the subrange it must report on.

    `warmUp` counts sessions before `reported` begins. Held as a count rather
    than as a second range because that is what an indicator asks for — "two
    hundred sessions of history" — and deriving one from the other is where the
    off-by-one lives. -/
structure Window where
  warmUp        : Nat
  reportedFirst : Nat
  reportedLast  : Nat
  deriving DecidableEq, Repr

namespace Window

/-- The first session the engine loads. Earlier than the reported period by
    exactly the warm-up, and never below zero. -/
def frameFirst (w : Window) : Nat := w.reportedFirst - w.warmUp

/-- Whether a session may be used to compute an indicator. -/
def inFrame (w : Window) (session : Nat) : Bool :=
  w.frameFirst ≤ session && session ≤ w.reportedLast

/-- Whether a session's economic events may reach the result. -/
def inReported (w : Window) (session : Nat) : Bool :=
  w.reportedFirst ≤ session && session ≤ w.reportedLast

/-- A session the engine loads but must not count from. -/
def isWarmUp (w : Window) (session : Nat) : Bool :=
  inFrame w session && !inReported w session

/-- **Reported is inside the frame.** The engine never reports on a session it
    did not load. -/
theorem reported_within_frame (w : Window) (session : Nat)
    (h : inReported w session = true) : inFrame w session = true := by
  simp only [inReported, inFrame, Bool.and_eq_true, decide_eq_true_eq] at *
  exact ⟨by simp [frameFirst]; omega, h.2⟩

/-- **Warm-up is in the frame and out of the report.** Both halves, because
    either alone permits the defect: a warm-up outside the frame cannot feed an
    indicator, and one inside the report is money counted twice. -/
theorem warm_up_is_loaded_and_not_reported (w : Window) (session : Nat)
    (h : isWarmUp w session = true) :
    inFrame w session = true ∧ inReported w session = false := by
  simp only [isWarmUp, Bool.and_eq_true, Bool.not_eq_true'] at h
  exact ⟨h.1, h.2⟩

/-- **Nothing is both.** The two ranges partition the frame. -/
theorem warm_up_and_reported_are_disjoint (w : Window) (session : Nat) :
    ¬(isWarmUp w session = true ∧ inReported w session = true) := by
  intro ⟨hw, hr⟩
  simp only [isWarmUp, Bool.and_eq_true, Bool.not_eq_true'] at hw
  rw [hr] at hw
  simp at hw

/-- Contributions that may be counted: the ones inside the reported period. -/
def reportable (w : Window) (sessions : List Nat) : List Nat :=
  sessions.filter (inReported w)

/-- **No event outside the requested period is reportable.** -/
theorem reportable_excludes_everything_outside (w : Window)
    (sessions : List Nat) (session : Nat)
    (h : session ∈ reportable w sessions) : inReported w session = true := by
  simpa using (List.mem_filter.mp h).2

/-- **And every reportable session was loaded.** Together with the theorem
    above this is the whole boundary: reportable ⊆ reported ⊆ frame. -/
theorem reportable_is_in_the_frame (w : Window) (sessions : List Nat)
    (session : Nat) (h : session ∈ reportable w sessions) :
    inFrame w session = true :=
  reported_within_frame w session
    (reportable_excludes_everything_outside w sessions session h)

/-! ## The three claims at once

A window is only interesting when the warm-up genuinely matters. A fixture
whose indicator ignores the earlier sessions would satisfy every theorem above
while proving nothing: of course nothing leaked, nothing was there.

So this one is built so that all three hold together —

    warm-up changes the indicator      ✓
    warm-up contributes no money       ✓
    warm-up produces no reported fill  ✓

and the mutation makes exactly one warm-up event leak while the indicator keeps
computing correctly.
-/

/-- Three sessions of history before a three-session report. -/
def threeMonths : Window := ⟨3, 3, 5⟩

#guard threeMonths.frameFirst == 0

-- Sessions 0..2 are loaded and silent; 3..5 are reported.
#guard threeMonths.isWarmUp 0 == true
#guard threeMonths.isWarmUp 2 == true
#guard threeMonths.isWarmUp 3 == false
#guard threeMonths.inFrame 0 == true
#guard threeMonths.inReported 0 == false
#guard threeMonths.inReported 3 == true
#guard threeMonths.inReported 5 == true
#guard threeMonths.inReported 6 == false

/-- Money moved on every session of the frame, warm-up included. -/
def everySession : List Nat := [0, 1, 2, 3, 4, 5]

-- **Only the requested period counts.** Six sessions of activity, three
-- reportable. The defect reported all six and called it three months.
#guard (threeMonths.reportable everySession).length == 3
#guard threeMonths.reportable everySession == [3, 4, 5]

/-- A mean over the frame, which is what an indicator does: it needs the
    warm-up and would be a different number without it.

    Present so the fixture cannot be satisfied by a window that simply ignores
    early data. `values` are sums in minor units; the point is only that the
    two averages differ. -/
def meanOver (values : List Int) : Int :=
  match values.length with
  | 0     => 0
  | n + 1 => values.foldl (· + ·) 0 / (n + 1 : Int)

def frameValues    : List Int := [100, 100, 100, 400, 400, 400]
def reportedValues : List Int := [400, 400, 400]

/-- **The warm-up changes the indicator.** 250 against 400 — so this window's
    early sessions are doing real work, and the silence proven above is a
    silence about *money*, not about data. -/
theorem warm_up_changes_the_indicator :
    meanOver frameValues = 250 ∧ meanOver reportedValues = 400 := by
  constructor <;> decide

end Window
end Quantify
