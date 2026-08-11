/-
  Crossing versus persistent, and why they are not the same instruction.

  The second money-moving defect this project shipped: a condition written as a
  crossing was executed as a persistent state. A portfolio that should have
  bought once bought on every session the condition held, and the figure came
  out confident and wrong.

  **Three layers, kept apart.**

      observations (value against threshold)
        → a predicate at one session
        → the sessions that signal
        → how many

  The threshold is a plain series here, not a moving average. Trigger semantics
  must not be coupled to indicator arithmetic: a later phase proves that a
  moving-average operator *produces* a threshold series correctly, and if these
  theorems depended on how the threshold was computed, that proof would have to
  restate all of them.

  **What this file does not do.** It says nothing about the English phrase
  "crosses below". Discovery owns the mapping from words to
  `crossing_event` or `persistent_condition`, and a Lean file that also claimed
  it would be asserting a fact about language in a place nobody would look for
  one.
-/

import Quantify.Types

namespace Quantify

/-- One session: what the series did, and what it was measured against. -/
structure Observation where
  value     : Int
  threshold : Int
  deriving DecidableEq, Repr

/-- A run of sessions, in order. Position is the session index. -/
abbrev Series := List Observation

namespace Trigger

/-- The state predicate. True whenever the value sits under its threshold. -/
def below (o : Observation) : Bool := o.value < o.threshold

/-- Sessions where the condition *holds*, with the index they occur at. -/
def persistentFrom : Nat → Series → List Nat
  | _, []        => []
  | i, o :: rest =>
      (if below o then [i] else []) ++ persistentFrom (i + 1) rest

/-- Whether this session is the one the condition becomes true on.

    A named function rather than a `let` inside the recursion. A `let`-bound
    condition has no stable name to rewrite on: every proof about it has to
    re-derive the same match, and the second attempt at this file failed on
    exactly that. The same lesson `Cadence.outside` records — one spelling of a
    predicate, everywhere.

    `none` for the first session is a semantic claim, not an off-by-one
    convenience: a crossing is a change, and the first session has nothing to
    have changed from. -/
def fires : Option Observation → Observation → Bool
  | none,   _ => false
  | some p, o => !below p && below o

/-- Sessions where the condition *becomes* true. -/
def crossingFrom : Nat → Option Observation → Series → List Nat
  | _, _,    []        => []
  | i, prev, o :: rest =>
      (if fires prev o then [i] else []) ++ crossingFrom (i + 1) (some o) rest

def persistent (s : Series) : List Nat := persistentFrom 0 s
def crossing   (s : Series) : List Nat := crossingFrom 0 none s

def persistentCount (s : Series) : Nat := (persistent s).length
def crossingCount   (s : Series) : Nat := (crossing s).length

/-- Indices only ever grow along the walk.

    Stated first because the "no crossing at session zero" proof needs it: that
    0 is absent from the tail's signals is a fact about the walk, not an
    assumption. -/
theorem crossingFrom_index_ge :
    ∀ (s : Series) (i : Nat) (prev : Option Observation) (j : Nat),
      j ∈ crossingFrom i prev s → i ≤ j
  | [],        _, _,    _, h => by simp [crossingFrom] at h
  | o :: rest, i, prev, j, h => by
    simp only [crossingFrom, List.append_eq, List.mem_append] at h
    cases h with
    | inl hhere =>
      by_cases hf : fires prev o = true
      · rw [if_pos hf] at hhere
        simp only [List.mem_singleton] at hhere
        omega
      · rw [if_neg hf] at hhere
        simp at hhere
    | inr hrest =>
      have := crossingFrom_index_ge rest (i + 1) (some o) j hrest
      omega

/-- **A crossing is always also a state.** Every session that signals a
    crossing is one where the condition holds.

    Proved over the general recursion rather than the wrapper, so the index
    offset is carried rather than assumed to start at zero. -/
theorem crossingFrom_subset_persistentFrom :
    ∀ (s : Series) (i : Nat) (prev : Option Observation) (j : Nat),
      j ∈ crossingFrom i prev s → j ∈ persistentFrom i s
  | [],        _, _,    _, h => by simp [crossingFrom] at h
  | o :: rest, i, prev, j, h => by
    simp only [crossingFrom, List.append_eq, List.mem_append] at h
    simp only [persistentFrom, List.append_eq, List.mem_append]
    by_cases hf : fires prev o = true
    · -- The crossing fired, so the condition holds here.
      have hbo : below o = true := by
        cases prev with
        | none   => simp [fires] at hf
        | some p => simp only [fires, Bool.and_eq_true] at hf; exact hf.2
      cases h with
      | inl hhere =>
        rw [if_pos hf] at hhere
        exact Or.inl (by rw [if_pos hbo]; exact hhere)
      | inr hrest =>
        exact Or.inr
          (crossingFrom_subset_persistentFrom rest (i + 1) (some o) j hrest)
    · cases h with
      | inl hhere => rw [if_neg hf] at hhere; simp at hhere
      | inr hrest =>
        exact Or.inr
          (crossingFrom_subset_persistentFrom rest (i + 1) (some o) j hrest)

theorem crossing_implies_persistent (s : Series) (j : Nat) :
    j ∈ crossing s → j ∈ persistent s :=
  crossingFrom_subset_persistentFrom s 0 none j

/-- **A crossing needs a predecessor.** The first session of a series never
    signals one, however far under the threshold it opens. -/
theorem no_crossing_without_a_previous_session (s : Series) :
    0 ∉ crossing s := by
  cases s with
  | nil => simp [crossing, crossingFrom]
  | cons o rest =>
    simp only [crossing, crossingFrom, List.append_eq, List.mem_append]
    intro h
    cases h with
    | inl hhere =>
      -- `fires none o` is false by definition: nothing precedes session zero.
      rw [if_neg (by simp [fires])] at hhere
      simp at hhere
    | inr hrest =>
      have := crossingFrom_index_ge rest (0 + 1) (some o) 0 hrest
      omega

/-! ## Non-equivalence, operationally

Saying `crossedBelow ≠ persistentBelow` is true and useless: two predicates
differing somewhere says nothing about how much money moves. What matters is
that one crossing can sit under many persistent sessions, because that is the
ratio by which the defect overspent.
-/

/-- The canonical series. One dip below and back out.

        value      110   95   90   92  105
        threshold  100  100  100  100  100

    One crossing, three sessions below. -/
def oneDip : Series :=
  [⟨110, 100⟩, ⟨95, 100⟩, ⟨90, 100⟩, ⟨92, 100⟩, ⟨105, 100⟩]

/-- **One crossing, three persistent sessions.** The theorem the defect fails.

    Not `1 ≠ 3` in the abstract: these are the counts a real schedule would act
    on, so a build that treated the two as interchangeable would contribute
    three times where the sentence asked for once. -/
theorem crossing_and_persistent_differ_materially :
    crossingCount oneDip = 1 ∧ persistentCount oneDip = 3 := by
  constructor <;> decide

/-- Staying below longer does not signal more crossings. The count is a
    property of the transition, not of the duration. -/
def longDip : Series :=
  [⟨110, 100⟩, ⟨95, 100⟩, ⟨90, 100⟩, ⟨92, 100⟩, ⟨91, 100⟩, ⟨93, 100⟩]

theorem duration_does_not_multiply_crossings :
    crossingCount longDip = 1 ∧ persistentCount longDip = 5 := by
  constructor <;> decide

/-- Leaving and re-entering does signal again. The converse guard: a
    definition that only ever fired once would satisfy every theorem above. -/
def twoDips : Series :=
  [⟨110, 100⟩, ⟨95, 100⟩, ⟨105, 100⟩, ⟨90, 100⟩, ⟨92, 100⟩]

theorem re_entry_signals_again :
    crossingCount twoDips = 2 ∧ persistentCount twoDips = 3 := by
  constructor <;> decide

/-- A series that opens below signals no crossing, and is persistent from the
    first session. The predecessor rule, in numbers. -/
def opensBelow : Series := [⟨90, 100⟩, ⟨92, 100⟩]

theorem opening_below_is_not_a_crossing :
    crossingCount opensBelow = 0 ∧ persistentCount opensBelow = 2 := by
  constructor <;> decide

end Trigger
end Quantify
