/-
  Scheduled contributions: how many, and how much.

  This is the first slice where the formal layer proves a *money-moving* defect
  impossible rather than restating a ledger identity. The defect is not
  hypothetical. This build once reported "$1,000 every year over five years" as
  $1,000 contributed — one payment, no refusal, no coverage flag, and ~3,900
  tests covering no path through the function that turned a schedule into money.

  **Calendar eligibility is kept apart from amount arithmetic.** `bucket` says
  which period a date falls in, `eligible` picks one date per period, and only
  then does `totalContributed` multiply. Later phases have to formalise first
  trading days, rolled dates and holidays; all of that changes `bucket` and
  `eligible` and leaves the accounting proof untouched.

  **The count theorem links two independent definitions.** `eligible` walks the
  sessions and keeps one per period; `periods` maps to keys and dedups them.
  Neither is defined in terms of the other, so `contributions_are_one_per_period`
  is a real claim rather than a restatement — which is exactly what makes the
  historical mutation (annual → the first date only) fail it.
-/

import Quantify.Types

namespace Quantify

/-- A calendar date. Days are carried but never bucketed on: no cadence here
    is finer than a month, and a `Day` cadence would need sessions rather than
    dates because markets are shut at weekends. -/
structure Date where
  year  : Int
  month : Int
  day   : Int
  deriving DecidableEq, Repr

/-- The scheduled cadences this build executes, and only those.

    `Weekly`, `Biweekly` and `Quarterly` are in the capability manifest and
    absent here on purpose: a cadence in this file is one whose accounting is
    proven, and listing an unproven constructor would let a later theorem quantify
    over it vacuously. -/
inductive Cadence where
  | once
  | monthly
  | annual
  deriving DecidableEq, Repr

namespace Cadence

/-- Which period a date belongs to, as a single key.

    Months are `year * 12 + month` rather than a pair so periods compare with
    one `Int` equality; `once` collapses every date to one period, which is
    what "once" means and why it needs no special case below. -/
def bucket : Cadence → Date → Int
  | once,    _ => 0
  | monthly, d => d.year * 12 + d.month
  | annual,  d => d.year

/-- "This session is not in period `k`", as a `Bool`.

    Named and Bool-valued rather than written as `fun e => bucket c e ≠ k` at
    each site. A `Prop` inequality normalises to `!decide (· = ·)` inside a
    `filter`, and the induction hypothesis then carries one spelling while the
    goal carries the other — two forms of the same predicate that no `simp`
    lemma relates. One definition, one spelling, everywhere. -/
def outside (c : Cadence) (k : Int) (e : Date) : Bool := bucket c e != k

/-- The distinct periods these sessions cover. -/
def periods (c : Cadence) : List Date → List Int
  | []      => []
  | d :: rest =>
      bucket c d :: periods c (rest.filter (outside c (bucket c d)))
  termination_by s => s.length
  decreasing_by
    simp_wf
    exact Nat.lt_succ_of_le (List.length_filter_le _ _)

/-- The dates money actually lands on: the first session of each period.

    Defined by walking the sessions, not by consulting `periods`. The two must
    stay independent or the theorem relating them says nothing. -/
def eligible (c : Cadence) : List Date → List Date
  | []      => []
  | d :: rest =>
      d :: eligible c (rest.filter (outside c (bucket c d)))
  termination_by s => s.length
  decreasing_by
    simp_wf
    exact Nat.lt_succ_of_le (List.length_filter_le _ _)

/-- How many contributions a schedule makes. -/
def contributionCount (c : Cadence) (sessions : List Date) : Nat :=
  (eligible c sessions).length

/-- What it moves, at a fixed amount per contribution. -/
def totalContributed (c : Cadence) (sessions : List Date) (amount : Money) :
    Money := (contributionCount c sessions : Int) * amount

/-- Bucketing a filtered list is filtering the bucketed one.

    The bridge between `eligible`, which filters dates, and `periods`, which
    filters keys. Without it the two recursions cannot be lined up. -/
theorem map_bucket_filter (c : Cadence) (k : Int) :
    ∀ (xs : List Date),
      (xs.filter (outside c k)).map (bucket c)
        = (xs.map (bucket c)).filter (fun b => b != k)
  | [] => by simp
  | x :: rest => by
    have ih := map_bucket_filter c k rest
    by_cases h : bucket c x = k <;>
      simp [List.filter_cons, outside, h, ih]

/-- **One contribution per period.** The count is the number of distinct
    periods the sessions cover — no more, and no fewer.

    This is the theorem the historical defect fails. A schedule that
    contributed only on the first eligible date would give a count of 1 against
    five distinct years, and there is no reading of the two sides that makes
    them agree. -/
theorem eligible_length_eq_periods (c : Cadence) :
    ∀ (sessions : List Date),
      (eligible c sessions).length = (periods c sessions).length
  | [] => by simp [eligible, periods]
  | d :: rest => by
    have ih := eligible_length_eq_periods c
      (rest.filter (outside c (bucket c d)))
    simp only [eligible, periods, List.length_cons, ih]
  termination_by s => s.length
  decreasing_by
    simp_wf
    exact Nat.lt_succ_of_le (List.length_filter_le _ _)

theorem contributions_are_one_per_period (c : Cadence) (sessions : List Date) :
    contributionCount c sessions = (periods c sessions).length :=
  eligible_length_eq_periods c sessions

/-- **Total contributed is N × A**, for every scheduled cadence.

    Stated over `periods` rather than over the count, so it says something
    about the calendar rather than about `eligible`'s own length. -/
theorem total_is_periods_times_amount
    (c : Cadence) (sessions : List Date) (amount : Money) :
    totalContributed c sessions amount
      = ((periods c sessions).length : Int) * amount := by
  simp [totalContributed, contributions_are_one_per_period c sessions]

/-- **Once means once.** At most one contribution, whatever the sessions. -/
theorem once_empties_the_tail (d : Date) (rest : List Date) :
    rest.filter (outside once (bucket once d)) = [] := by
  induction rest with
  | nil => simp
  | cons x xs ih => simp [List.filter_cons, outside, bucket, ih]

theorem once_contributes_exactly_once (d : Date) (rest : List Date) :
    contributionCount once (d :: rest) = 1 := by
  simp [contributionCount, eligible, once_empties_the_tail d rest]

theorem once_contributes_at_most_once (sessions : List Date) :
    contributionCount once sessions ≤ 1 := by
  cases sessions with
  | nil => simp [contributionCount, eligible]
  | cons d rest => simp [once_contributes_exactly_once d rest]

/-- Every period key came from a session. Needed by the distinctness proof:
    without it a key could appear from nowhere and dodge the contradiction. -/
theorem periods_subset (c : Cadence) :
    ∀ (sessions : List Date) {k : Int},
      k ∈ periods c sessions → k ∈ sessions.map (bucket c)
  | [], _, h => by simp [periods] at h
  | d :: rest, k, h => by
    simp only [periods, List.mem_cons] at h
    cases h with
    | inl hk => simp [hk]
    | inr hk =>
      have := periods_subset c (rest.filter (outside c (bucket c d))) hk
      rw [map_bucket_filter] at this
      simp only [List.mem_filter] at this
      simp [this.1]
  termination_by sessions => sessions.length
  decreasing_by
    simp_wf
    exact Nat.lt_succ_of_le (List.length_filter_le _ _)

/-- No period is contributed to twice. `N × A` for the wrong `N` is still
    wrong, and the count theorem alone cannot see it. -/
theorem periods_are_distinct (c : Cadence) :
    ∀ (sessions : List Date), (periods c sessions).Nodup
  | [] => by simp [periods]
  | d :: rest => by
    simp only [periods, List.nodup_cons]
    refine ⟨?_, periods_are_distinct c _⟩
    intro mem
    have := periods_subset c (rest.filter (outside c (bucket c d))) mem
    simp only [List.mem_map] at this
    obtain ⟨e, he, heq⟩ := this
    have hne : outside c (bucket c d) e = true := (List.mem_filter.mp he).2
    simp only [outside, bne_iff_ne, ne_eq] at hne
    exact hne heq
  termination_by sessions => sessions.length
  decreasing_by
    simp_wf
    exact Nat.lt_succ_of_le (List.length_filter_le _ _)

end Cadence
end Quantify
