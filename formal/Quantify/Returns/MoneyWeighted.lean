/-
  Money-weighted return: the definition, and when a number may be reported.

  Frozen in `docs/MWR.md` before this file existed. Writing the proof first
  would have formalised whatever the root finder happened to do, and "our
  solver returned a number" is a theorem that looks like success and means
  nothing.

  **This is not the algorithm.** The production solver is an implementation
  that must return a root satisfying the predicate below; it does not decide
  what MWR means. What is proven here is the reporting contract — a rate may be
  published only when it is the unique admissible root — independently of how
  any root is found.

  **Why the daily factor.** The contract is Actual/365 Fixed:

      Σ cashᵢ / (1 + r) ^ (dayᵢ / 365) = 0

  A fractional power of a rational is not rational, so that equation cannot be
  stated over `ℚ` as written. Substituting the daily growth factor `d`, where
  `d ^ 365 = 1 + r`, gives

      Σ cashᵢ / d ^ dayᵢ = 0

  which is the same equation, polynomial in `d`, and exact. The annualised rate
  is recovered as `d ^ 365 - 1`. Reals would also have worked and would have
  brought `Real.rpow` into a file that needs no analysis.
-/

import Mathlib.Data.Rat.Defs
import Mathlib.Algebra.BigOperators.Group.List
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

namespace Quantify
namespace Returns

/-- Money crossing the boundary between investor and portfolio, on a day
    counted from the evaluation start.

    Contributions are negative, withdrawals positive, and the terminal
    portfolio value is a positive flow on the end date. -/
structure DatedFlow where
  day    : Nat
  amount : ℚ
  deriving DecidableEq, Repr

/-- A candidate rate, held as its daily growth factor.

    Positive rather than non-negative: `d = 0` is not a rate, it is a portfolio
    that vanished, and the equation divides by it. -/
structure Rate where
  daily    : ℚ
  positive : 0 < daily

/-- The annualised rate this factor represents, Actual/365 Fixed. -/
def annualised (x : Rate) : ℚ := x.daily ^ 365 - 1

/-- Present value of the flows at a candidate rate. -/
def npv (flows : List DatedFlow) (x : Rate) : ℚ :=
  (flows.map (fun f => f.amount / x.daily ^ f.day)).sum

/-- **A rate solves the series.** The equation, and nothing about whether it
    may be reported. -/
def isMWRRate (flows : List DatedFlow) (x : Rate) : Prop := npv flows x = 0

/-- **And it is the only one that does.** -/
def uniqueMWR (flows : List DatedFlow) (x : Rate) : Prop :=
  isMWRRate flows x ∧ ∀ y : Rate, isMWRRate flows y → y.daily = x.daily

/-- What Quantify publishes. Four cases, and only one carries a number. -/
inductive MWRResult where
  | rate (r : ℚ)
  | noSolution
  | nonUnique
  | insufficientCashFlows
  deriving DecidableEq, Repr

/-- Whether the series is economically capable of having a return: money must
    have gone in and value must have come back. A series of one sign has no
    admissible root and is not a failure of the solver. -/
def hasBothSigns (flows : List DatedFlow) : Prop :=
  (∃ f ∈ flows, f.amount < 0) ∧ (∃ f ∈ flows, 0 < f.amount)

/-- **The reporting contract.** A rate may be published only when it is the
    unique admissible root — never because a solver reached it first.

    Stated as an implication *into* the result, so it constrains what may be
    emitted rather than describing what some function happens to emit. -/
def mayReport (flows : List DatedFlow) (result : MWRResult) : Prop :=
  match result with
  | MWRResult.rate r => ∃ x : Rate, uniqueMWR flows x ∧ annualised x = r
                                     ∧ hasBothSigns flows
  | _ => True

/-- **A published rate is the unique root.** The contract, unfolded — there is
    no path from `mayReport` to a number that skips uniqueness. -/
theorem a_reported_rate_is_the_unique_root
    (flows : List DatedFlow) (r : ℚ) (h : mayReport flows (MWRResult.rate r)) :
    ∃ x : Rate, uniqueMWR flows x ∧ annualised x = r := by
  obtain ⟨x, hu, hr, _⟩ := h
  exact ⟨x, hu, hr⟩

/-- **A second distinct root forbids reporting either.** -/
theorem two_roots_cannot_be_reported
    (flows : List DatedFlow) (x y : Rate)
    (hx : isMWRRate flows x) (hy : isMWRRate flows y)
    (hne : x.daily ≠ y.daily) :
    ¬ ∃ z : Rate, uniqueMWR flows z := by
  rintro ⟨z, _, hall⟩
  exact hne ((hall x hx).trans (hall y hy).symm)

/-! ## Whole-year series

    Every fixture below places flows on year boundaries, where `dayᵢ = 365 * k`
    and the daily factor appears only as `d ^ 365`. That is the annual growth
    factor, so the equation becomes a polynomial in one variable with small
    exponents — exactly computable, where the general form is not.
-/

/-- The same present value, over flows dated in whole years. -/
def npvYears (flows : List (Nat × ℚ)) (g : ℚ) : ℚ :=
  (flows.map (fun p => p.2 / g ^ p.1)).sum

/-- Those flows, as dated ones on an Actual/365 calendar. -/
def dated (flows : List (Nat × ℚ)) : List DatedFlow :=
  flows.map (fun p => ⟨365 * p.1, p.2⟩)

/-- **The annual form is the dated form.** The bridge, so a fixture stated in
    years is about the same object the contract defines.

    Proved by induction rather than by `congr` and a map lemma: the pair
    projections under `List.enum` sent the elaborator to maximum recursion
    depth, and the fact needed at each element is a single `pow_mul`. -/
theorem npvYears_eq_npv :
    ∀ (flows : List (Nat × ℚ)) (x : Rate),
      npvYears flows (x.daily ^ 365) = npv (dated flows) x
  | [],     _ => by simp [npvYears, npv, dated]
  | p :: rest, x => by
    simp only [npvYears, npv, dated, List.map_cons, List.sum_cons]
    rw [← pow_mul]
    exact congrArg _ (npvYears_eq_npv rest x)

/-! ## Fixtures

    Stated over `npvYears` in the annual growth factor `g = d ^ 365`, which the
    bridge above shows is the same equation. The daily factor itself is
    irrational for any pleasant annual rate — `(11/10) ^ (1/365)` is not a
    rational number — so a fixture naming a concrete `Rate` is impossible over
    `ℚ`. That is a property of the calendar convention, not of the model.
-/

/-- Conventional: 100 in, 110 back a year later. -/
def conventional : List (Nat × ℚ) := [(0, -100), (1, 110)]

/-- **Ten percent, and only ten percent.** The root is `g = 11/10`, and the
    equation is linear in `1/g` so there is no second one. -/
theorem conventional_is_ten_percent :
    npvYears conventional (11 / 10) = 0 := by
  norm_num [npvYears, conventional]

theorem conventional_has_no_other_root (g : ℚ) (hg : 0 < g)
    (h : npvYears conventional g = 0) : g = 11 / 10 := by
  simp only [npvYears, conventional, List.map_cons, List.map_nil,
             List.sum_cons, List.sum_nil, pow_zero, pow_one] at h
  field_simp at h
  linarith

/-- **Two admissible roots.** 100 out, 230 back, 132 out again: the sign
    changes twice and the equation is a genuine quadratic in `1/g`.

    Both `11/10` and `6/5` solve it — a ten percent return and a twenty percent
    return, from one series. A solver reporting either would be reporting a
    number the data does not determine. -/
def signChangesTwice : List (Nat × ℚ) := [(0, -100), (1, 230), (2, -132)]

theorem two_rates_solve_it :
    npvYears signChangesTwice (11 / 10) = 0 ∧
    npvYears signChangesTwice (6 / 5) = 0 := by
  constructor <;> norm_num [npvYears, signChangesTwice]

theorem and_they_are_different : (11 : ℚ) / 10 ≠ 6 / 5 := by norm_num

/-- **No admissible root.** Money only ever goes in, so the present value is
    negative at every positive factor and no rate can set it to zero. Not a
    failure of the solver — there is nothing to find. -/
def onlyContributions : List (Nat × ℚ) := [(0, -100), (1, -50)]

theorem no_rate_solves_it (g : ℚ) (hg : 0 < g) :
    npvYears onlyContributions g < 0 := by
  simp only [npvYears, onlyContributions, List.map_cons, List.map_nil,
             List.sum_cons, List.sum_nil, pow_zero, pow_one]
  -- `-50 / g` and `50 / g` are different terms to `linarith`; the negation
  -- has to be pulled out before the two can be related.
  have hpos : (0 : ℚ) < 50 / g := by positivity
  have hneg : (-50 : ℚ) / g = -(50 / g) := by ring
  rw [hneg]
  linarith

/-- And therefore the series has both signs failing too, which is the condition
    that should stop Quantify before a solver is ever called. -/
theorem only_contributions_lacks_a_positive_flow :
    ¬ hasBothSigns (dated onlyContributions) := by
  rintro ⟨_, ⟨f, hf, hpos⟩⟩
  simp only [dated, onlyContributions, List.map_cons, List.map_nil,
             List.mem_cons, List.not_mem_nil, or_false] at hf
  rcases hf with rfl | rfl <;> norm_num at hpos

end Returns
end Quantify
