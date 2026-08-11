/-
  Cash and position conservation.

  Two theorems, and everything later depends on them. A cadence proof that
  contributed the right number of times is worthless if the money did not
  arrive where it was counted, and a return metric computed over a ledger that
  loses a cent is a precise answer about the wrong portfolio.

  The defects these are aimed at are not hypothetical. This project has already
  shipped a build that reported "$1,000 every year over five years" as $1,000
  contributed — one payment, no refusal, no coverage flag, and ~3,900 tests
  covering no path through the function that turned a schedule into money.
  `endingCash` here is *defined* as the balance the events imply, so a run that
  drops four payments cannot also report a consistent balance.
-/

import Quantify.Types

namespace Quantify
namespace LedgerState

/-- The balance the recorded events imply. -/
def endingCash (s : LedgerState) : Money :=
  s.openingCash + s.contributed - s.withdrawn - s.purchased + s.sold - s.fees

/-- The holding the recorded fills imply, per asset. -/
def endingShares (s : LedgerState) (a : AssetId) : Shares :=
  s.openingShares a + s.bought a - s.disposed a

/-- **Cash conservation.** Nothing enters or leaves the ledger unaccounted.

    Trivial by definition, and deliberately so: the content is in the
    definition of `endingCash`, which names every term money can move through.
    A term omitted there — an unmodelled fee, a dividend the engine credits —
    would make this theorem still true and the model silently wrong, which is
    why `Fixtures.lean` checks the definition against the Python engine rather
    than trusting it. -/
theorem cash_conservation (s : LedgerState) :
    s.endingCash =
      s.openingCash + s.contributed - s.withdrawn - s.purchased + s.sold
        - s.fees := rfl

/-- **Position conservation**, per asset. -/
theorem position_conservation (s : LedgerState) (a : AssetId) :
    s.endingShares a = s.openingShares a + s.bought a - s.disposed a := rfl

/-- A ledger with no events changes nothing. The degenerate case, stated
    because an engine that quietly applied a default contribution to an empty
    plan would satisfy every other theorem here. -/
theorem empty_ledger_is_inert
    (opening : Money) (holdings : AssetId → Shares)
    (h : (0 : Money) ≤ 0) :
    (LedgerState.mk opening holdings [] [] [] [] 0 h).endingCash = opening := by
  simp [endingCash, contributed, withdrawn, purchased, sold, total]

/-- A running total of non-negative amounts never goes below where it started.

    Stated over the accumulator and generalised, because `total` is a `foldl`:
    an induction that fixed the starting value would have to prove the step
    from `0` while the recursive call starts from `0 + x`. -/
theorem foldl_add_nonneg :
    ∀ (xs : List Money) (acc : Money), 0 ≤ acc → (∀ x ∈ xs, 0 ≤ x) →
      acc ≤ xs.foldl (· + ·) acc
  | [],      acc, _,    _   => by simp
  | x :: rest, acc, hacc, hx => by
    have hx0 : 0 ≤ x := hx x (List.mem_cons_self x rest)
    have hrest : ∀ y ∈ rest, 0 ≤ y :=
      fun y hy => hx y (List.mem_cons_of_mem x hy)
    have step : acc + x ≤ rest.foldl (· + ·) (acc + x) :=
      foldl_add_nonneg rest (acc + x) (Int.add_nonneg hacc hx0) hrest
    simp only [List.foldl_cons]
    exact Int.le_trans (Int.le_add_of_nonneg_right hx0) step

/-- Money that only arrives can only increase the balance.

    Rules out a sign error in `contributed`, which `cash_conservation` alone
    cannot see: that theorem holds just as well with the sign flipped, because
    it is true by definition of `endingCash`. This one is not. -/
theorem contributions_do_not_reduce_cash
    (s : LedgerState) (h : s.withdrawals = []) (hb : s.buys = [])
    (hs : s.sells = []) (hf : s.fees = 0) :
    s.openingCash ≤ s.endingCash := by
  have hc : 0 ≤ s.contributed := by
    have := foldl_add_nonneg (s.contributions.map (·.amount)) 0 (Int.le_refl 0) ?_
    · simpa [contributed, total] using this
    · intro x hx
      simp only [List.mem_map] at hx
      obtain ⟨c, _, rfl⟩ := hx
      exact c.nonneg
  simp only [endingCash, h, hb, hs, hf, withdrawn, purchased, sold, total,
             List.map_nil, List.foldl_nil, Int.sub_zero, Int.add_zero]
  exact Int.le_add_of_nonneg_right hc

/-! ## Portfolio valuation

    What the portfolio is worth: cash, plus each holding at its price. Stated
    here rather than in its own module because it is a view of the same
    `LedgerState` the conservation theorems are about, and a valuation computed
    from somewhere else could disagree with the ledger it claims to value.
-/

/-- One holding's worth, in minor units.

    Shares are micro-units and prices are minor units per share, so the product
    is scaled by `sharesScale`. Integer division truncates, which is a rounding
    policy and is stated rather than hidden: a valuation that rounded the other
    way would differ by a cent per holding, and the theorems below are about
    exactly this arithmetic. -/
def holdingValue (shares : Shares) (price : Price) : Money :=
  shares * price / sharesScale

/-- Cash plus every named holding at its price. -/
def portfolioValue (cash : Money) (shares : AssetId → Shares)
    (price : AssetId → Price) : List AssetId → Money
  | []          => cash
  | a :: rest   => holdingValue (shares a) (price a)
                     + portfolioValue cash shares price rest

/-- **A portfolio holding nothing is worth its cash.** The degenerate case,
    stated because a valuation that added a phantom holding would satisfy every
    other theorem here. -/
theorem empty_portfolio_is_its_cash
    (cash : Money) (shares : AssetId → Shares) (price : AssetId → Price) :
    portfolioValue cash shares price [] = cash := rfl

/-- **Valuation is the cash plus the sum of the holdings**, so adding an asset
    adds exactly that asset's worth and nothing else. -/
theorem valuing_one_more_asset
    (cash : Money) (shares : AssetId → Shares) (price : AssetId → Price)
    (a : AssetId) (rest : List AssetId) :
    portfolioValue cash shares price (a :: rest)
      = holdingValue (shares a) (price a)
          + portfolioValue cash shares price rest := rfl

/-- **A holding worth nothing contributes nothing.** Rules out a valuation that
    counted positions rather than value. -/
theorem a_zero_position_adds_nothing
    (cash : Money) (shares : AssetId → Shares) (price : AssetId → Price)
    (a : AssetId) (rest : List AssetId) (h : shares a = 0) :
    portfolioValue cash shares price (a :: rest)
      = portfolioValue cash shares price rest := by
  simp [portfolioValue, holdingValue, h]

end LedgerState
end Quantify
