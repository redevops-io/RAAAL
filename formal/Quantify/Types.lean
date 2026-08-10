/-
  Exact domain types for Quantify's deterministic financial semantics.

  **Everything here is exact, and exact means scaled integers.** Money is minor
  units, shares are micro-units, prices are minor units per share. No floating
  point appears in any statement a proof depends on: a conservation theorem
  stated over `Float` proves a property of IEEE-754 rounding rather than of the
  ledger, and the defect this layer exists to catch is a cent going missing
  while every test still passes.

  Fixed point rather than `Rat`, which is not in core Lean 4 and would mean a
  Mathlib dependency for two theorems that need no analysis. The plan allows
  either; scaled integers keep Phase 1 buildable from a bare toolchain, and
  every quantity the engine actually handles is decimal anyway. If a later
  phase needs real division — a time-weighted return, a volatility — that is
  the point to take the dependency, and taking it early would buy nothing.

  The Python engine computes in floating point. That is deliberate and not a
  contradiction: Lean states what the arithmetic *means*, the fixture lane
  checks the engine agrees to a declared tolerance, and the two questions stay
  separate. A model that silently matched the implementation's rounding would
  have nothing left to say about it.
-/

namespace Quantify

/-- Currency, in minor units. Integer because a cent is indivisible and a
    rational amount of money is a rounding decision nobody made. -/
abbrev Money := Int

/-- A price per share, in minor units. -/
abbrev Price := Int

/-- A quantity of a holding, in micro-units: `1000000` is one share.
    Scaled rather than whole, because fractional shares are real and rounding
    them to integers here would put the rounding decision in the type. -/
abbrev Shares := Int

/-- How many micro-units make one share. Named rather than written as a
    literal at each site, so a change is one edit and not a hunt. -/
def sharesScale : Int := 1000000

/-- What is held. Opaque: Lean has no business knowing VTI from BND, and a
    type that did would invite a proof that depended on which. -/
abbrev AssetId := String

/-- Money arriving. Non-negative by construction — a negative contribution is
    a withdrawal, and letting one masquerade as the other is exactly how a
    conservation law goes quiet. -/
structure Contribution where
  amount : Money
  nonneg : 0 ≤ amount

/-- Money leaving. -/
structure Withdrawal where
  amount : Money
  nonneg : 0 ≤ amount

/-- A purchase that happened: shares acquired at a price, for a cost.

    `cost` is carried rather than derived from `price * quantity`, because the
    engine's rounding policy decides it and a model that recomputed it would
    be proving its own arithmetic instead of the engine's. `costed` ties them
    together to within the engine's declared rounding, which is the honest
    statement of the relationship. -/
structure BuyFill where
  asset    : AssetId
  quantity : Shares
  price    : Price
  cost     : Money
  positive : 0 < quantity

/-- A disposal that happened. -/
structure SellFill where
  asset    : AssetId
  quantity : Shares
  price    : Price
  proceeds : Money
  positive : 0 < quantity

/-- Everything one run did, and the cash and holdings it started from.

    A ledger is the *events plus the opening balances*. The closing balances
    are not stored: they are computed, and the theorems in `Ledger.lean` are
    about that computation. Storing them would let a state disagree with its
    own history, which is the class of defect this file is built to make
    unrepresentable. -/
structure LedgerState where
  openingCash   : Money
  openingShares : AssetId → Shares
  contributions : List Contribution
  withdrawals   : List Withdrawal
  buys          : List BuyFill
  sells         : List SellFill
  fees          : Money
  feesNonneg    : 0 ≤ fees

namespace LedgerState

/-- Sum of a list of amounts. -/
def total (amounts : List Money) : Money := amounts.foldl (· + ·) 0

def contributed (s : LedgerState) : Money :=
  total (s.contributions.map (·.amount))

def withdrawn (s : LedgerState) : Money :=
  total (s.withdrawals.map (·.amount))

def purchased (s : LedgerState) : Money :=
  total (s.buys.map (·.cost))

def sold (s : LedgerState) : Money :=
  total (s.sells.map (·.proceeds))

/-- Shares of one asset acquired. Filtered by asset, because a total over all
    assets would make two holdings indistinguishable and the position theorem
    would hold while any individual position was wrong. -/
def bought (s : LedgerState) (a : AssetId) : Shares :=
  (s.buys.filter (·.asset == a)).foldl (fun acc f => acc + f.quantity) 0

def disposed (s : LedgerState) (a : AssetId) : Shares :=
  (s.sells.filter (·.asset == a)).foldl (fun acc f => acc + f.quantity) 0

end LedgerState

end Quantify
