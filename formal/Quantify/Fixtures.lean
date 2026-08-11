/-
  Canonical ledgers, exported from Python and checked here.

  **The failure this exists for.** Everything in `Ledger.lean` is true by
  definition of `endingCash`, and that is exactly why it is not enough. A term
  omitted from that definition — a fee the engine charges, a dividend it
  credits — leaves both theorems standing while the model quietly describes a
  different ledger from the one Quantify runs. A beautiful proof about the
  wrong semantics is worse than none, because it is quotable.

  So the model is checked against the engine on fixtures the engine produced.
  Lean does not recompute the simulation; it is handed opening balances, the
  events, and the closing balances Python arrived at, and asserts that its own
  definition reaches the same place. A disagreement means one of them is wrong
  and neither side gets to say which — that is a conversation, not a rebuild.

  `#guard` rather than `theorem`: these are decidable checks on concrete
  numbers, and stating them as theorems would suggest a generality one fixture
  does not have.
-/

import Quantify.Ledger
import Quantify.Cadence

namespace Quantify
namespace Fixtures

open LedgerState

/-- Trivial contribution proof for fixture construction. -/
private def c (n : Money) (h : 0 ≤ n := by decide) : Contribution := ⟨n, h⟩
private def w (n : Money) (h : 0 ≤ n := by decide) : Withdrawal := ⟨n, h⟩

private def noHoldings : AssetId → Shares := fun _ => 0

/-- `$100 contributed, one share bought at $20, nothing else.`

    The plan's own first unit case: 100 in, 5 shares out, no residual. Written
    in minor units and micro-shares, so 10000 is $100 and 5000000 is 5 shares.
-/
def oneBuy : LedgerState :=
  { openingCash   := 0
    openingShares := noHoldings
    contributions := [c 10000]
    withdrawals   := []
    buys          := [⟨"VTI", 5000000, 2000, 10000, by decide⟩]
    sells         := []
    fees          := 0
    feesNonneg    := by decide }

#guard oneBuy.endingCash == 0
#guard oneBuy.endingShares "VTI" == 5000000
#guard oneBuy.endingShares "BND" == 0

/-- Five annual contributions of $1,000, none of them spent.

    This is the shape of the defect that started the capability manifest: the
    product reported "$1,000 every year over five years" as $1,000 contributed
    — one payment, no refusal, no coverage flag. A ledger that dropped four
    payments here would fail on the first line rather than on a figure nobody
    checked.
-/
def fiveAnnual : LedgerState :=
  { openingCash   := 0
    openingShares := noHoldings
    contributions := [c 100000, c 100000, c 100000, c 100000, c 100000]
    withdrawals   := []
    buys          := []
    sells         := []
    fees          := 0
    feesNonneg    := by decide }

#guard fiveAnnual.contributed == 500000
#guard fiveAnnual.endingCash == 500000

/-- A round trip with a fee: money in, bought, sold, fee charged. -/
def roundTrip : LedgerState :=
  { openingCash   := 50000
    openingShares := noHoldings
    contributions := [c 10000]
    withdrawals   := [w 2000]
    buys          := [⟨"VTI", 3000000, 2000, 6000, by decide⟩]
    sells         := [⟨"VTI", 1000000, 2500, 2500, by decide⟩]
    fees          := 100
    feesNonneg    := by decide }

-- 50000 + 10000 - 2000 - 6000 + 2500 - 100
#guard roundTrip.endingCash == 54400
#guard roundTrip.endingShares "VTI" == 2000000

/-- Selling more than was bought is representable and *not* prevented here.

    Deliberate. This module states what a ledger means, not what Quantify
    permits: the engine forbids shorting, and that is a capability claim
    belonging in the manifest and in a later phase's theorems. A type that made
    it unrepresentable would also make it impossible to state the theorem that
    it never happens.
-/
def oversold : LedgerState :=
  { openingCash   := 0
    openingShares := noHoldings
    contributions := []
    withdrawals   := []
    buys          := [⟨"VTI", 1000000, 2000, 2000, by decide⟩]
    sells         := [⟨"VTI", 3000000, 2000, 6000, by decide⟩]
    fees          := 0
    feesNonneg    := by decide }

#guard oversold.endingShares "VTI" == -2000000

/-! ## Valuation

The `roundTrip` ledger, priced. Two hundred shares of VTI at 25, plus the cash
the ledger closed at — so the valuation reads the same state the conservation
theorems are about rather than a parallel copy of it.
-/

-- $25, in minor units, matching the prices on the fills above.
private def priced (_ : AssetId) : Price := 2500

private def held : AssetId → Shares
  | "VTI" => 2000000
  | _     => 0

#guard holdingValue (held "VTI") (priced "VTI") == 5000
#guard portfolioValue roundTrip.endingCash held priced ["VTI"] == 59400
#guard portfolioValue roundTrip.endingCash held priced [] == 54400
#guard portfolioValue roundTrip.endingCash held priced ["BND"] == 54400

/-! ## Cadence

The historical case, as a fixture rather than only as a theorem. The theorem
says `N × A` for every schedule; this says which `N` five calendar years of
month-ends actually produce, and it is the number the shipped build got wrong.
-/

open Cadence

/-- Sixty month-ends, five calendar years. -/
def fiveYearsOfMonths : List Date :=
  (List.range 5).flatMap fun (y : Nat) =>
    (List.range 12).map fun (m : Nat) =>
      (⟨2020 + (y : Int), (m : Int) + 1, 28⟩ : Date)

#guard fiveYearsOfMonths.length == 60

-- Annual: five contributions, not one. The defect was one.
#guard contributionCount Cadence.annual fiveYearsOfMonths == 5
#guard totalContributed Cadence.annual fiveYearsOfMonths 100000 == 500000

-- Monthly control: the same sessions, sixty contributions.
#guard contributionCount Cadence.monthly fiveYearsOfMonths == 60
#guard totalContributed Cadence.monthly fiveYearsOfMonths 100000 == 6000000

-- Once: one contribution over the same span.
#guard contributionCount Cadence.once fiveYearsOfMonths == 1
#guard totalContributed Cadence.once fiveYearsOfMonths 100000 == 100000

end Fixtures
end Quantify
